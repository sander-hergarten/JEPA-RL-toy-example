"""Offline JEPA pretraining on a saved dataset: MSE latent prediction + explicit anti-collapse.

    python -m atari_jepa.pretrain --dataset data/breakout --out runs/pretrain_breakout --updates 50000

Phase 2 of "learn by observing": the encoder and the action-conditioned dynamics are trained *only* to
predict future latents on frames an RL agent already collected. No reward, no value, no environment
interaction, and a fixed data distribution — so representation collapse can be debugged on its own.

Differences from the online objective:

* the latent loss is **MSE** by default rather than cosine distance, so the scale of the latent is
  constrained too (and a constant latent becomes an optimum that must be excluded explicitly),
* anti-collapse is an explicit term: **SIGReg** (random-projection isotropic-Gaussian regularization,
  LeJEPA-style) by default, or VICReg, or the per-dimension variance floor used online.

The result is a checkpoint whose encoder/dynamics can initialize an RL run
(`train.init_from`, with `train.freeze_encoder` for the strict "observation only" test).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .agent import WorldModel
from .checkpoint import save_checkpoint
from .collect import load_dataset
from .config import Config
from .losses import anti_collapse, cosine_distance, jepa_distances, masked_mean, mse_latent_distance, unroll
from .utils import JsonlWriter, configure_threads, runtime_versions, seed_everything, select_device, write_json


def pretrain_losses(model: WorldModel, batch, cfg) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """K-step latent prediction on real action sequences, plus the anti-collapse term."""
    obs, actions = batch.observations, batch.actions
    B, K = actions.shape
    valid, term = batch.valid.bool(), batch.terminated.bool()
    latent_mask = valid & ~term

    with torch.no_grad():
        target_z = model.target_encoder(obs[:, 1:].flatten(0, 1)).unflatten(0, (B, K))
        target_z0 = model.target_encoder(obs[:, 0]) if cfg.jepa_target == "delta" else None

    z0 = model.encoder(obs[:, 0])
    z_hat = unroll(model.dynamics, z0, actions, K)
    if cfg.latent_loss == "mse":
        dist = mse_latent_distance(z_hat, target_z, target_z0, cfg.jepa_target)
    else:
        dist = jepa_distances(z_hat, target_z, target_z0, cfg.jepa_target, 1e-8)
    l_jepa = masked_mean(dist, latent_mask)

    z_root = z0.flatten(1)
    l_anti = anti_collapse(cfg.anti_collapse, z_root, cfg.variance_floor, 1e-4, cfg.sigreg_directions)
    total = cfg.lambda_jepa * l_jepa + cfg.lambda_anti * l_anti

    metrics: dict[str, torch.Tensor] = {"loss_total": total, "loss_jepa": l_jepa, "loss_anti": l_anti}
    with torch.no_grad():
        for k in range(K):
            metrics[f"jepa_d{k + 1}"] = masked_mean(dist[:, k], latent_mask[:, k])
        persist = [z0] * (K + 1)
        base = (mse_latent_distance(persist, target_z, target_z0, cfg.jepa_target) if cfg.latent_loss == "mse"
                else jepa_distances(persist, target_z, target_z0, cfg.jepa_target, 1e-8))
        metrics["persist_d1"] = masked_mean(base[:, 0], latent_mask[:, 0])
        std = z_root.std(0, unbiased=False)
        metrics["latent_std_mean"] = std.mean()
        metrics["latent_rms"] = z_root.pow(2).mean().sqrt()
        zc = z_root - z_root.mean(0, keepdim=True)
        cov_eig = torch.linalg.eigvalsh((zc @ zc.T) / max(B - 1, 1)).clamp(min=0)
        p = cov_eig / cov_eig.sum().clamp(min=1e-12)
        # NOTE: computed from the B x B Gram matrix, so this is capped at the batch size, not the 3136
        # latent dimensions. It is a cheap collapse alarm during training; atari_jepa.embeddings
        # measures the real effective rank over thousands of held-out roots.
        metrics["batch_effective_rank"] = torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum())
        zn = torch.nn.functional.normalize(z_root, dim=1)
        sim = zn @ zn.T
        metrics["pairwise_cos"] = (sim.sum() - sim.diagonal().sum()) / max(B * (B - 1), 1)
        zcn = torch.nn.functional.normalize(zc, dim=1)  # after removing the shared component
        csim = zcn @ zcn.T
        metrics["centered_pairwise_cos"] = (csim.sum() - csim.diagonal().sum()) / max(B * (B - 1), 1)
    return total, metrics


def run_pretrain(cfg: Config, dataset_dir: str, out_dir: Path, device: torch.device) -> dict[str, Any]:
    replay, meta, dataset_cfg = load_dataset(dataset_dir)
    pre = cfg.pretrain
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    rng = np.random.default_rng([cfg.seed, 11])
    num_actions = len(meta["env"]["action_meanings"])
    model = WorldModel(cfg, num_actions).to(device)
    params = list(model.encoder.parameters()) + list(model.dynamics.parameters())
    opt = torch.optim.Adam(params, lr=pre.lr, eps=pre.adam_eps)
    log = JsonlWriter(out_dir / "pretrain.jsonl")
    roots = replay.valid_roots()
    print(f"dataset {dataset_dir}: {replay.n_written} frames, {len(roots)} valid roots, "
          f"{meta['episodes']} episodes, collected by {meta['policy']['source']} "
          f"(return {meta['return_mean']:+.1f})")
    print(f"pretraining {pre.updates} updates: {pre.latent_loss} latent loss, target {pre.jepa_target}, "
          f"anti-collapse {pre.anti_collapse} (weight {pre.lambda_anti}) on {device}")

    start = time.perf_counter()
    agg: dict[str, list[float]] = {}
    for step in range(1, pre.updates + 1):
        batch = replay.sample(pre.batch_size, pre.rollout_steps, rng).to_torch(device)
        total, metrics = pretrain_losses(model, batch, pre)
        opt.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, pre.grad_clip_norm)
        opt.step()
        with torch.no_grad():  # EMA target encoder, same convention as the online trainer
            t, o = list(model.target_encoder.parameters()), list(model.encoder.parameters())
            torch._foreach_mul_(t, pre.tau)
            torch._foreach_add_(t, o, alpha=1.0 - pre.tau)
        values = torch.stack([v.detach().float() for v in metrics.values()] + [grad_norm.detach().float()]).tolist()
        row = dict(zip(list(metrics.keys()) + ["grad_norm"], values))
        if not np.isfinite(row["loss_total"]):
            raise FloatingPointError(f"non-finite pretrain loss at update {step}: {row}")
        for k, v in row.items():
            agg.setdefault(k, []).append(v)
        if step % pre.log_every == 0 or step == pre.updates:
            record = {k: float(np.mean(v)) for k, v in agg.items()}
            record.update({"update": step, "wall_time_s": time.perf_counter() - start})
            log.write(record)
            print(f"  update {step}: jepa={record['loss_jepa']:.4f} (persist d1 {record['persist_d1']:.4f}) "
                  f"anti={record['loss_anti']:.4f} rank={record['batch_effective_rank']:.1f}/{cfg.pretrain.batch_size} "
                  f"cos={record['pairwise_cos']:.3f} rms={record['latent_rms']:.3f}", flush=True)
            agg.clear()

    counters = {"decisions": 0, "updates": pre.updates, "episodes": 0, "train_emulator_frames": 0,
                "sampled_sequences": pre.updates * pre.batch_size, "sampled_valid_transitions": 0,
                "sampled_latent_targets": 0, "eval_decisions": 0, "eval_emulator_frames": 0,
                "warmup_until": 0, "resume_count": 0, "wall_time_s": time.perf_counter() - start}
    save_checkpoint(out_dir / "checkpoint.pt", {
        "config": cfg.to_dict(), "variant": f"pretrain_{pre.latent_loss}_{pre.anti_collapse}",
        "model": model.state_dict(), "optimizer": opt.state_dict(), "counters": counters,
        "schedule": {}, "rng": {"torch": torch.get_rng_state(), "python": None,
                                "explore": None, "replay": rng.bit_generator.state},
        "env": meta["env"], "replay_n_written": None, "pretrain": True,
        "dataset": {"path": str(dataset_dir), **{k: meta[k] for k in ("decisions", "episodes", "return_mean")}},
    })
    summary = {"dataset": dataset_dir, "updates": pre.updates, "wall_time_s": counters["wall_time_s"],
               "latent_loss": pre.latent_loss, "anti_collapse": pre.anti_collapse,
               "jepa_target": pre.jepa_target, "num_actions": num_actions,
               "versions": runtime_versions()}
    write_json(out_dir / "pretrain.json", summary)
    print(f"-> {out_dir} ({counters['wall_time_s'] / 60:.1f} min)")
    return summary


def main(argv: list[str] | None = None) -> None:
    from .config import load_config

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="directory written by atari_jepa.collect")
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=None, help="config for the network/env shape (default: the dataset's)")
    p.add_argument("--updates", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--latent-loss", choices=["mse", "cosine"], default=None)
    p.add_argument("--anti-collapse", choices=["sigreg", "vicreg", "variance", "none"], default=None)
    p.add_argument("--jepa-target", choices=["absolute", "batch_centered", "delta"], default=None)
    p.add_argument("--lambda-anti", type=float, default=None)
    p.add_argument("--motion-channels", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args(argv)

    configure_threads(args.threads)
    device = select_device(args.device)
    _, _, dataset_cfg = load_dataset(args.dataset)
    cfg = load_config(args.config) if args.config else dataset_cfg
    cfg.seed = args.seed
    over = {k: v for k, v in (("updates", args.updates), ("batch_size", args.batch_size),
                              ("latent_loss", args.latent_loss), ("anti_collapse", args.anti_collapse),
                              ("jepa_target", args.jepa_target), ("lambda_anti", args.lambda_anti))
            if v is not None}
    cfg.pretrain = replace(cfg.pretrain, dataset=args.dataset, **over)
    if args.motion_channels:
        cfg.network.motion_channels = True
    run_pretrain(cfg, args.dataset, Path(args.out), device)


if __name__ == "__main__":
    main()
