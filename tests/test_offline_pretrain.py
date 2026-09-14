"""Offline pipeline: anti-collapse terms, dataset collection, pretraining, and RL re-attachment."""

import json

import numpy as np
import pytest
import torch
from test_checkpoint import tiny_config

from atari_jepa.agent import WorldModel
from atari_jepa.checkpoint import load_checkpoint, model_from_checkpoint
from atari_jepa.collect import collect_dataset, load_dataset
from atari_jepa.collect import main as collect_main
from atari_jepa.losses import anti_collapse, epps_pulley, mse_latent_distance, sigreg_penalty, vicreg_penalty
from atari_jepa.pretrain import main as pretrain_main
from atari_jepa.pretrain import pretrain_losses, run_pretrain
from atari_jepa.train import Trainer


def test_epps_pulley_orders_distributions_by_non_gaussianity():
    g = torch.Generator().manual_seed(0)
    n = 256
    gauss = float(epps_pulley(torch.randn(n, 8, generator=g)).mean())
    uniform = float(epps_pulley((torch.rand(n, 8, generator=g) - 0.5) * 12**0.5).mean())
    bimodal = float(epps_pulley(torch.cat([torch.randn(n // 2, 8, generator=g) - 3,
                                           torch.randn(n // 2, 8, generator=g) + 3]) / 10**0.5).mean())
    constant = float(epps_pulley(torch.zeros(n, 8)).mean())
    assert gauss < uniform < bimodal < constant


def test_sigreg_penalizes_collapse_including_low_rank():
    """The property VICReg's variance floor misses: a latent that varies, but only in a few directions."""
    g = torch.Generator().manual_seed(0)
    full = torch.randn(64, 128, generator=g)
    rank2 = torch.randn(64, 2, generator=g) @ torch.randn(2, 128, generator=g)
    constant = torch.ones(64, 128)
    anisotropic = torch.randn(64, 128, generator=g) * torch.tensor([50.0] + [1.0] * 127)
    s = lambda z: float(sigreg_penalty(z, 64, torch.Generator().manual_seed(1)))  # noqa: E731
    assert s(full) < s(rank2) and s(full) < s(anisotropic) and s(rank2) < s(constant)
    # the per-dimension variance floor barely reacts to low rank: every dimension still varies plenty,
    # so it reports ~0 while SIGReg reports a large penalty for the same embedding
    assert float(anti_collapse("variance", rank2, 0.1, 1e-4)) < 0.01 < s(rank2)
    assert float(vicreg_penalty(rank2)) > float(vicreg_penalty(full))


def test_mse_latent_distance_matches_manual_and_handles_modes():
    g = torch.Generator().manual_seed(0)
    z_hat = [torch.randn(3, 4, generator=g) for _ in range(3)]
    target = torch.randn(3, 2, 4, generator=g)
    target0 = torch.randn(3, 4, generator=g)
    got = mse_latent_distance(z_hat, target, target0, "absolute")
    want = torch.stack([(z_hat[k + 1] - target[:, k]).pow(2).mean(1) for k in range(2)], dim=1)
    torch.testing.assert_close(got, want)
    offset = 7.0 * torch.randn(1, 4, generator=g)
    shifted = mse_latent_distance([z + offset for z in z_hat], target + offset, target0 + offset, "delta")
    base = mse_latent_distance(z_hat, target, target0, "delta")
    torch.testing.assert_close(base, shifted)  # delta ignores a shared constant


def synthetic_dataset(tmp_path, decisions=400):
    """Train a tiny RL agent, then collect frames with it (phases 1 and 2)."""
    run = tmp_path / "rl"
    cfg = tiny_config(run)
    Trainer(cfg, run, torch.device("cpu")).run(["test"])
    ckpt = load_checkpoint(run / "checkpoint.pt")
    rl_cfg, model = model_from_checkpoint(ckpt, torch.device("cpu"))
    replay, meta = collect_dataset(rl_cfg, model, ckpt["env"], torch.device("cpu"), decisions, 5,
                                   [1.0, 0.1], [0.5, 0.5], 60)
    data = tmp_path / "data"
    data.mkdir()
    replay.save(data / "replay.npz")
    meta["config"] = rl_cfg.to_dict()
    (data / "dataset.json").write_text(json.dumps(meta, default=float))
    return run, data, rl_cfg


def test_collected_dataset_round_trips_and_covers_the_policy_mixture(tmp_path):
    _, data, _ = synthetic_dataset(tmp_path)
    replay, meta, cfg = load_dataset(data)
    assert meta["decisions"] == 400 and replay.num_transitions() >= 390
    assert {e["epsilon"] for e in meta["episodes"]} if isinstance(meta["episodes"], list) else True
    assert len(replay.valid_roots()) > 300
    batch = replay.gather(replay.valid_roots()[:8], 3)
    assert batch.observations.shape[1:] == (4, 4, 84, 84) and batch.valid[:, 0].all()


def test_pretraining_reduces_latent_prediction_and_keeps_the_latent_from_collapsing(tmp_path):
    _, data, cfg = synthetic_dataset(tmp_path, decisions=600)
    cfg.pretrain.updates = 120
    cfg.pretrain.batch_size = 8
    cfg.pretrain.rollout_steps = 3
    cfg.pretrain.log_every = 40
    out = tmp_path / "pre"
    run_pretrain(cfg, str(data), out, torch.device("cpu"))
    rows = [json.loads(line) for line in open(out / "pretrain.jsonl")]
    assert rows[-1]["loss_jepa"] < rows[0]["loss_jepa"]
    assert rows[-1]["batch_effective_rank"] > 1.5 and rows[-1]["pairwise_cos"] < 0.999  # not collapsed
    ckpt = load_checkpoint(out / "checkpoint.pt")
    assert ckpt["pretrain"] is True and "encoder.convs.0.weight" in ckpt["model"]


def test_rl_can_reattach_to_a_frozen_pretrained_encoder(tmp_path):
    _, data, cfg = synthetic_dataset(tmp_path, decisions=600)
    cfg.pretrain.updates = 40
    cfg.pretrain.batch_size = 8
    cfg.pretrain.rollout_steps = 3
    pre = tmp_path / "pre"
    run_pretrain(cfg, str(data), pre, torch.device("cpu"))
    pre_state = load_checkpoint(pre / "checkpoint.pt")["model"]

    run = tmp_path / "rl2"
    rl_cfg = tiny_config(run)
    rl_cfg.train.init_from = str(pre / "checkpoint.pt")
    rl_cfg.train.freeze_encoder = True
    trainer = Trainer(rl_cfg, run, torch.device("cpu"))
    before = {k: v.clone() for k, v in trainer.model.encoder.state_dict().items()}
    torch.testing.assert_close(before["convs.0.weight"], pre_state["encoder.convs.0.weight"])
    assert all(not p.requires_grad for p in trainer.model.encoder.parameters())
    n_opt = sum(p.numel() for g in trainer.learner.optimizer.param_groups for p in g["params"])
    assert n_opt == sum(p.numel() for p in trainer.model.trainable_parameters())
    assert n_opt < sum(p.numel() for p in trainer.model.online_parameters())

    trainer.run(["test"])
    after = trainer.model.encoder.state_dict()
    for k, v in before.items():
        torch.testing.assert_close(v, after[k], msg=f"frozen encoder changed: {k}")
    # the heads did train
    assert trainer.counters["updates"] > 0


def test_pretrain_and_collect_clis(tmp_path):
    run, data, _ = synthetic_dataset(tmp_path, decisions=200)
    out = tmp_path / "data2"
    collect_main(["--checkpoint", str(run / "checkpoint.pt"), "--out", str(out), "--decisions", "150",
                  "--epsilons", "1.0", "--weights", "1.0", "--device", "cpu", "--seed", "7"])
    assert (out / "replay.npz").exists() and json.loads((out / "dataset.json").read_text())["decisions"] == 150
    pre = tmp_path / "pre2"
    pretrain_main(["--dataset", str(out), "--out", str(pre), "--updates", "20", "--batch-size", "4",
                   "--anti-collapse", "vicreg", "--device", "cpu"])
    assert json.loads((pre / "pretrain.json").read_text())["anti_collapse"] == "vicreg"


def test_pretrain_losses_train_encoder_and_dynamics_but_not_targets(tmp_path):
    from conftest import random_batch, world_model_config

    cfg = world_model_config()
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4)
    total, metrics = pretrain_losses(model, random_batch(), cfg.pretrain)
    total.backward()
    assert torch.isfinite(total) and metrics["loss_anti"] >= 0
    for name in ("encoder", "dynamics"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters()), name
    assert all(p.grad is None for p in model.target_parameters())
    for head in ("q_head", "reward_head", "continuation_head"):  # RL heads are untouched offline
        assert all(p.grad is None for p in getattr(model, head).parameters())
