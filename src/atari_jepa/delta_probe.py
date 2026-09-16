"""Can the latent space support long-range prediction that iterated one-step rollout cannot reach?

The K sweep left an asymmetric question. Our dynamics are supervised on K-step unrolls (K <= 30) and
reach depth d by applying the one-step map d times, so querying them at d = 100 is extrapolation on a
model never trained there. If that fails it does not tell us whether the *latent space* could support
a 100-step prediction -- only that this training strategy did not produce one.

This probe supplies the missing control. With the encoder frozen, it fits a head directly on pairs
(x_t, x_{t+delta}) for each delta, and compares it at matched delta against:

* ``iterated``    -- the checkpoint's own one-step dynamics applied delta times with the real actions
* ``persistence`` -- predict no change
* ``mean``        -- predict the training-split mean latent

The mean baseline is the one that matters for reading the result. Breakout with sticky actions is
stochastic, so at long delta the best a deterministic predictor can do is the conditional mean, which
under a cosine metric looks like collapse. Both learned methods regress toward it, so comparing them to
each other *and* to ``mean`` separates "we never trained for this range" from "this range is
irreducible under a deterministic loss".

    python -m atari_jepa.delta_probe --checkpoint RUN/checkpoint.pt --deltas 1,5,10,30,100,300

Pairs are drawn without materializing the span between them, so memory is O(1) in delta -- the
property that makes this the cheap prototype of a delta-conditioned dynamics model.

Reading the output (``score`` = 1 - distance/mean_distance, so 0 is the mean predictor and 1 exact):

* direct >> iterated  -- the space supports the range; our rollout training never covered it
* direct ~= iterated  -- iterated rollout already generalizes across the time domain
* both ~= 0           -- irreducible at this range for a deterministic predictor
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .checkpoint import load_checkpoint, model_from_checkpoint
from .losses import cosine_distance
from .networks import JumpHead
from .replay import SequenceReplay
from .utils import configure_threads, select_device, write_json


def episode_split(episode_ids: np.ndarray, holdout: int = 4) -> np.ndarray:
    """True where the episode is held out. Splitting by episode (not by time) keeps a whole episode on
    one side -- no pair can straddle the split -- without putting the later, better-played episodes
    entirely in one bucket."""
    return (episode_ids % holdout) == 0


def pair_indices(replay: SequenceReplay, delta: int, rng: np.random.Generator, count: int,
                 holdout: bool, tries: int = 200) -> np.ndarray:
    """``count`` absolute indices r whose r+delta is live and in the same episode.

    Slots inside one episode are consecutive, so "same episode and both slots live" already implies
    every step between them is a real transition: an episode's final slot carries no transition, and a
    termination or truncation ends the episode. The span itself is never touched, which is what keeps
    this O(1) in delta.
    """
    cap, out = replay.capacity, []
    lo, hi = replay.oldest, replay.n_written - delta
    if hi <= lo:
        raise ValueError(f"buffer holds no pair at delta={delta}")
    for _ in range(tries):
        r = rng.integers(lo, hi, size=4 * count)
        slot, tgt = r % cap, (r + delta) % cap
        ok = (
            (replay.abs_index[slot] == r)
            & (replay.abs_index[tgt] == r + delta)
            & (replay.episode_id[slot] == replay.episode_id[tgt])
            & replay.has_transition[slot]
        )
        ok &= episode_split(replay.episode_id[slot]) == holdout
        out.append(r[ok])
        if sum(len(a) for a in out) >= count:
            break
    idx = np.concatenate(out) if out else np.empty(0, dtype=np.int64)
    if len(idx) < count:
        raise ValueError(f"only {len(idx)} of {count} pairs available at delta={delta} "
                         f"(holdout={holdout}); the buffer may hold too few long episodes")
    return idx[:count]


@torch.no_grad()
def encode(model, replay: SequenceReplay, idx: np.ndarray, device: torch.device,
           chunk: int = 512) -> torch.Tensor:
    out = []
    for i in range(0, len(idx), chunk):
        stacks = replay.stacks_at(idx[i : i + chunk])
        out.append(model.encoder(torch.from_numpy(stacks).to(device)))
    return torch.cat(out)


def action_summary(replay: SequenceReplay, roots: np.ndarray, delta: int, num_actions: int,
                   device: torch.device) -> torch.Tensor:
    """Normalized counts of each action over the span; ``[B, A]``. A summary, not the sequence: the
    H-JEPA result says conditioning a jump on the full action sequence is what fails."""
    idx = (roots[:, None] + np.arange(delta)[None, :]) % replay.capacity
    counts = np.zeros((len(roots), num_actions), dtype=np.float32)
    acts = replay.actions[idx]
    for a in range(num_actions):
        counts[:, a] = (acts == a).sum(axis=1)
    return torch.from_numpy(counts / max(delta, 1)).to(device)


@torch.no_grad()
def iterated_prediction(model, z: torch.Tensor, replay: SequenceReplay, roots: np.ndarray,
                        delta: int, chunk: int = 256) -> torch.Tensor:
    """The checkpoint's own one-step dynamics applied ``delta`` times along the real action sequence."""
    preds = []
    for i in range(0, len(roots), chunk):
        zi = z[i : i + chunk]
        idx = (roots[i : i + chunk, None] + np.arange(delta)[None, :]) % replay.capacity
        acts = torch.from_numpy(replay.actions[idx]).to(z.device)
        for k in range(delta):
            zi = model.dynamics(zi, acts[:, k])
        preds.append(zi)
    return torch.cat(preds)


def train_head(head: nn.Module, z: torch.Tensor, target: torch.Tensor, summary: torch.Tensor | None,
               updates: int, batch_size: int, lr: float, seed: int) -> nn.Module:
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = z.shape[0]
    head.train()
    for _ in range(updates):
        sel = torch.randint(0, n, (min(batch_size, n),), generator=g).to(z.device)
        pred = head(z[sel].float(), None if summary is None else summary[sel])
        loss = cosine_distance(pred, target[sel].float()).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    head.eval()
    return head


@torch.no_grad()
def evaluate_head(head: nn.Module, z: torch.Tensor, target: torch.Tensor,
                  summary: torch.Tensor | None, chunk: int = 512) -> float:
    dists = []
    for i in range(0, z.shape[0], chunk):
        pred = head(z[i : i + chunk].float(), None if summary is None else summary[i : i + chunk])
        dists.append(cosine_distance(pred, target[i : i + chunk].float()))
    return float(torch.cat(dists).mean())


def run_delta(model, cfg, replay: SequenceReplay, delta: int, device: torch.device, args,
              rng: np.random.Generator) -> dict[str, Any]:
    num_actions = model.num_actions
    train_roots = pair_indices(replay, delta, rng, args.train_pairs, holdout=False)
    test_roots = pair_indices(replay, delta, rng, args.test_pairs, holdout=True)

    z_tr = encode(model, replay, train_roots, device)
    y_tr = encode(model, replay, train_roots + delta, device)
    z_te = encode(model, replay, test_roots, device)
    y_te = encode(model, replay, test_roots + delta, device)

    # Baselines. "mean" is the deterministic predictor's floor: the conditional mean of the target.
    mean_latent = y_tr.mean(dim=0, keepdim=True).expand_as(y_te)
    d_mean = float(cosine_distance(mean_latent, y_te).mean())
    d_persist = float(cosine_distance(z_te, y_te).mean())
    t0 = time.perf_counter()
    d_iter = float(cosine_distance(iterated_prediction(model, z_te, replay, test_roots, delta), y_te).mean())
    iter_s = time.perf_counter() - t0

    results = {"delta": delta, "train_pairs": len(train_roots), "test_pairs": len(test_roots),
               "mean": d_mean, "persistence": d_persist, "iterated": d_iter,
               "iterated_eval_s": iter_s}
    for variant in ("action_summary", "none"):
        cond = variant == "action_summary"
        s_tr = action_summary(replay, train_roots, delta, num_actions, device) if cond else None
        s_te = action_summary(replay, test_roots, delta, num_actions, device) if cond else None
        torch.manual_seed(args.seed)
        head = JumpHead(model.latent_shape, num_actions if cond else 0, cfg.network).to(device)
        train_head(head, z_tr, y_tr, s_tr, args.updates, args.batch_size, args.lr, args.seed)
        results[f"direct_{variant}"] = evaluate_head(head, z_te, y_te, s_te)
    # score: 1 at an exact prediction, 0 at the mean predictor, negative for worse than the mean
    for k in ("persistence", "iterated", "direct_action_summary", "direct_none"):
        results[f"score_{k}"] = (d_mean - results[k]) / d_mean if d_mean > 0 else float("nan")
    return results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--replay", default=None, help="default: replay.npz beside the checkpoint")
    p.add_argument("--deltas", default="1,5,10,30,100,300")
    p.add_argument("--train-pairs", type=int, default=20000)
    p.add_argument("--test-pairs", type=int, default=4000)
    p.add_argument("--updates", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    configure_threads(args.threads)
    device = select_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    cfg, model = model_from_checkpoint(ckpt, device)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)  # the encoder is frozen: this probes the space, not the encoder

    path = Path(args.replay) if args.replay else Path(args.checkpoint).parent / "replay.npz"
    if not path.exists():
        raise SystemExit(f"no replay at {path}; the run needs replay.save_with_checkpoint")
    replay = SequenceReplay.load(path, (cfg.env.screen_size, cfg.env.screen_size))
    rng = np.random.default_rng(args.seed)
    print(f"{args.checkpoint}: variant {cfg.loss.variant_name()}, K={cfg.replay.rollout_steps}, "
          f"replay {len(replay)} slots; device {device}")

    rows = []
    for delta in [int(d) for d in args.deltas.split(",")]:
        try:
            row = run_delta(model, cfg, replay, delta, device, args, rng)
        except ValueError as exc:
            print(f"delta {delta}: skipped ({exc})")
            continue
        rows.append(row)
        print(f"delta {row['delta']:>4}  mean {row['mean']:.4f}  persist {row['persistence']:.4f}  "
              f"iterated {row['iterated']:.4f} (score {row['score_iterated']:+.3f})  "
              f"direct/actions {row['direct_action_summary']:.4f} (score {row['score_direct_action_summary']:+.3f})  "
              f"direct/none {row['direct_none']:.4f} (score {row['score_direct_none']:+.3f})")

    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "delta_probe.json"
    write_json(out, {"checkpoint": str(args.checkpoint), "variant": cfg.loss.variant_name(),
                     "rollout_steps": cfg.replay.rollout_steps, "train_counters": ckpt["counters"],
                     "updates": args.updates, "rows": rows})
    print(f"-> {out}")


if __name__ == "__main__":
    main()
