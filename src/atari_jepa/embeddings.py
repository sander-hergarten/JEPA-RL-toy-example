"""Is the learned latent space a representative one?

    python -m atari_jepa.embeddings --checkpoint RUN/checkpoint.pt

Low prediction loss does not mean the representation is good: a nearly constant latent, or one that
keeps only the slow background, can predict itself well. This pass collects fresh held-out episodes and
measures, on the frozen checkpoint's online encoder:

1. **Spectrum** -- PCA of root latents: participation ratio, entropy-based effective rank, variance in
   the leading components, and the number of directions holding 90/99% of the variance. A collapsed or
   low-rank latent shows here even when per-dimension std passes the variance floor.
2. **Temporal structure** -- centered cosine distance between latents `t` and `t + gap` inside an
   episode against pairs from different episodes. A representation that tracks the game state grows
   with the gap and saturates near the across-episode level.
3. **State decodability** (evaluation-only) -- ridge regression from the latent to the emulator state
   (ALE RAM, or the toy game's true state), fit on some held-out episodes and scored by R^2 on the
   others. This says whether the ball/paddle information survives in the latent. The emulator state is
   never an input to the agent or to training.
4. **Task relevance** -- a linear probe for "a reward event happens within the next H decisions",
   scored against its base rate, and one for the movement class of the next action.

Everything here is a read-only probe on a frozen checkpoint: no agent parameter is ever updated.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .agent import trained_modules
from .checkpoint import load_checkpoint, model_from_checkpoint
from .diagnostics import collect_heldout, movement_classes, movement_probe
from .replay import SequenceReplay
from .utils import configure_threads, select_device, write_json

# RAM byte annotations, reported alongside the unlabelled per-byte summary.
# Pong: commonly cited (Anand et al., "Atari Annotated RAM Interface"), not verified here.
PONG_RAM_LABELS = {"player_y": 51, "player_x": 46, "enemy_y": 50, "enemy_x": 45,
                   "ball_x": 49, "ball_y": 54, "enemy_score": 13, "player_score": 14}
# Breakout: verified here by correlating each byte with pixel measurements over 3k random-policy frames
# (paddle centroid |r| = 0.94, ball x 0.97, ball y 0.79); the rest are the cited values.
BREAKOUT_RAM_LABELS = {"player_x": 72, "ball_x": 99, "ball_y": 101, "blocks_hit_count": 77, "score": 84}
RAM_LABELS = {"pong": PONG_RAM_LABELS, "breakout": BREAKOUT_RAM_LABELS}


def ram_labels_for(env_id: str) -> tuple[str, dict[str, int]]:
    """Labelled RAM bytes for a game id, or an empty mapping when the game has no table here."""
    game = env_id.split("/")[-1].split("-")[0].lower()
    return game, RAM_LABELS.get(game, {})


@torch.no_grad()
def encode_roots(model, replay: SequenceReplay, roots: np.ndarray, device, batch_size: int = 64) -> torch.Tensor:
    out = []
    for start in range(0, len(roots), batch_size):
        stacks = replay.stacks_at(roots[start : start + batch_size])
        out.append(model.encoder(torch.from_numpy(stacks).to(device)).flatten(1).cpu())
    return torch.cat(out)


def spectrum(z: torch.Tensor) -> dict[str, Any]:
    """PCA statistics of the latent cloud."""
    zc = (z - z.mean(0)).double()
    n = zc.shape[0]
    # eigenvalues of the covariance via the smaller Gram matrix when samples < dims
    gram = (zc @ zc.T) / max(n - 1, 1) if n <= zc.shape[1] else (zc.T @ zc) / max(n - 1, 1)
    eig = torch.linalg.eigvalsh(gram).clamp(min=0).flip(0)
    total = float(eig.sum())
    if total <= 0:
        return {"degenerate": True, "total_variance": total}
    p = (eig / total).numpy()
    cum = np.cumsum(p)
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return {
        "dims": int(z.shape[1]),
        "samples": int(n),
        "total_variance": total,
        "participation_ratio": float(eig.sum() ** 2 / (eig**2).sum()),
        "effective_rank_entropy": float(np.exp(entropy)),
        "top1_variance_frac": float(p[0]),
        "top10_variance_frac": float(cum[min(9, len(cum) - 1)]),
        "dims_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "dims_for_99pct": int(np.searchsorted(cum, 0.99) + 1),
    }


def temporal_structure(z: torch.Tensor, episodes: np.ndarray, steps: np.ndarray,
                       gaps: tuple[int, ...] = (1, 2, 5, 10, 20, 50), rng_seed: int = 0) -> dict[str, Any]:
    """Centered cosine distance at increasing time gaps, vs pairs from different episodes."""
    zc = F.normalize(z - z.mean(0), dim=1)
    order = np.lexsort((steps, episodes))
    ep, st = episodes[order], steps[order]
    zs = zc[order]
    out: dict[str, Any] = {}
    for gap in gaps:
        same = (ep[gap:] == ep[:-gap]) & (st[gap:] - st[:-gap] == gap)
        if same.sum() < 10:
            continue
        a, b = zs[:-gap][same], zs[gap:][same]
        out[f"gap_{gap}"] = {"pairs": int(same.sum()), "mean_cosine_distance": float((1 - (a * b).sum(1)).mean())}
    rng = np.random.default_rng(rng_seed)
    n = len(zc)
    i, j = rng.integers(0, n, 4096), rng.integers(0, n, 4096)
    cross = episodes[i] != episodes[j]
    if cross.sum() > 10:
        out["different_episode"] = {
            "pairs": int(cross.sum()),
            "mean_cosine_distance": float((1 - (zc[i[cross]] * zc[j[cross]]).sum(1)).mean()),
        }
    return out


def ridge_fit_predict(x_tr: torch.Tensor, y_tr: torch.Tensor, x_te: torch.Tensor, lam: float) -> torch.Tensor:
    d = x_tr.shape[1]
    a = x_tr.T @ x_tr + lam * torch.eye(d, dtype=x_tr.dtype)
    w = torch.linalg.solve(a, x_tr.T @ y_tr)
    return x_te @ w


def probe_features(z: torch.Tensor, train: np.ndarray, max_components: int = 256) -> tuple[torch.Tensor, int]:
    """Standardize and project onto the training half's leading principal components, plus a bias.

    The latent has thousands of dimensions and the held-out set only has thousands of samples, so an
    unprojected ridge would interpolate the training half and score negative R^2 on the test half.
    """
    x = z.double()
    x = (x - x[train].mean(0)) / (x[train].std(0) + 1e-6)
    k = int(min(max_components, max(8, train.sum() // 10), x.shape[1]))
    _, _, v = torch.pca_lowrank(x[train], q=k, niter=4)
    x = x @ v
    return torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], dim=1), k


def ridge_r2(z: torch.Tensor, targets: torch.Tensor, train: np.ndarray,
             lams: tuple[float, ...] = (1e-2, 1e0, 1e1, 1e2, 1e3, 1e4)) -> tuple[np.ndarray, float]:
    """Per-target test R^2 of a ridge probe (lambda picked on a split inside the training half)."""
    x, _ = probe_features(z, train)
    y = targets.double()
    y_mean = y[train].mean(0)
    tr_idx = np.where(train)[0]
    inner = tr_idx[: int(0.8 * len(tr_idx))]
    inner_val = tr_idx[int(0.8 * len(tr_idx)) :]
    best, best_lam = -np.inf, lams[0]
    for lam in lams:
        pred = ridge_fit_predict(x[inner], y[inner], x[inner_val], lam)
        sse = ((y[inner_val] - pred) ** 2).sum()
        score = float(-sse)
        if score > best:
            best, best_lam = score, lam
    test = ~train
    pred = ridge_fit_predict(x[train], y[train], x[test], best_lam)
    sse = ((y[test] - pred) ** 2).sum(0)
    sst = ((y[test] - y_mean) ** 2).sum(0)
    r2 = (1 - sse / sst.clamp(min=1e-9)).numpy()
    r2[sst.numpy() <= 1e-9] = np.nan  # constant targets carry no information
    return r2, best_lam


def state_decodability(z: torch.Tensor, states: np.ndarray, train: np.ndarray, env_kind: str,
                       env_id: str = "") -> dict[str, Any]:
    y = torch.from_numpy(states.astype(np.float64))
    varying = np.asarray(y.std(0) > 1e-6)
    r2, lam = ridge_r2(z, y, train)
    finite = r2[np.isfinite(r2)]
    out: dict[str, Any] = {
        "targets": "ALE RAM bytes" if env_kind == "ale" else "toy game state",
        "ridge_lambda": lam,
        "probe_components": int(probe_features(z, train)[1]),
        "varying_targets": int(varying.sum()),
        "mean_r2_over_varying": float(np.nanmean(r2[varying])) if varying.any() else None,
        "median_r2_over_varying": float(np.nanmedian(r2[varying])) if varying.any() else None,
        "targets_r2_above_0.5": int((finite > 0.5).sum()),
        "targets_r2_above_0.9": int((finite > 0.9).sum()),
        "best_targets": sorted(
            [{"index": int(i), "r2": float(r2[i])} for i in np.where(np.isfinite(r2))[0]],
            key=lambda d: -d["r2"],
        )[:8],
        "note": "evaluation-only probe; the emulator state is never an agent input",
    }
    if env_kind == "ale":
        game, labels = ram_labels_for(env_id)
        out["labelled_ram"] = {
            name: (float(r2[idx]) if idx < len(r2) and np.isfinite(r2[idx]) else None)
            for name, idx in labels.items()
        }
        out["label_caveat"] = (
            "Breakout player_x/ball_x/ball_y verified here by pixel correlation; other indices are "
            "published annotations" if game == "breakout" else
            f"RAM indices from published {game} annotations; not verified here"
        )
    return out


def reward_event_probe(z: torch.Tensor, replay: SequenceReplay, roots: np.ndarray, train: np.ndarray,
                       horizon: int = 8) -> dict[str, Any]:
    """Linear probe for 'a reward event occurs within the next `horizon` decisions'."""
    batch = replay.gather(roots, horizon)
    label = ((np.abs(batch.rewards) > 0) & batch.valid).any(axis=1).astype(np.float64)
    if label.sum() < 10 or label.sum() == len(label):
        return {"skipped": f"only {int(label.sum())} positive windows"}
    y = torch.from_numpy(label)[:, None]
    r2, lam = ridge_r2(z, y, train)
    x, _ = probe_features(z, train)
    pred = ridge_fit_predict(x[train], y[train], x[~train], lam).squeeze(1).numpy()
    yt = label[~train]
    order = np.argsort(pred)
    ranks = np.empty(len(pred));  ranks[order] = np.arange(len(pred))
    pos, neg = yt == 1, yt == 0
    auc = float((ranks[pos].mean() - (pos.sum() - 1) / 2) / max(neg.sum(), 1)) if pos.any() and neg.any() else float("nan")
    return {
        "horizon_decisions": horizon,
        "base_rate_test": float(yt.mean()),
        "test_r2": float(r2[0]),
        "test_auc": auc,
        "note": "AUC 0.5 means the latent carries no linear information about imminent rewards",
    }


def analyse(model, cfg, ckpt, device, episodes: int, max_decisions: int, seed_base: int, epsilon: float,
            horizon: int = 8) -> dict[str, Any]:
    t0 = time.perf_counter()
    replay, ep_stats = collect_heldout(model, cfg, ckpt["env"], device, episodes, max_decisions, seed_base,
                                       epsilon, record_states=True)
    if replay.states is None or len(replay.states) < replay.n_written:
        raise RuntimeError("emulator states were not recorded for every stored frame")
    roots = replay.valid_roots()
    if len(roots) < 50:
        raise ValueError(f"only {len(roots)} held-out roots; collect more decisions")
    slots = roots % replay.capacity
    ep_ids, steps = replay.episode_id[slots], replay.step_in_episode[slots]
    unique = np.unique(ep_ids)
    if len(unique) >= 2:
        train = np.isin(ep_ids, unique[: (len(unique) + 1) // 2])
        split = f"episodes {unique[: (len(unique) + 1) // 2].tolist()} train / rest test"
    else:
        train = np.arange(len(roots)) < int(0.7 * len(roots))
        split = "single episode: first 70% train / last 30% test"

    z = encode_roots(model, replay, roots, device)
    with torch.no_grad():
        z_target = torch.cat([
            model.target_encoder(torch.from_numpy(replay.stacks_at(roots[i : i + 64])).to(device))
            .flatten(1).cpu() for i in range(0, len(roots), 64)])

    actions = replay.actions[slots]
    move = movement_classes(ckpt["env"]["action_meanings"])
    z_next = encode_roots(model, replay, roots + 1, device)  # every root has a real next observation

    result: dict[str, Any] = {
        "checkpoint": None,
        "variant": cfg.loss.variant_name(),
        "trained_modules": trained_modules(cfg),
        "heldout": {"episodes": ep_stats, "roots": int(len(roots)), "split": split,
                    "policy": f"epsilon-greedy Q, epsilon={epsilon}", "seed_base": seed_base},
        "spectrum_online": spectrum(z),
        "spectrum_target": spectrum(z_target),
        "temporal_structure": temporal_structure(z, ep_ids, steps),
        "state_decodability": state_decodability(z, replay.states[roots], train, ckpt["env"].get("kind", "ale"),
                                                 ckpt["env"].get("id", "")),
        "reward_event_probe": reward_event_probe(z, replay, roots, train, horizon),
    }
    result["movement_probe"] = movement_probe(z, z_next, move[actions], ep_ids)
    result["wall_time_s"] = time.perf_counter() - t0
    return result


def print_summary(r: dict[str, Any]) -> None:
    s, st = r["spectrum_online"], r["spectrum_target"]
    print(f"variant {r['variant']}, {r['heldout']['roots']} held-out roots ({r['heldout']['split']})")
    print(f"spectrum (online):  participation ratio {s['participation_ratio']:.1f}, effective rank "
          f"{s['effective_rank_entropy']:.1f} of {s['dims']} dims; top1 {s['top1_variance_frac']:.2f}, "
          f"top10 {s['top10_variance_frac']:.2f}; 90% in {s['dims_for_90pct']} dims, 99% in {s['dims_for_99pct']}")
    print(f"spectrum (target):  participation ratio {st['participation_ratio']:.1f}, effective rank "
          f"{st['effective_rank_entropy']:.1f}")
    ts = " ".join(f"{k.replace('gap_', 'Δ')}={v['mean_cosine_distance']:.3f}" for k, v in r["temporal_structure"].items()
                  if k != "different_episode")
    cross = r["temporal_structure"].get("different_episode", {}).get("mean_cosine_distance")
    print(f"temporal distance:  {ts} | across episodes={cross:.3f}" if cross else f"temporal distance: {ts}")
    d = r["state_decodability"]
    print(f"state decodability: mean R2 {d['mean_r2_over_varying']:.3f} over {d['varying_targets']} varying "
          f"{d['targets']}; {d['targets_r2_above_0.5']} above 0.5, {d['targets_r2_above_0.9']} above 0.9")
    if d.get("labelled_ram"):
        print("  labelled RAM R2: " + ", ".join(f"{k} {v:.2f}" for k, v in d["labelled_ram"].items()
                                                if v is not None))
    rp = r["reward_event_probe"]
    if "skipped" not in rp:
        print(f"reward within {rp['horizon_decisions']}: AUC {rp['test_auc']:.3f} (base rate "
              f"{rp['base_rate_test']:.3f}, R2 {rp['test_r2']:.3f})")
    mp = r.get("movement_probe", {})
    if "test_balanced_acc" in mp:
        print(f"movement probe:     balanced acc {mp['test_balanced_acc']:.3f} (chance "
              f"{mp['chance_balanced_acc']:.3f}, majority acc {mp['test_majority_baseline_acc']:.3f})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, nargs="+", help="one or more checkpoints to compare")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--max-decisions", type=int, default=6000)
    p.add_argument("--seed-base", type=int, default=None)
    p.add_argument("--policy-epsilon", type=float, default=None)
    p.add_argument("--reward-horizon", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", default=None, help="JSON output (default: embeddings.json next to each checkpoint)")
    args = p.parse_args(argv)

    configure_threads(args.threads)
    device = select_device(args.device)
    for path in args.checkpoint:
        ckpt = load_checkpoint(path, device)
        cfg, model = model_from_checkpoint(ckpt, device)
        seed_base = args.seed_base if args.seed_base is not None else cfg.diagnostics.seed_base
        eps = args.policy_epsilon if args.policy_epsilon is not None else cfg.diagnostics.policy_epsilon
        result = analyse(model, cfg, ckpt, device, args.episodes, args.max_decisions, seed_base, eps,
                         args.reward_horizon)
        result["checkpoint"] = str(path)
        result["train_counters"] = ckpt["counters"]
        out = Path(args.out) if args.out and len(args.checkpoint) == 1 else Path(path).parent / "embeddings.json"
        write_json(out, result)
        print(f"\n=== {path}")
        print_summary(result)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
