"""Aggregate evaluation and diagnostics results across runs into a markdown report.

    python -m atari_jepa.report runs            # writes runs/report.md and prints it

Only runs with an ``eval_*.json`` are included. Per-training-seed results are listed individually;
the across-seed mean/std is over per-seed means (no pooling of episodes across seeds).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _fmt(x, digits=3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def build_report(root: Path) -> str:
    runs = sorted(p for p in root.glob("*/seed*") if p.is_dir())
    evals = defaultdict(dict)  # (experiment name, variant, controller) -> seed -> result
    rows = []
    for run in runs:
        meta = _load(run / "metadata.json") or {}
        ckpt_counters = None
        for ev in sorted(run.glob("eval_*.json")):
            r = _load(ev)
            if r is None:
                continue
            label = r["controller"]
            if r.get("planning_horizon", 1) not in (None, 1):
                label += f" (H={r['planning_horizon']})"
            if r.get("epsilon"):  # evaluation epsilon, when not the default 0
                label += f" (eps={r['epsilon']:g})"
            evals[(run.parent.name, r["variant"], label)][r["train_seed"]] = r
            ckpt_counters = r["train_counters"]
        if ckpt_counters:
            rows.append((run, meta, ckpt_counters, _load(run / "diagnostics.json")))

    out = ["# Results", ""]
    out.append("## Evaluation returns (raw game score; episodes per training seed in parentheses)")
    out.append("")
    out.append("| Experiment | Variant | Controller | Per-seed mean ± std (episodes) | Across-seed mean ± std "
               "| Disagreement w/ Q | Latency ms |")
    out.append("|---|---|---|---|---|---|---|")
    for (experiment, variant, ctrl), by_seed in sorted(evals.items()):
        per_seed = [f"s{s}: {r['return_mean']:+.1f} ± {r['return_std']:.1f} ({r['episodes']})" for s, r in sorted(by_seed.items())]
        means = np.array([r["return_mean"] for r in by_seed.values()])
        dis = np.mean([r["controller_stats"]["disagreement_with_q"] for r in by_seed.values()])
        lat = np.mean([r["controller_stats"]["latency_ms_mean"] for r in by_seed.values()])
        out.append(f"| {experiment} | {variant} | {ctrl} | {'; '.join(per_seed)} | {means.mean():+.2f} ± {means.std():.2f} (n={len(means)}) "
                   f"| {dis:.3f} | {lat:.2f} |")
    out.append("")
    out.append("## Training budget and compute")
    out.append("")
    out.append("| Run | Decisions | Updates | Train frames | Eval decisions (during training) | Trained params | Wall time h |")
    out.append("|---|---|---|---|---|---|---|")
    for run, meta, c, _ in rows:
        out.append(f"| {run.parent.name}/{run.name} | {c['decisions']} | {c['updates']} | {c['train_emulator_frames']} "
                   f"| {c['eval_decisions']} | {meta.get('trained_parameters', 'n/a')} | {c['wall_time_s'] / 3600:.2f} |")
    out.append("")
    out.append("## Held-out diagnostics (frozen checkpoints)")
    out.append("")
    out.append("Centered = cosine distance after subtracting the mean held-out target latent. "
               "Depth 10 is extrapolative (training K=5).")
    out.append("")
    out.append("| Run | Roots | Latent d1 pred / persist (centered) | d5 pred / persist (centered) | d5 shuffled (centered) "
               "| d10 pred / persist (centered) | Root std median | Frac < floor | Pairwise cos "
               "| Reward CE d0 (prior) | +1 recall d0 | -1 recall d0 | Q huber d0 / d4 |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run, _, _, d in rows:
        if d is None:
            continue
        lat = d["latent_prediction_cosine_distance"]
        g = lambda k, f: lat.get(k, {}).get(f)  # noqa: E731
        s = d["root_latents"]["online"]
        r0 = d["reward_prediction"]["depth_0"]
        q = d["q_td_error"]
        q4 = q.get("depth_4", {}).get("huber")
        out.append(
            f"| {run.parent.name}/{run.name} | {d['num_roots']} "
            f"| {_fmt(g('depth_1', 'centered_predicted'))} / {_fmt(g('depth_1', 'centered_persistence'))} "
            f"| {_fmt(g('depth_5', 'centered_predicted'))} / {_fmt(g('depth_5', 'centered_persistence'))} "
            f"| {_fmt(g('depth_5', 'centered_shuffled_actions'))} "
            f"| {_fmt(g('depth_10', 'centered_predicted'))} / {_fmt(g('depth_10', 'centered_persistence'))} "
            f"| {s['std_quantiles_5_25_50_75_95'][2]:.4f} | {s['frac_below_variance_floor']:.3f} "
            f"| {s['mean_pairwise_cosine_similarity']:.3f} "
            f"| {_fmt(r0['cross_entropy'])} ({_fmt(r0['prior_cross_entropy'])}) "
            f"| {_fmt(r0['event_+1']['recall'], 2)} (n={r0['event_+1']['events']}) "
            f"| {_fmt(r0['event_-1']['recall'], 2)} (n={r0['event_-1']['events']}) "
            f"| {_fmt(q['depth_0']['huber'], 4)} / {_fmt(q4, 4)} |"
        )
    out.append("")
    out.append("Reward/continuation columns are meaningful only for variants that train those heads (C).")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", nargs="?", default="runs")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    root = Path(args.root)
    text = build_report(root)
    out = Path(args.out) if args.out else root / "report.md"
    out.write_text(text)
    print(text)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
