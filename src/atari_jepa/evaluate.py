"""Evaluate a checkpoint with the Q-policy or one-step (or H-step) lookahead.

    python -m atari_jepa.evaluate --checkpoint RUN/checkpoint.pt --controller q
    python -m atari_jepa.evaluate --checkpoint RUN/checkpoint.pt --controller lookahead

Both controllers load the same parameters (the checkpoint is never modified) and use the same list of
reset seeds ``seed_base + i``. Evaluation runs in its own environment instance; nothing is written to
training replay. Returns are raw game reward.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from .config import Config
from .envs import FrameStacker, clip_reward, make_env
from .planning import make_controller
from .utils import configure_threads, select_device, write_json


def run_evaluation(
    model,
    cfg: Config,
    env_meta: dict[str, Any],
    device: torch.device,
    controller_name: str,
    episodes: int,
    seed_base: int,
    max_decisions: int,
    epsilon: float = 0.0,
    horizon: int | None = None,
    bootstrap: str | None = None,
    log=None,
) -> dict[str, Any]:
    env = make_env(cfg.env)
    check_compatible(env_meta, env.metadata())
    controller = make_controller(controller_name, model, cfg, device, epsilon, seed=seed_base,
                                 horizon=horizon, bootstrap=bootstrap)
    stacker = FrameStacker(cfg.env.history)
    was_training = model.training
    model.eval()
    records = []
    start = time.perf_counter()
    try:
        for i in range(episodes):
            seed = seed_base + i
            frames_before = env.total_frames
            frame, info = env.reset(seed=seed)
            history = stacker.reset(frame)
            raw_return = clipped_return = 0.0
            decisions = 0
            terminated = truncated = capped = False
            while True:
                action, _ = controller.act(history)
                frame, reward, terminated, truncated, _ = env.step(action)
                history = stacker.push(frame)
                raw_return += reward
                clipped_return += float(clip_reward(reward))
                decisions += 1
                if terminated or truncated:
                    break
                if decisions >= max_decisions:
                    capped = True
                    break
            rec = {
                "episode": i,
                "reset_seed": seed,
                "raw_return": raw_return,
                "clipped_return": clipped_return,
                "decisions": decisions,
                "emulator_frames": env.total_frames - frames_before,
                "reset_noops": info["reset_noops"],
                "reset_fire": info["reset_fire"],
                "terminated": terminated,
                "truncated": truncated,
                "capped_by_eval_limit": capped,
            }
            records.append(rec)
            if log is not None:
                log(f"  [{controller_name}] episode {i} seed {seed}: return {raw_return:+.0f} in {decisions} decisions")
    finally:
        env.close()
        model.train(was_training)
    returns = np.array([r["raw_return"] for r in records], dtype=np.float64)
    return {
        "controller": controller_name,
        "bootstrap": bootstrap or cfg.planning.bootstrap,
        "planning_horizon": (horizon or cfg.planning.horizon) if controller_name == "lookahead" else None,
        "training_rollout_steps": cfg.replay.rollout_steps,
        "epsilon": epsilon,
        "episodes": episodes,
        "seed_base": seed_base,
        "max_episode_decisions": max_decisions,
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_min": float(returns.min()),
        "return_max": float(returns.max()),
        "total_decisions": int(sum(r["decisions"] for r in records)),
        "total_emulator_frames": int(sum(r["emulator_frames"] for r in records)),
        "wall_time_s": time.perf_counter() - start,
        "controller_stats": controller.stats(),
        "records": records,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--controller", choices=["q", "sf", "lookahead", "hierarchical", "both"], default="q")
    p.add_argument("--bootstrap", choices=["q", "sf"], default=None,
                   help="value at the search leaf (default: planning.bootstrap from the checkpoint)")
    p.add_argument("--episodes", type=int, default=None, help="default: eval.episodes from the checkpoint config")
    p.add_argument("--seed-base", type=int, default=None, help="reset seed of episode i is seed_base + i")
    p.add_argument("--epsilon", type=float, default=None)
    p.add_argument("--horizon", type=int, default=None, help="lookahead horizon (training uses K steps)")
    p.add_argument("--max-episode-decisions", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", default=None, help="output JSON (default: next to the checkpoint)")
    p.add_argument(
        "--allow-untrained-model",
        action="store_true",
        help="allow lookahead on a checkpoint whose dynamics/reward/continuation were not all trained",
    )
    args = p.parse_args(argv)

    configure_threads(args.threads)
    device = select_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    cfg, model = model_from_checkpoint(ckpt, device)
    controllers = ["q", "lookahead"] if args.controller == "both" else [args.controller]
    if "lookahead" in controllers and not cfg.loss.trains_model and not args.allow_untrained_model:
        raise SystemExit(
            f"checkpoint variant {cfg.loss.variant_name()} did not train dynamics, reward and continuation; "
            "lookahead would use untrained components. Pass --allow-untrained-model to run it anyway."
        )
    episodes = args.episodes if args.episodes is not None else cfg.eval.episodes
    seed_base = args.seed_base if args.seed_base is not None else cfg.eval.seed_base
    epsilon = args.epsilon if args.epsilon is not None else cfg.eval.epsilon
    max_dec = args.max_episode_decisions or cfg.eval.max_episode_decisions
    horizon = args.horizon or cfg.planning.horizon

    print(f"checkpoint {args.checkpoint}: variant {cfg.loss.variant_name()}, "
          f"{ckpt['counters']['decisions']} training decisions, {ckpt['counters']['updates']} updates; device {device}")
    if "lookahead" in controllers and horizon > cfg.replay.rollout_steps:
        print(f"note: planning horizon {horizon} exceeds the training rollout horizon K={cfg.replay.rollout_steps} (extrapolative)")
    for name in controllers:
        result = run_evaluation(
            model, cfg, ckpt["env"], device, name, episodes, seed_base, max_dec, epsilon,
            horizon=horizon, bootstrap=args.bootstrap, log=print,
        )
        result["checkpoint"] = str(args.checkpoint)
        result["variant"] = cfg.loss.variant_name()
        result["train_seed"] = cfg.seed
        result["train_counters"] = ckpt["counters"]
        if args.out and len(controllers) == 1:
            out = Path(args.out)
        else:
            suffix = f"_h{horizon}" if name == "lookahead" and horizon != 1 else ""
            if (args.bootstrap or cfg.planning.bootstrap) == "sf" and name != "sf":
                suffix += "_sfboot"
            out = Path(args.checkpoint).parent / f"eval_{name}{suffix}.json"
        write_json(out, result)
        s = result["controller_stats"]
        print(
            f"[{name}] return {result['return_mean']:+.2f} ± {result['return_std']:.2f} "
            f"(min {result['return_min']:+.0f}, max {result['return_max']:+.0f}) over {episodes} episodes; "
            f"{result['total_decisions']} decisions / {result['total_emulator_frames']} frames; "
            f"latency {s['latency_ms_mean']:.2f} ms; disagreement with Q {s['disagreement_with_q']:.3f} -> {out}"
        )


if __name__ == "__main__":
    main()
