"""Save frames from a trained (or random) policy as an offline dataset.

    python -m atari_jepa.collect --checkpoint RUN/checkpoint.pt --decisions 200000 --out data/breakout

Phase 2 of the "train RL first, then learn the representation by observing" pipeline: the frames an
already-trained agent generates become a fixed dataset, so representation learning can be debugged
without a moving data distribution.

The policy is a mixture over exploration levels (``--epsilons`` with ``--weights``), because a dataset
from a single greedy policy covers a narrow slice of the game. Everything is stored in the same
``SequenceReplay`` format the trainer uses, so masks, episode boundaries and K-step sampling behave
identically online and offline.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from .config import Config, config_from_dict
from .envs import FrameStacker, make_env
from .replay import SequenceReplay
from .utils import configure_threads, select_device, write_json


@torch.no_grad()
def collect_dataset(cfg: Config, model, env_meta: dict[str, Any], device, decisions: int, seed: int,
                    epsilons: list[float], weights: list[float], max_episode_decisions: int,
                    log_every: int = 25_000) -> tuple[SequenceReplay, dict[str, Any]]:
    env = make_env(cfg.env)
    check_compatible(env_meta, env.metadata())
    size = cfg.env.screen_size
    replay = SequenceReplay(decisions + decisions // 100 + 16, (size, size), cfg.env.history)
    stacker = FrameStacker(cfg.env.history)
    rng = np.random.default_rng(seed)
    probs = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    episodes: list[dict[str, Any]] = []
    start = time.perf_counter()
    taken = 0
    try:
        while taken < decisions:
            epsilon = float(rng.choice(epsilons, p=probs))  # one exploration level per episode
            frame, _ = env.reset(seed=seed + len(episodes))
            history = stacker.reset(frame)
            replay.start_episode(frame)
            ret, steps = 0.0, 0
            while taken < decisions:
                if rng.random() < epsilon:
                    action = int(rng.integers(env.num_actions))
                else:
                    obs = torch.from_numpy(history).to(device).unsqueeze(0)
                    action = int(model.q_values(obs).argmax(dim=-1))
                frame, reward, terminated, truncated, _ = env.step(action)
                replay.add(action, reward, terminated, truncated, frame)
                history = stacker.push(frame)
                ret += reward
                steps += 1
                taken += 1
                if steps >= max_episode_decisions and not (terminated or truncated):
                    break  # collection boundary: the episode stays open, no sequence crosses it
                if terminated or truncated:
                    break
            episodes.append({"epsilon": epsilon, "raw_return": ret, "decisions": steps,
                             "terminated": terminated, "truncated": truncated})
            if taken % log_every < steps:
                print(f"  {taken}/{decisions} decisions, {len(episodes)} episodes, "
                      f"last return {ret:+.0f} (eps {epsilon})", flush=True)
    finally:
        env.close()
    returns = np.array([e["raw_return"] for e in episodes])
    meta = {
        "decisions": taken,
        "episodes": len(episodes),
        "episode_records": episodes,  # per-episode epsilon/return/length, for dataset-composition checks
        "frames_stored": replay.n_written,
        "transitions": replay.num_transitions(),
        "emulator_frames": env.total_frames,
        "policy": {"epsilons": epsilons, "weights": list(probs), "source": "epsilon-greedy Q"},
        "return_mean": float(returns.mean()), "return_min": float(returns.min()),
        "return_max": float(returns.max()),
        "reward_events": int((np.abs(replay.raw_rewards[: replay.n_written]) > 0).sum()),
        "seed": seed,
        "env": env_meta,
        "collect_wall_time_s": time.perf_counter() - start,
    }
    return replay, meta


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="policy used to generate the frames")
    p.add_argument("--out", required=True, help="output directory (replay.npz + dataset.json)")
    p.add_argument("--decisions", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=30_000)
    p.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 0.1, 0.01])
    p.add_argument("--weights", type=float, nargs="+", default=[0.2, 0.6, 0.2])
    p.add_argument("--max-episode-decisions", type=int, default=5_000)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args(argv)
    if len(args.epsilons) != len(args.weights):
        p.error("--epsilons and --weights must have the same length")

    configure_threads(args.threads)
    device = select_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    cfg, model = model_from_checkpoint(ckpt, device)
    print(f"collecting {args.decisions} decisions on {cfg.env.id} with a mixture policy "
          f"(epsilons {args.epsilons}, weights {args.weights}) from {args.checkpoint}")
    replay, meta = collect_dataset(cfg, model, ckpt["env"], device, args.decisions, args.seed,
                                   args.epsilons, args.weights, args.max_episode_decisions)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    replay.save(out / "replay.npz")
    meta["source_checkpoint"] = str(args.checkpoint)
    meta["config"] = cfg.to_dict()
    write_json(out / "dataset.json", meta)
    print(f"-> {out}: {meta['frames_stored']} frames, {meta['episodes']} episodes, "
          f"{meta['reward_events']} reward events, return {meta['return_mean']:+.1f} "
          f"[{meta['return_min']:+.0f}, {meta['return_max']:+.0f}], {meta['collect_wall_time_s']:.0f}s")


def load_dataset(path: str | Path, cfg: Config | None = None) -> tuple[SequenceReplay, dict[str, Any], Config]:
    import json

    path = Path(path)
    meta = json.loads((path / "dataset.json").read_text())
    dataset_cfg = config_from_dict(meta["config"])
    size = dataset_cfg.env.screen_size
    replay = SequenceReplay.load(path / "replay.npz", (size, size))
    return replay, meta, dataset_cfg


if __name__ == "__main__":
    main()
