"""Train a Q baseline, temporal-JEPA representation, or full latent world model.

    python -m atari_jepa.train --config configs/pong_smoke.yaml
    python -m atari_jepa.train --config configs/pong_world_model.yaml --seed 0
    python -m atari_jepa.train --resume runs/pong_world_model/seed0

Data is always collected with an epsilon-greedy Q-policy (random during warmup), for every variant.
Outputs go to the run directory: config.yaml, metadata.json, train_episodes.jsonl, updates.jsonl,
eval.jsonl, checkpoint.pt (and replay.npz if replay.save_with_checkpoint).
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .agent import Learner, WorldModel, check_precision_supported, trained_modules
from .checkpoint import check_compatible, load_checkpoint, restore_rng, rng_state, save_checkpoint
from .config import Config, _set_nested, config_from_dict, load_config, save_config
from .envs import FrameStacker, clip_reward, make_env
from .evaluate import run_evaluation
from .replay import SequenceReplay
from .utils import JsonlWriter, configure_threads, runtime_versions, seed_everything, select_device, write_json


def epsilon_at(decisions: int, cfg) -> float:
    frac = min(1.0, decisions / max(cfg.eps_decay_decisions, 1))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def _collector_seed(cfg: Config, resume_count: int) -> int:
    return int(np.random.SeedSequence([cfg.seed, 7919, resume_count]).generate_state(1)[0])


class Trainer:
    def __init__(self, cfg: Config, run_dir: Path, device: torch.device, resume: dict[str, Any] | None = None):
        self.cfg = cfg
        self.run_dir = run_dir
        self.device = device
        run_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(cfg.seed)
        self.explore_rng = np.random.default_rng([cfg.seed, 1])
        self.replay_rng = np.random.default_rng([cfg.seed, 2])

        check_precision_supported(cfg, device)
        self.env = make_env(cfg.env)
        self.env_meta = self.env.metadata()
        self.model = WorldModel(cfg, self.env.num_actions).to(device)
        self.learner = Learner(self.model, cfg)
        size = cfg.env.screen_size
        self.replay = SequenceReplay(cfg.replay.capacity, (size, size), cfg.env.history)
        self.stacker = FrameStacker(cfg.env.history)

        self.counters: dict[str, Any] = {
            "decisions": 0,  # training decisions (agent steps), including warmup
            "updates": 0,
            "episodes": 0,
            "train_emulator_frames": 0,
            "sampled_sequences": 0,
            "sampled_valid_transitions": 0,
            "sampled_latent_targets": 0,
            "eval_decisions": 0,
            "eval_emulator_frames": 0,
            "warmup_until": cfg.train.warmup_decisions,
            "resume_count": 0,
            "wall_time_s": 0.0,
        }
        self.episodes_log = JsonlWriter(run_dir / "train_episodes.jsonl")
        self.updates_log = JsonlWriter(run_dir / "updates.jsonl")
        self.eval_log = JsonlWriter(run_dir / "eval.jsonl")
        if resume is not None:
            self._restore(resume)

    # --------------------------------------------------------------------------------- persistence
    def _restore(self, ckpt: dict[str, Any]) -> None:
        check_compatible(ckpt["env"], self.env_meta)
        self.model.load_state_dict(ckpt["model"])
        self.learner.optimizer.load_state_dict(ckpt["optimizer"])
        self.counters.update(ckpt["counters"])
        self.learner.updates = self.counters["updates"]
        restore_rng(ckpt["rng"], self.explore_rng, self.replay_rng)
        self.counters["resume_count"] += 1
        self._trim_logs(self.counters["decisions"])
        replay_path = self.run_dir / "replay.npz"
        saved_written = ckpt.get("replay_n_written")
        if saved_written is not None and replay_path.exists():
            replay = SequenceReplay.load(replay_path, self.replay.frame_shape)
            if replay.n_written == saved_written and replay.capacity == self.cfg.replay.capacity:
                self.replay = replay
                self._log(f"resumed with persisted replay ({len(replay)} frames)")
            else:
                self._log("replay.npz does not match the checkpoint; ignoring it")
        if len(self.replay) == 0:
            self.counters["warmup_until"] = self.counters["decisions"] + self.cfg.train.warmup_decisions
            self._log(
                f"resumed WITHOUT replay: re-entering random warmup until decision {self.counters['warmup_until']}"
            )
        self._log(
            f"resumed at decision {self.counters['decisions']}, update {self.counters['updates']}; "
            "collector and emulator reset (valid training resume, not a bit-for-bit continuation)"
        )

    def _trim_logs(self, decisions: int) -> None:
        """Drop log records written after the checkpoint, since those decisions are re-run on resume."""
        for writer in (self.episodes_log, self.updates_log, self.eval_log):
            if not writer.path.exists():
                continue
            lines = writer.path.read_text().splitlines()
            kept = [line for line in lines if line and json.loads(line).get("decisions", 0) <= decisions]
            if len(kept) != len(lines):
                writer.path.write_text("".join(line + "\n" for line in kept))
                self._log(f"trimmed {len(lines) - len(kept)} records after decision {decisions} from {writer.path.name}")

    def save(self) -> None:
        replay_written = None
        if self.cfg.replay.save_with_checkpoint:
            self.replay.save(self.run_dir / "replay.npz")
            replay_written = self.replay.n_written
        save_checkpoint(
            self.run_dir / "checkpoint.pt",
            {
                "config": self.cfg.to_dict(),
                "variant": self.cfg.loss.variant_name(),
                "model": self.model.state_dict(),
                "optimizer": self.learner.optimizer.state_dict(),
                "counters": dict(self.counters),
                "schedule": {
                    "eps_start": self.cfg.train.eps_start,
                    "eps_end": self.cfg.train.eps_end,
                    "eps_decay_decisions": self.cfg.train.eps_decay_decisions,
                    "epsilon_now": epsilon_at(self.counters["decisions"], self.cfg.train),
                    "warmup_until": self.counters["warmup_until"],
                },
                "rng": rng_state(self.explore_rng, self.replay_rng),
                "env": self.env_meta,
                "replay_n_written": replay_written,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    # --------------------------------------------------------------------------------------- helpers
    def _log(self, msg: str) -> None:
        print(f"[{self.cfg.name} s{self.cfg.seed} d{self.counters['decisions']}] {msg}", flush=True)

    def _write_metadata(self, argv: list[str]) -> None:
        counts = self.model.parameter_counts()
        used = trained_modules(self.cfg)
        write_json(
            self.run_dir / "metadata.json",
            {
                "name": self.cfg.name,
                "variant": self.cfg.loss.variant_name(),
                "seed": self.cfg.seed,
                "device": str(self.device),
                "torch_threads": torch.get_num_threads(),
                "versions": runtime_versions(),
                "env": self.env_meta,
                "num_actions": self.env.num_actions,
                "latent_shape": list(self.model.latent_shape),
                "parameter_counts": counts,
                "trained_modules": used,
                "trained_parameters": sum(counts[m] for m in used),
                "command": " ".join(shlex.quote(a) for a in argv),
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        save_config(self.cfg, self.run_dir / "config.yaml")

    @torch.no_grad()
    def _greedy(self, history: np.ndarray) -> int:
        obs = torch.from_numpy(history).to(self.device).unsqueeze(0)
        return int(self.model.q_values(obs).argmax(dim=-1))

    def _evaluate(self) -> None:
        t = self.cfg.train
        for name in t.eval_controllers:
            result = run_evaluation(
                self.model, self.cfg, self.env_meta, self.device, name, t.eval_episodes,
                self.cfg.eval.seed_base, t.eval_max_episode_decisions, self.cfg.eval.epsilon,
            )
            self.counters["eval_decisions"] += result["total_decisions"]
            self.counters["eval_emulator_frames"] += result["total_emulator_frames"]
            record = {k: v for k, v in result.items() if k != "records"}
            record["episode_returns"] = [r["raw_return"] for r in result["records"]]
            record.update({k: self.counters[k] for k in ("decisions", "updates")})
            self.eval_log.write(record)
            self._log(
                f"eval [{name}] return {result['return_mean']:+.2f} ± {result['return_std']:.2f} "
                f"({t.eval_episodes} episodes, {result['total_decisions']} eval decisions)"
            )

    def _update(self, agg: dict[str, list[float]]) -> None:
        cfg = self.cfg
        batch = self.replay.sample(cfg.optim.batch_size, cfg.replay.rollout_steps, self.replay_rng)
        metrics = self.learner.update(batch.to_torch(self.device))
        c = self.counters
        c["updates"] += 1
        c["sampled_sequences"] += cfg.optim.batch_size
        c["sampled_valid_transitions"] += int(metrics["valid_transitions"])
        c["sampled_latent_targets"] += int(metrics["latent_targets"])
        for k, v in metrics.items():
            agg[k].append(v)
        if c["updates"] % cfg.train.log_every_updates == 0:
            record = {k: float(np.mean(v)) for k, v in agg.items()}
            record.update(
                {
                    "decisions": c["decisions"],
                    "updates": c["updates"],
                    "epsilon": epsilon_at(c["decisions"], cfg.train),
                    "sampled_valid_transitions": c["sampled_valid_transitions"],
                    "wall_time_s": c["wall_time_s"] + time.perf_counter() - self._t0,
                    "lambda_q": cfg.loss.lambda_q,
                    "lambda_jepa": cfg.loss.lambda_jepa if cfg.loss.jepa else 0.0,
                    "lambda_reward": cfg.loss.lambda_reward if cfg.loss.reward else 0.0,
                    "lambda_continue": cfg.loss.lambda_continue if cfg.loss.continuation else 0.0,
                    "lambda_var": cfg.loss.lambda_var if cfg.loss.variance else 0.0,
                    "lambda_inverse": cfg.loss.lambda_inverse if cfg.loss.inverse != "none" else 0.0,
                }
            )
            self.updates_log.write(record)
            parts = [f"{k}={record[k]:.4f}" for k in
                     ("loss_q", "loss_jepa", "loss_reward", "loss_continue", "loss_var", "loss_inverse",
                      "inverse_acc", "latent_std_mean")
                     if k in record]
            self._log(f"update {c['updates']}: " + " ".join(parts))
            agg.clear()

    # ------------------------------------------------------------------------------------------ loop
    def run(self, argv: list[str]) -> None:
        cfg, t, c = self.cfg, self.cfg.train, self.counters
        if c["resume_count"] == 0:
            self._write_metadata(argv)
        else:
            JsonlWriter(self.run_dir / "resumes.jsonl").write(
                {"resume_count": c["resume_count"], "decisions": c["decisions"], "updates": c["updates"],
                 "replay_frames": len(self.replay), "warmup_until": c["warmup_until"],
                 "command": " ".join(shlex.quote(a) for a in argv),
                 "at": datetime.now(timezone.utc).isoformat()}
            )
        self._log(
            f"variant {cfg.loss.variant_name()} on {self.device}; env {cfg.env.id} "
            f"(sticky={cfg.env.sticky_action_prob}); actions {self.env.action_meanings}; "
            f"budget {t.total_decisions} decisions, warmup until {c['warmup_until']}"
        )
        self._t0 = time.perf_counter()
        frames_base = c["train_emulator_frames"]  # env.total_frames restarts at 0 in this process
        frame, info = self.env.reset(seed=_collector_seed(cfg, c["resume_count"]))
        history = self.stacker.reset(frame)
        self.replay.start_episode(frame)
        ep = {"raw": 0.0, "clipped": 0.0, "len": 0, "info": info, "frames_start": 0}
        agg: dict[str, list[float]] = defaultdict(list)

        while c["decisions"] < t.total_decisions:
            if c["decisions"] < c["warmup_until"]:
                action = int(self.explore_rng.integers(self.env.num_actions))
            elif self.explore_rng.random() < epsilon_at(c["decisions"], t):
                action = int(self.explore_rng.integers(self.env.num_actions))
            else:
                action = self._greedy(history)

            frame, reward, terminated, truncated, _ = self.env.step(action)
            self.replay.add(action, reward, terminated, truncated, frame)
            history = self.stacker.push(frame)
            c["decisions"] += 1
            ep["raw"] += reward
            ep["clipped"] += float(clip_reward(reward))
            ep["len"] += 1

            if c["decisions"] > c["warmup_until"] and c["decisions"] % t.update_every == 0:
                self._update(agg)

            if terminated or truncated:
                c["episodes"] += 1
                self.episodes_log.write(
                    {
                        "episode": c["episodes"],
                        "decisions": c["decisions"],
                        "length": ep["len"],
                        "raw_return": ep["raw"],
                        "clipped_return": ep["clipped"],
                        "emulator_frames": self.env.total_frames - ep["frames_start"],
                        "reset_noops": ep["info"]["reset_noops"],
                        "reset_fire": ep["info"]["reset_fire"],
                        "terminated": terminated,
                        "truncated": truncated,
                        "updates": c["updates"],
                        "epsilon": epsilon_at(c["decisions"], t),
                    }
                )
                self._log(f"episode {c['episodes']}: return {ep['raw']:+.0f}, {ep['len']} decisions")
                frames_start = self.env.total_frames
                frame, info = self.env.reset()
                history = self.stacker.reset(frame)
                self.replay.start_episode(frame)
                ep = {"raw": 0.0, "clipped": 0.0, "len": 0, "info": info, "frames_start": frames_start}
            c["train_emulator_frames"] = frames_base + self.env.total_frames

            done = c["decisions"] >= t.total_decisions
            if done or c["decisions"] % t.eval_every_decisions == 0:
                self._evaluate()
            if done or c["decisions"] % t.checkpoint_every_decisions == 0:
                c["wall_time_s"] += time.perf_counter() - self._t0
                self._t0 = time.perf_counter()
                self.save()
                self._log(f"checkpoint saved ({c['updates']} updates, {c['wall_time_s']:.0f}s)")
        self.env.close()
        self._log(
            f"done: {c['decisions']} decisions, {c['updates']} updates, {c['episodes']} episodes, "
            f"{c['train_emulator_frames']} training frames, {c['eval_decisions']} eval decisions"
        )


def _validate(cfg: Config) -> None:
    if "lookahead" in cfg.train.eval_controllers and not cfg.loss.trains_model:
        raise ValueError("train.eval_controllers includes lookahead but the model is not trained for it")
    for name in cfg.train.eval_controllers:
        if name not in ("q", "lookahead"):
            raise ValueError(f"unknown eval controller {name!r}")
    if cfg.replay.rollout_steps < 1:
        raise ValueError("replay.rollout_steps must be >= 1")


def main(argv: list[str] | None = None) -> None:
    argv_full = [sys.executable, "-m", "atari_jepa.train", *(argv if argv is not None else sys.argv[1:])]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML experiment config")
    p.add_argument("--resume", help="run directory containing checkpoint.pt to resume")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help="auto | cpu | cuda | mps (overrides config)")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    p.add_argument("--auto-resume", action="store_true",
                   help="with --config: resume the run directory if it has an unfinished checkpoint, "
                        "exit if it is already complete")
    args = p.parse_args(argv)

    if args.auto_resume and args.config and not args.resume:
        cfg = load_config(args.config, args.set)
        if args.seed is not None:
            cfg.seed = args.seed
        if args.run_dir is not None:
            cfg.run_dir = args.run_dir
        ckpt_path = cfg.resolved_run_dir() / "checkpoint.pt"
        if ckpt_path.exists():
            done = load_checkpoint(ckpt_path)["counters"]["decisions"]
            if done >= cfg.train.total_decisions:
                print(f"{cfg.resolved_run_dir()} is complete ({done} decisions); nothing to do")
                return
            args.resume = str(cfg.resolved_run_dir())
            args.set = [s for s in args.set if s.startswith(("train.total_decisions", "torch_threads", "device"))]

    resume = None
    if args.resume:
        run_dir = Path(args.resume)
        resume = load_checkpoint(run_dir / "checkpoint.pt")
        data = resume["config"]
        for item in args.set:
            key, raw = item.split("=", 1)
            _set_nested(data, key, yaml.safe_load(raw))
            print(f"resume override: {key}={raw}")
        cfg = config_from_dict(data)
    else:
        if not args.config:
            p.error("--config is required unless --resume is given")
        cfg = load_config(args.config, args.set)
        if args.seed is not None:
            cfg.seed = args.seed
        if args.run_dir is not None:
            cfg.run_dir = args.run_dir
        run_dir = cfg.resolved_run_dir()
        if (run_dir / "checkpoint.pt").exists():
            raise SystemExit(f"{run_dir} already has a checkpoint; use --resume {run_dir} or another --run-dir")
    if args.device is not None:
        cfg.device = args.device
    _validate(cfg)
    configure_threads(cfg.torch_threads)
    device = select_device(cfg.device)
    trainer = Trainer(cfg, run_dir, device, resume=resume)
    trainer.run(argv_full)


if __name__ == "__main__":
    main()
