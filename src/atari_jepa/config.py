"""Typed experiment configuration.

Configurations are nested dataclasses loaded from YAML. Unknown keys are rejected so a typo cannot
silently fall back to a default. ``--set section.key=value`` overrides are applied after loading.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    # Gymnasium id. "ALE/..." ids use ALE; "synthetic:catch" is a ROM-free toy game for tests/smoke runs.
    id: str = "ALE/Pong-v5"
    # Must be stated explicitly in every experiment config (primary 0.25, debug 0.0).
    sticky_action_prob: float | None = None
    frame_skip: int = 1  # base-environment frame skipping; repetition happens in our wrapper
    action_repeat: int = 4
    screen_size: int = 84
    history: int = 4
    noop_max: int = 30
    full_action_space: bool = False
    terminal_on_life_loss: bool = False
    # Press the action whose meaning is "FIRE" after the no-ops. Pong serves automatically, so off;
    # Breakout needs it to launch the ball.
    fire_on_reset: bool = False
    # Press FIRE once whenever a life is lost (games like Breakout need a serve after each life, and a
    # greedy policy that never fires would otherwise idle until the emulator's frame limit). The press is
    # counted in the emulator-frame budget and recorded in the run metadata.
    fire_on_life_loss: bool = False
    max_episode_frames: int = 108_000  # ALE internal truncation (emulator frames)
    max_episode_decisions: int | None = None  # external time-limit truncation (agent decisions)
    reward_transform: str = "sign"  # the 3-class reward head only supports "sign"

    def __post_init__(self) -> None:
        if self.sticky_action_prob is None:
            raise ValueError("env.sticky_action_prob must be set explicitly in the config")
        if self.frame_skip != 1:
            raise ValueError("env.frame_skip must be 1; repetition is done by action_repeat")
        if self.terminal_on_life_loss:
            raise ValueError("life-loss termination is not supported in this experiment")
        if self.reward_transform != "sign":
            raise ValueError(
                "only reward_transform='sign' is supported: the reward head is a 3-class "
                "classifier over {-1, 0, +1}; unclipped rewards need a different head"
            )


@dataclass
class NetworkConfig:
    encoder_channels: list[int] = field(default_factory=lambda: [32, 64, 64])
    action_embed_dim: int = 16
    dynamics_blocks: int = 2
    q_hidden: int = 512
    reward_hidden: int = 256
    continuation_hidden: int = 256
    inverse_hidden: int = 256
    # Append signed differences of consecutive frames in the stack as extra encoder input channels.
    # The ball is the only fast small mover, so differencing removes the static background from the input.
    motion_channels: bool = False
    macro_hidden: int = 256
    # Compute dtype for the conv trunks (encoder and dynamics convs) via autocast. Parameters, optimizer
    # state, LayerNorm, MLP heads and all losses stay float32: latent cosine distances are ~1e-2, below
    # bfloat16's resolution near 1.0 (~4e-3), so loss math must not run in bfloat16.
    conv_dtype: str = "float32"

    def encoder_in_channels(self, history: int) -> int:
        return history + (history - 1 if self.motion_channels else 0)

    def __post_init__(self) -> None:
        if self.conv_dtype not in ("float32", "bfloat16"):
            raise ValueError(f"network.conv_dtype must be float32 or bfloat16, got {self.conv_dtype!r}")


@dataclass
class LossConfig:
    # Root Q-loss is always on. The flags below select the ablation variant.
    q_imagined: bool = False  # Q-loss on imagined depths 1..K as well
    # How many imagined depths the Q-loss covers; 0 means all K. Decouples the value head's
    # supervision from the rollout length: the K sweep found that longer rollouts give a strictly
    # better model (lower latent error at every depth) but worse control, and the damage was largest
    # for the shallow controllers, which is what a Q head trained mostly on deep imagined latents
    # predicts. Capping it trains long rollouts without moving the Q supervision deeper.
    q_imagined_depth: int = 0
    jepa: bool = False
    reward: bool = False
    continuation: bool = False
    variance: bool = False
    covariance: bool = False  # optional, off by default
    # Inverse dynamics: predict a_k from a pair of consecutive latents. "real" uses online encodings of
    # the real observations (f(x_k), f(x_k+1)); "predicted" uses (z_hat[k], z_hat[k+1]).
    inverse: str = "none"
    # What the temporal loss compares. "absolute": the latents themselves (the static background
    # dominates the cosine). "batch_centered": latents minus the batch-mean target latent.
    # "delta": the per-step change, so only what moves is predicted.
    jepa_target: str = "absolute"
    # H-JEPA level 2: a jumpy dynamics that predicts `macro_horizon` steps ahead in one shot, with its
    # own macro-return and macro-continuation heads. Enables planning over action sequences without
    # unrolling the one-step model.
    hierarchical: bool = False
    macro_horizon: int = 5
    lambda_macro: float = 1.0
    # Train level 2 on detached level-1 latents. Without this, the jumpy objective pulls the shared
    # encoder toward slowly-varying (action-insensitive) features and destroys level-1 planning.
    macro_detach: bool = True
    lambda_q: float = 1.0
    lambda_jepa: float = 1.0
    lambda_reward: float = 1.0
    lambda_continue: float = 1.0
    lambda_var: float = 1.0
    lambda_cov: float = 0.0
    lambda_inverse: float = 1.0
    gamma: float = 0.99
    # Steps accumulated in the Q target before bootstrapping. 1 = the specified single-step Double DQN;
    # larger n propagates reward faster (capped by the rollout window K at each depth).
    n_step: int = 1
    huber_delta: float = 1.0
    variance_floor: float = 0.1
    variance_eps: float = 1.0e-4
    cosine_eps: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.inverse not in ("none", "real", "predicted"):
            raise ValueError(f"loss.inverse must be none, real or predicted, got {self.inverse!r}")
        if self.n_step < 1:
            raise ValueError(f"loss.n_step must be >= 1, got {self.n_step}")
        if self.hierarchical and self.macro_horizon < 2:
            raise ValueError(f"loss.macro_horizon must be >= 2, got {self.macro_horizon}")
        if self.q_imagined_depth < 0:
            raise ValueError(f"loss.q_imagined_depth must be >= 0 (0 = all K), got {self.q_imagined_depth}")
        if self.jepa_target not in ("absolute", "batch_centered", "delta"):
            raise ValueError(
                f"loss.jepa_target must be absolute, batch_centered or delta, got {self.jepa_target!r}")

    @property
    def needs_rollout(self) -> bool:
        return self.q_imagined or self.jepa or self.reward or self.continuation or self.inverse == "predicted"

    @property
    def trains_model(self) -> bool:
        """Whether dynamics, reward and continuation are all trained (lookahead is meaningful)."""
        return self.jepa and self.reward and self.continuation

    def variant_name(self) -> str:
        suffix = "" if self.inverse == "none" else f"+inverse_{self.inverse}"
        if self.jepa and self.jepa_target != "absolute":
            suffix += f"+jepa_{self.jepa_target}"
        if self.q_imagined and self.q_imagined_depth:
            suffix += f"+qdepth{self.q_imagined_depth}"
        if self.q_imagined and self.jepa and self.reward and self.continuation:
            return "C_world_model" + suffix
        if self.jepa and not (self.reward or self.continuation or self.q_imagined):
            return "B_temporal_jepa" + suffix
        if not self.needs_rollout:
            return "A_q_baseline" + suffix
        return "custom" + suffix


@dataclass
class ReplayConfig:
    capacity: int = 100_000  # stored processed frames (one per decision, plus one final frame per episode)
    rollout_steps: int = 5  # K
    save_with_checkpoint: bool = False


@dataclass
class OptimConfig:
    batch_size: int = 32
    lr: float = 1.0e-4
    adam_eps: float = 1.5e-4
    grad_clip_norm: float = 10.0
    tau: float = 0.99  # target = tau * target + (1 - tau) * online


@dataclass
class TrainConfig:
    total_decisions: int = 100_000  # includes warmup
    warmup_decisions: int = 5_000
    update_every: int = 4
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay_decisions: int = 50_000
    log_every_updates: int = 250
    checkpoint_every_decisions: int = 25_000
    eval_every_decisions: int = 25_000
    eval_episodes: int = 3
    eval_controllers: list[str] = field(default_factory=lambda: ["q"])
    eval_max_episode_decisions: int = 27_000
    # Initialize the encoder (and dynamics) from an offline-pretrained checkpoint, and optionally keep
    # the encoder fixed so RL only learns on top of representations trained by observation.
    init_from: str | None = None
    freeze_encoder: bool = False
    # Policy used to *collect* training data: the greedy Q head, or the model-based lookahead
    # controller (each run always collects with its own network).
    collect_controller: str = "q"
    # Decisions to keep collecting with the Q-policy before switching to collect_controller. Collecting
    # with an untrained planner is worse than not planning at all, so the switch can be delayed until
    # the model is good.
    collect_switch_decisions: int = 0
    # Shrink-and-perturb the Q head (online and target) every N updates: theta <- a*theta + (1-a)*random.
    # Counteracts the plasticity loss that shows up at higher replay ratios. 0 disables it.
    reset_heads_every_updates: int = 0
    reset_shrink: float = 0.5


@dataclass
class PretrainConfig:
    """Offline JEPA pretraining on frames collected by an already-trained RL agent."""

    dataset: str | None = None  # directory written by atari_jepa.collect
    updates: int = 50_000
    batch_size: int = 32
    rollout_steps: int = 5
    lr: float = 1.0e-4
    adam_eps: float = 1.5e-4
    grad_clip_norm: float = 10.0
    tau: float = 0.99
    latent_loss: str = "mse"  # mse | cosine
    jepa_target: str = "absolute"  # absolute | batch_centered | delta
    anti_collapse: str = "sigreg"  # sigreg | vicreg | variance | none
    lambda_jepa: float = 1.0
    lambda_anti: float = 1.0
    sigreg_directions: int = 64
    variance_floor: float = 0.1
    log_every: int = 500

    def __post_init__(self) -> None:
        if self.latent_loss not in ("mse", "cosine"):
            raise ValueError(f"pretrain.latent_loss must be mse or cosine, got {self.latent_loss!r}")
        if self.anti_collapse not in ("sigreg", "vicreg", "variance", "none"):
            raise ValueError(f"unknown pretrain.anti_collapse {self.anti_collapse!r}")
        if self.jepa_target not in ("absolute", "batch_centered", "delta"):
            raise ValueError(f"unknown pretrain.jepa_target {self.jepa_target!r}")


@dataclass
class EvalConfig:
    episodes: int = 10
    seed_base: int = 10_000  # reset seed of eval episode i is seed_base + i (same for all controllers)
    epsilon: float = 0.0
    max_episode_decisions: int = 27_000  # = ALE's 108k-frame limit at action repeat 4


@dataclass
class PlanningConfig:
    horizon: int = 1
    beam_width: int = 16
    # Hierarchical planner: candidate action sequences scored with the level-2 jumpy model.
    macro_candidates: int = 128


@dataclass
class DiagnosticsConfig:
    episodes: int = 4
    max_decisions: int = 20_000
    seed_base: int = 20_000
    policy_epsilon: float = 0.1  # epsilon-greedy Q policy used to collect held-out trajectories
    depths: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    batch_size: int = 64


@dataclass
class Config:
    name: str = "unnamed"
    seed: int = 0
    device: str = "auto"
    torch_threads: int = 0  # 0 = torch default
    run_dir: str | None = None  # default: runs/<name>/seed<seed>
    env: EnvConfig = field(default_factory=lambda: EnvConfig(sticky_action_prob=0.25))
    network: NetworkConfig = field(default_factory=NetworkConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)

    def resolved_run_dir(self) -> Path:
        if self.run_dir is not None:
            return Path(self.run_dir)
        return Path("runs") / self.name / f"seed{self.seed}"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = typing.get_args(tp)
        if value is None and type(None) in args:
            return None
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(arg, value, where)
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        raise TypeError(f"{where}: {value!r} does not match {tp}")
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise TypeError(f"{where}: expected a mapping, got {value!r}")
        return _from_dict(tp, value, where)
    if origin is list:
        (item_tp,) = typing.get_args(tp)
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{where}: expected a list, got {value!r}")
        return [_coerce(item_tp, v, f"{where}[{i}]") for i, v in enumerate(value)]
    if tp is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{where}: expected bool, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{where}: expected int, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{where}: expected float, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise TypeError(f"{where}: expected str, got {value!r}")
        return value
    raise TypeError(f"{where}: unsupported config type {tp}")


def _from_dict(cls: type, data: dict[str, Any], where: str = "config") -> Any:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"{where}: unknown keys {sorted(unknown)}; valid keys: {sorted(names)}")
    kwargs = {k: _coerce(hints[k], v, f"{where}.{k}") for k, v in data.items()}
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any]) -> Config:
    if "env" not in data:
        raise KeyError("config must contain an env section (with an explicit sticky_action_prob)")
    return _from_dict(Config, data)


def _set_nested(data: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise KeyError(f"override {dotted}: {key} is not a section")
    node[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config and apply ``section.key=value`` overrides (values parsed as YAML)."""
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        key, raw = item.split("=", 1)
        _set_nested(data, key, yaml.safe_load(raw))
    return config_from_dict(data)


def save_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
