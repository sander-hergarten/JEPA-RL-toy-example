"""Checkpoint save/load and compatibility checks.

A checkpoint stores online and target networks, optimizer state, the full configuration, counters,
exploration-schedule state, RNG states, and environment metadata (action meanings, preprocessing,
library versions). Emulator and collector state are *not* saved: resuming resets the collector and
starts a new replay episode, which is a valid training resume but not a bit-for-bit continuation.
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .agent import WorldModel
from .config import Config, config_from_dict

FORMAT_VERSION = 1


def rng_state(explore_rng: np.random.Generator, replay_rng: np.random.Generator) -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "explore": explore_rng.bit_generator.state,
        "replay": replay_rng.bit_generator.state,
    }


def restore_rng(state: dict[str, Any], explore_rng: np.random.Generator, replay_rng: np.random.Generator) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    explore_rng.bit_generator.state = state["explore"]
    replay_rng.bit_generator.state = state["replay"]


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomic write: a crash mid-save cannot corrupt the previous checkpoint."""
    path = Path(path)
    payload = {"format_version": FORMAT_VERSION, **payload}
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {ckpt.get('format_version')} in {path}")
    return ckpt


def model_from_checkpoint(ckpt: dict[str, Any], device: torch.device) -> tuple[Config, WorldModel]:
    cfg = config_from_dict(ckpt["config"])
    model = WorldModel(cfg, num_actions=len(ckpt["env"]["action_meanings"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return cfg, model


COMPAT_KEYS = ("id", "action_meanings", "observation_shape")


def check_compatible(saved: dict[str, Any], current: dict[str, Any]) -> None:
    """Refuse a checkpoint for a different game, action set, observation shape or preprocessing."""
    for key in COMPAT_KEYS:
        if saved.get(key) != current.get(key):
            raise ValueError(f"checkpoint/environment mismatch in {key}: {saved.get(key)!r} != {current.get(key)!r}")
    sp, cp = saved.get("preprocessing", {}), current.get("preprocessing", {})
    for key in sorted(set(sp) | set(cp)):
        if sp.get(key) != cp.get(key):
            raise ValueError(f"preprocessing mismatch in {key}: checkpoint {sp.get(key)!r} vs env {cp.get(key)!r}")
    for key in ("gymnasium_version", "ale_py_version"):
        if saved.get(key) != current.get(key):
            warnings.warn(f"{key} differs: checkpoint {saved.get(key)} vs installed {current.get(key)}")
