"""Sequence replay over a ring buffer of processed frames.

Each slot holds one processed observation ``o_t`` plus the transition taken after it (if any). An
episode of ``T`` decisions occupies ``T + 1`` consecutive slots: slots for ``t = 0..T-1`` carry
transitions, and the final slot holds the actual final observation (terminal or truncated) with
``has_transition = False``. Stacked histories are reconstructed at sample time, padding the start of an
episode with its first observation, so no overlapping stacks or latents are stored.

Every slot also stores its episode id and absolute write index, so ordering survives eviction and a
sampled sequence can never cross into another episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .envs import clip_reward


@dataclass
class SequenceBatch:
    """Masked sequence batch. Invalid (padded) entries are zero and must be masked by ``valid``."""

    observations: np.ndarray  # [B, K+1, H, S, S] uint8
    actions: np.ndarray  # [B, K] int64
    rewards: np.ndarray  # [B, K] float32, clipped training reward sign(r_raw)
    raw_rewards: np.ndarray  # [B, K] float32
    terminated: np.ndarray  # [B, K] bool
    truncated: np.ndarray  # [B, K] bool
    valid: np.ndarray  # [B, K] bool
    roots: np.ndarray  # [B] absolute slot index of each root

    def to_torch(self, device: torch.device) -> TorchBatch:
        return TorchBatch(
            observations=torch.from_numpy(self.observations).to(device),
            actions=torch.from_numpy(self.actions).to(device),
            rewards=torch.from_numpy(self.rewards).to(device),
            terminated=torch.from_numpy(self.terminated).to(device),
            truncated=torch.from_numpy(self.truncated).to(device),
            valid=torch.from_numpy(self.valid).to(device),
        )


@dataclass
class TorchBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    valid: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.actions.shape[0]

    @property
    def horizon(self) -> int:
        return self.actions.shape[1]


class SequenceReplay:
    def __init__(self, capacity: int, frame_shape: tuple[int, int], history: int):
        self.capacity = capacity
        self.history = history
        self.frame_shape = frame_shape
        self.frames = np.zeros((capacity, *frame_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.raw_rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=bool)
        self.truncated = np.zeros(capacity, dtype=bool)
        self.has_transition = np.zeros(capacity, dtype=bool)
        self.episode_id = np.full(capacity, -1, dtype=np.int64)
        self.step_in_episode = np.zeros(capacity, dtype=np.int64)
        self.abs_index = np.full(capacity, -1, dtype=np.int64)
        self.n_written = 0  # absolute index of the next slot to be written
        self.n_episodes = 0
        self._open = False  # whether the latest slot is an unfinished episode's pending observation

    # ----------------------------------------------------------------------------------------- writing
    def _write(self, frame: np.ndarray, episode: int, t: int) -> None:
        slot = self.n_written % self.capacity
        self.frames[slot] = frame
        self.actions[slot] = 0
        self.raw_rewards[slot] = 0.0
        self.terminated[slot] = False
        self.truncated[slot] = False
        self.has_transition[slot] = False
        self.episode_id[slot] = episode
        self.step_in_episode[slot] = t
        self.abs_index[slot] = self.n_written
        self.n_written += 1

    def start_episode(self, frame: np.ndarray) -> None:
        """Store the first observation of a new episode.

        Any unfinished episode (e.g. a collection boundary on resume) is left open-ended: its last slot
        has no transition, so no sequence can extend past it or join the new episode.
        """
        self._write(frame, self.n_episodes, 0)
        self.n_episodes += 1
        self._open = True

    def add(
        self, action: int, raw_reward: float, terminated: bool, truncated: bool, next_frame: np.ndarray
    ) -> None:
        """Record the transition from the latest observation and store the actual next observation."""
        if not self._open:
            raise RuntimeError("add() called without an open episode; call start_episode() first")
        slot = (self.n_written - 1) % self.capacity
        self.actions[slot] = action
        self.raw_rewards[slot] = raw_reward
        self.terminated[slot] = terminated
        self.truncated[slot] = truncated
        self.has_transition[slot] = True
        self._write(next_frame, self.episode_id[slot], self.step_in_episode[slot] + 1)
        if terminated or truncated:
            self._open = False

    # ----------------------------------------------------------------------------------------- reading
    @property
    def oldest(self) -> int:
        return max(0, self.n_written - self.capacity)

    def __len__(self) -> int:
        return self.n_written - self.oldest

    def num_transitions(self) -> int:
        return int(self.has_transition[: min(self.n_written, self.capacity)].sum())

    def _is_valid_root(self, abs_idx: np.ndarray) -> np.ndarray:
        slot = abs_idx % self.capacity
        t = self.step_in_episode[slot]
        history_start = abs_idx - np.minimum(t, self.history - 1)
        return (
            (abs_idx >= self.oldest)
            & (abs_idx < self.n_written)
            & self.has_transition[slot]
            & (self.abs_index[slot] == abs_idx)
            & (history_start >= self.oldest)
        )

    def valid_roots(self) -> np.ndarray:
        candidates = np.arange(self.oldest, self.n_written, dtype=np.int64)
        return candidates[self._is_valid_root(candidates)]

    def sample_roots(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        """Uniform over all valid roots in the buffer (independently, so not one adjacent segment)."""
        if len(self) == 0:
            raise ValueError("replay is empty")
        roots = np.empty(0, dtype=np.int64)
        for _ in range(100):
            cand = rng.integers(self.oldest, self.n_written, size=2 * batch_size)
            roots = np.concatenate([roots, cand[self._is_valid_root(cand)]])
            if len(roots) >= batch_size:
                return roots[:batch_size]
        raise ValueError("could not sample enough valid roots; is the buffer nearly empty?")

    def _stacks(self, abs_idx: np.ndarray) -> np.ndarray:
        """Histories for absolute indices ``abs_idx`` (any shape) -> ``[..., H, S, S]``."""
        t = self.step_in_episode[abs_idx % self.capacity]
        offsets = np.arange(self.history - 1, -1, -1)  # oldest first
        hist = abs_idx[..., None] - np.minimum(offsets, t[..., None])
        return self.frames[hist % self.capacity]

    def gather(self, roots: np.ndarray, horizon: int) -> SequenceBatch:
        """Sequences of ``horizon`` transitions starting at ``roots`` (absolute indices)."""
        roots = np.asarray(roots, dtype=np.int64)
        if not self._is_valid_root(roots).all():
            raise ValueError("gather() received an invalid root")
        B, K = len(roots), horizon
        idx = roots[:, None] + np.arange(K)[None, :]  # transition k lives in slot of obs k
        safe = np.minimum(idx, self.n_written - 1)
        slot = safe % self.capacity
        same_episode = (idx < self.n_written) & (
            self.episode_id[slot] == self.episode_id[roots % self.capacity][:, None]
        )
        valid = same_episode & self.has_transition[slot]
        # Nothing after a termination/truncation is valid, even though truncation continues the task.
        ended = self.terminated[slot] | self.truncated[slot]
        ended_before = np.zeros_like(ended)
        ended_before[:, 1:] = np.cumsum(ended[:, :-1], axis=1) > 0
        valid &= ~ended_before
        valid &= np.cumprod(valid, axis=1).astype(bool)

        obs_valid = np.ones((B, K + 1), dtype=bool)
        obs_valid[:, 1:] = valid
        obs_idx = np.minimum(roots[:, None] + np.arange(K + 1)[None, :], self.n_written - 1)
        observations = self._stacks(obs_idx)
        observations[~obs_valid] = 0

        raw = np.where(valid, self.raw_rewards[slot], 0.0).astype(np.float32)
        return SequenceBatch(
            observations=observations,
            actions=np.where(valid, self.actions[slot], 0),
            rewards=clip_reward(raw),
            raw_rewards=raw,
            terminated=valid & self.terminated[slot],
            truncated=valid & self.truncated[slot],
            valid=valid,
            roots=roots,
        )

    def sample(self, batch_size: int, horizon: int, rng: np.random.Generator) -> SequenceBatch:
        return self.gather(self.sample_roots(batch_size, rng), horizon)

    # ------------------------------------------------------------------------------------- persistence
    _ARRAYS = (
        "frames",
        "actions",
        "raw_rewards",
        "terminated",
        "truncated",
        "has_transition",
        "episode_id",
        "step_in_episode",
        "abs_index",
    )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        tmp = path.with_suffix(".tmp.npz")
        n = min(self.n_written, self.capacity)
        arrays = {name: getattr(self, name)[:n] for name in self._ARRAYS}
        meta = np.array([self.capacity, self.history, self.n_written, self.n_episodes], dtype=np.int64)
        with open(tmp, "wb") as fh:
            np.savez(fh, meta=meta, **arrays)
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path, frame_shape: tuple[int, int]) -> SequenceReplay:
        """Load a saved buffer. The unfinished episode (if any) stays closed-off."""
        with np.load(path) as data:
            capacity, history, n_written, n_episodes = (int(v) for v in data["meta"])
            buf = cls(capacity, frame_shape, history)
            n = min(n_written, capacity)
            for name in cls._ARRAYS:
                getattr(buf, name)[:n] = data[name]
        buf.n_written = n_written
        buf.n_episodes = n_episodes
        buf._open = False
        return buf
