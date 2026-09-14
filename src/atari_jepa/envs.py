"""Environment contract: one agent decision per ``step``, processed single frames out.

``make_env`` returns an object with

* ``reset(seed=None) -> (frame, info)`` with ``frame`` a ``uint8 [S, S]`` processed observation,
* ``step(action) -> (frame, raw_reward, terminated, truncated, info)`` where ``raw_reward`` is the
  unclipped game reward summed over the repeated action and ``frame`` is the actual observation after
  the action (never a reset observation),
* ``action_meanings`` / ``num_actions`` and ``metadata()``.

History stacking is done by :class:`FrameStacker` for acting and by the replay buffer for training, so
that both use exactly the same padding convention (repeat the first observation of the episode).

The ALE wrapper is written here instead of using ``gymnasium.wrappers.AtariPreprocessing`` because
that wrapper (as of Gymnasium 1.x) hard-codes action index 0 for reset no-ops and returns stale
max-pool frames when the episode ends in the middle of an action repeat.
"""

from __future__ import annotations

import collections
from typing import Any, Protocol

import numpy as np

from .config import EnvConfig


class SetupError(RuntimeError):
    """A missing dependency or ROM; the message says how to fix it."""


class Env(Protocol):
    action_meanings: list[str]

    @property
    def num_actions(self) -> int: ...

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]: ...

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]: ...

    def metadata(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def find_action(meanings: list[str], meaning: str) -> int:
    """Index of the action with the given meaning; raises instead of guessing an id."""
    try:
        return meanings.index(meaning)
    except ValueError:
        raise ValueError(f"action {meaning!r} is not in the action set {meanings}") from None


def clip_reward(raw_reward: float | np.ndarray) -> np.ndarray:
    """Training reward r_t = sign(r_raw_t)."""
    return np.sign(raw_reward).astype(np.float32)


class FrameStacker:
    """Causal history x_t = (o_{t-h+1}, ..., o_t), padded with the episode's first observation."""

    def __init__(self, history: int):
        self.history = history
        self._frames: collections.deque[np.ndarray] = collections.deque(maxlen=history)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self._frames.clear()
        for _ in range(self.history):
            self._frames.append(frame)
        return self.stack()

    def push(self, frame: np.ndarray) -> np.ndarray:
        self._frames.append(frame)
        return self.stack()

    def stack(self) -> np.ndarray:
        return np.stack(self._frames, axis=0)


class AtariEnv:
    """ALE environment with action repeat, 2-frame max-pool, grayscale, resize, no-op/FIRE reset."""

    def __init__(self, cfg: EnvConfig):
        try:
            import ale_py
            import cv2
            import gymnasium as gym
        except ImportError as exc:
            raise SetupError(
                f"missing dependency ({exc.name}). Install the package with `uv sync --extra cpu` "
                "(or --extra cu130), which pulls in gymnasium, ale-py and opencv-python-headless."
            ) from exc
        self._cv2 = cv2
        gym.register_envs(ale_py)
        try:
            self._env = gym.make(
                cfg.id,
                frameskip=cfg.frame_skip,
                repeat_action_probability=cfg.sticky_action_prob,
                full_action_space=cfg.full_action_space,
                max_num_frames_per_episode=cfg.max_episode_frames,
            )
        except Exception as exc:  # ALE raises several error types for missing ROMs / namespaces
            raise SetupError(
                f"could not create {cfg.id!r}: {exc}. ale-py >= 0.9 ships the Atari ROMs in its wheel; "
                "check that ale-py is installed in this environment (`python -c 'import ale_py'`) and "
                "that the id is an ALE v5 id such as 'ALE/Pong-v5'."
            ) from exc
        self.cfg = cfg
        self._ale = self._env.unwrapped.ale
        self.action_meanings: list[str] = list(self._env.unwrapped.get_action_meanings())
        self._noop = find_action(self.action_meanings, "NOOP")
        needs_fire = cfg.fire_on_reset or cfg.fire_on_life_loss
        self._fire = find_action(self.action_meanings, "FIRE") if needs_fire else None
        self._lives = 0
        h, w = self._env.observation_space.shape[:2]
        self._screens = np.zeros((2, h, w), dtype=np.uint8)
        self._rng = np.random.default_rng()
        self._seeded = False
        self._episode_decisions = 0
        self.total_frames = 0  # emulator frames, including reset no-ops and FIRE presses
        self.frame_sink = None  # optional callable(rgb screen) per emulator frame, for recording

    @property
    def num_actions(self) -> int:
        return len(self.action_meanings)

    def _act_frame(self, action: int) -> tuple[float, bool, bool]:
        _, reward, terminated, truncated, _ = self._env.step(action)
        self.total_frames += 1
        if self.frame_sink is not None:
            self.frame_sink(self._ale.getScreenRGB())
        self._screens[0] = self._screens[1]
        self._ale.getScreenGrayscale(self._screens[1])
        return float(reward), bool(terminated), bool(truncated)

    def _processed(self) -> np.ndarray:
        pooled = np.maximum(self._screens[0], self._screens[1])
        size = self.cfg.screen_size
        return self._cv2.resize(pooled, (size, size), interpolation=self._cv2.INTER_AREA)

    def _reset_game(self, seed: int | None) -> None:
        self._env.reset(seed=seed)
        self._ale.getScreenGrayscale(self._screens[1])
        self._screens[0] = self._screens[1]

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._reset_game(seed)
            self._seeded = True
        elif not self._seeded:
            raise RuntimeError("the first reset must be seeded")
        else:
            self._reset_game(None)
        info: dict[str, Any] = {"reset_noops": 0, "reset_fire": 0, "reset_frames": 0}
        start_frames = self.total_frames
        noops = int(self._rng.integers(1, self.cfg.noop_max + 1)) if self.cfg.noop_max > 0 else 0
        for _ in range(noops):
            _, term, trunc = self._act_frame(self._noop)
            if term or trunc:
                self._reset_game(None)
        info["reset_noops"] = noops
        if self.cfg.fire_on_reset:
            _, term, trunc = self._act_frame(self._fire)
            if term or trunc:
                self._reset_game(None)
            info["reset_fire"] = 1
        info["reset_frames"] = self.total_frames - start_frames
        self._episode_decisions = 0
        self._lives = self._ale.lives()
        return self._processed(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        total = 0.0
        terminated = truncated = False
        frames = 0
        for _ in range(self.cfg.action_repeat):
            reward, terminated, truncated = self._act_frame(int(action))
            total += reward
            frames += 1
            if terminated or truncated:
                break  # screens hold the last two frames actually emitted
        fire_frames = 0
        if self.cfg.fire_on_life_loss and not (terminated or truncated):
            lives = self._ale.lives()
            if lives < self._lives:  # serve the next ball; otherwise a non-firing policy idles forever
                reward, terminated, truncated = self._act_frame(self._fire)
                total += reward
                frames += 1
                fire_frames = 1
            self._lives = lives
        self._episode_decisions += 1
        limit = self.cfg.max_episode_decisions
        if limit is not None and self._episode_decisions >= limit and not terminated:
            truncated = True
        return self._processed(), total, terminated, truncated, {"frames": frames, "fire_frames": fire_frames}

    def state_vector(self) -> np.ndarray:
        """Emulator RAM (128 bytes). Evaluation-only: never an input to the agent or its training."""
        return self._ale.getRAM().astype(np.int64)

    def metadata(self) -> dict[str, Any]:
        import ale_py
        import gymnasium as gym

        return {
            "kind": "ale",
            "id": self.cfg.id,
            "gymnasium_version": gym.__version__,
            "ale_py_version": ale_py.__version__,
            "action_meanings": self.action_meanings,
            "observation_shape": [self.cfg.history, self.cfg.screen_size, self.cfg.screen_size],
            "preprocessing": {
                "base_frame_skip": self.cfg.frame_skip,
                "action_repeat": self.cfg.action_repeat,
                "max_pool_last_two_frames": True,
                "grayscale": "ALE getScreenGrayscale",
                "resize": f"cv2.INTER_AREA to {self.cfg.screen_size}x{self.cfg.screen_size}",
                "history": self.cfg.history,
                "history_padding": "repeat first episode observation",
                "noop_max": self.cfg.noop_max,
                "noop_distribution": "uniform in [1, noop_max], seeded per reset seed",
                "fire_on_reset": self.cfg.fire_on_reset,
                "fire_on_life_loss": self.cfg.fire_on_life_loss,
                "sticky_action_prob": self.cfg.sticky_action_prob,
                "full_action_space": self.cfg.full_action_space,
                "terminal_on_life_loss": self.cfg.terminal_on_life_loss,
                "max_episode_frames": self.cfg.max_episode_frames,
                "max_episode_decisions": self.cfg.max_episode_decisions,
                "reward_transform": self.cfg.reward_transform,
            },
        }

    def close(self) -> None:
        self._env.close()


class CatchEnv:
    """ROM-free synthetic game with the same contract, for tests and ROM-free smoke runs.

    A 6x6 ball falls one grid cell per decision; the agent moves a paddle on the bottom row.
    Catching gives +1, missing -1; the episode terminates after ``balls`` drops. It is deliberately tiny
    and fully observable so tests can check learning plumbing quickly.
    """

    CELL = 6  # pixels per grid cell
    GRID = 14  # 14 * 6 = 84

    def __init__(self, cfg: EnvConfig, balls: int = 5):
        if cfg.screen_size != self.CELL * self.GRID:
            raise ValueError("synthetic:catch requires screen_size 84")
        self.cfg = cfg
        self.balls = balls
        self.action_meanings = ["NOOP", "FIRE", "RIGHT", "LEFT"]
        self._rng = np.random.default_rng()
        self._seeded = False
        self.total_frames = 0
        self._last_action = 0

    @property
    def num_actions(self) -> int:
        return len(self.action_meanings)

    def _render(self) -> np.ndarray:
        img = np.zeros((84, 84), dtype=np.uint8)
        c = self.CELL
        img[self.ball_row * c : (self.ball_row + 1) * c, self.ball_col * c : (self.ball_col + 1) * c] = 255
        img[84 - 3 : 84, self.paddle * c : (self.paddle + 1) * c] = 160
        return img

    def _new_ball(self) -> None:
        self.ball_row = int(self._rng.integers(0, 4))  # varied drop lengths -> varied episode lengths
        self.ball_col = int(self._rng.integers(0, self.GRID))

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seeded = True
        elif not self._seeded:
            raise RuntimeError("the first reset must be seeded")
        self.paddle = self.GRID // 2
        self.dropped = 0
        self.decisions = 0
        self._last_action = 0
        self._new_ball()
        return self._render(), {"reset_noops": 0, "reset_fire": 0, "reset_frames": 0}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._rng.random() < self.cfg.sticky_action_prob:
            action = self._last_action
        self._last_action = int(action)
        move = {"RIGHT": 1, "LEFT": -1}.get(self.action_meanings[int(action)], 0)
        self.paddle = int(np.clip(self.paddle + move, 0, self.GRID - 1))
        self.ball_row += 1
        self.decisions += 1
        self.total_frames += 1
        reward = 0.0
        if self.ball_row == self.GRID - 1:
            reward = 1.0 if self.ball_col == self.paddle else -1.0
            self.dropped += 1
            self._new_ball()
        terminated = self.dropped >= self.balls
        limit = self.cfg.max_episode_decisions
        truncated = bool(limit is not None and self.decisions >= limit and not terminated)
        return self._render(), reward, terminated, truncated, {"frames": 1}

    def state_vector(self) -> np.ndarray:
        """True state of the toy game; evaluation-only, like the ALE RAM."""
        return np.array([self.ball_row, self.ball_col, self.paddle], dtype=np.int64)

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "synthetic",
            "id": self.cfg.id,
            "action_meanings": self.action_meanings,
            "observation_shape": [self.cfg.history, self.cfg.screen_size, self.cfg.screen_size],
            "preprocessing": {
                "history": self.cfg.history,
                "history_padding": "repeat first episode observation",
                "sticky_action_prob": self.cfg.sticky_action_prob,
                "max_episode_decisions": self.cfg.max_episode_decisions,
                "reward_transform": self.cfg.reward_transform,
                "balls": self.balls,
            },
        }

    def close(self) -> None:
        pass


def make_env(cfg: EnvConfig) -> Env:
    if cfg.id == "synthetic:catch":
        return CatchEnv(cfg)
    if cfg.id.startswith("synthetic:"):
        raise ValueError(f"unknown synthetic env {cfg.id!r}")
    return AtariEnv(cfg)


def env_metadata_for(cfg: EnvConfig) -> dict[str, Any]:
    env = make_env(cfg)
    try:
        return env.metadata()
    finally:
        env.close()
