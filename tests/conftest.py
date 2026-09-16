from __future__ import annotations

import numpy as np
import pytest
import torch

from atari_jepa.agent import WorldModel
from atari_jepa.config import Config, EnvConfig, LossConfig, ReplayConfig


def frame(episode: int, t: int) -> np.ndarray:
    """A frame whose pixels identify (episode, t): [0,0] = episode, [0,1] = t, the rest = t."""
    f = np.full((84, 84), t % 256, dtype=np.uint8)
    f[0, 0] = episode
    f[0, 1] = t
    return f


def ids(stack: np.ndarray) -> list[tuple[int, int]]:
    """(episode, t) of each frame in a [H, 84, 84] stack."""
    return [(int(f[0, 0]), int(f[0, 1])) for f in stack]


def world_model_config(**loss_overrides) -> Config:
    loss = dict(q_imagined=True, jepa=True, reward=True, continuation=True, variance=True)
    loss.update(loss_overrides)
    return Config(
        name="test",
        env=EnvConfig(id="synthetic:catch", sticky_action_prob=0.0),
        loss=LossConfig(**loss),
        replay=ReplayConfig(capacity=1000, rollout_steps=3),
    )


@pytest.fixture
def cfg() -> Config:
    return world_model_config()


@pytest.fixture
def model(cfg) -> WorldModel:
    torch.manual_seed(0)
    return WorldModel(cfg, num_actions=4)


def random_batch(B: int = 4, K: int = 3, A: int = 4, seed: int = 0):
    """A TorchBatch with random frames and a mix of valid, terminal and padded steps."""
    from atari_jepa.replay import TorchBatch

    g = torch.Generator().manual_seed(seed)
    obs = torch.randint(0, 256, (B, K + 1, 4, 84, 84), generator=g, dtype=torch.uint8)
    valid = torch.ones(B, K, dtype=torch.bool)
    terminated = torch.zeros(B, K, dtype=torch.bool)
    truncated = torch.zeros(B, K, dtype=torch.bool)
    # row 1 terminates at step 1; row 2 is truncated at step 0
    terminated[1, 1] = True
    valid[1, 2:] = False
    truncated[2, 0] = True
    valid[2, 1:] = False
    base = torch.tensor([[0.0, 1.0, -1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rewards = torch.zeros(B, K)
    rows, cols = min(B, base.shape[0]), min(K, base.shape[1])
    rewards[:rows, :cols] = base[:rows, :cols]  # extra rows/steps stay zero-reward
    rewards = torch.where(valid, rewards, torch.zeros_like(rewards))
    obs_valid = torch.ones(B, K + 1, dtype=torch.bool)
    obs_valid[:, 1:] = valid
    obs[~obs_valid] = 0
    actions = torch.randint(0, A, (B, K), generator=g)
    actions = torch.where(valid, actions, torch.zeros_like(actions))
    return TorchBatch(obs, actions, rewards, terminated, truncated, valid)
