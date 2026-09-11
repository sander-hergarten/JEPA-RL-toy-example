"""Encoder, action-conditioned latent dynamics, and prediction heads.

Tensor contracts (defaults):

* ``Encoder``:   ``uint8 [B, 4, 84, 84]`` -> ``float [B, 64, 7, 7]`` (layer-normalized)
* ``Dynamics``:  ``([B, 64, 7, 7], int64 [B])`` -> ``[B, 64, 7, 7]``
* ``QHead``:     ``[B, 64, 7, 7]`` -> ``[B, A]``
* ``RewardHead``: ``([B, 64, 7, 7], [B])`` -> logits ``[B, 3]`` over rewards ``[-1, 0, +1]``
* ``ContinuationHead``: ``([B, 64, 7, 7], [B])`` -> logits ``[B]``

Heads flatten their latent input, so they also work for other latent shapes (used by tests).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import NetworkConfig

REWARD_VALUES = (-1.0, 0.0, 1.0)  # class index -> clipped reward


def conv_out_size(size: int) -> int:
    for kernel, stride in ((8, 4), (4, 2), (3, 1)):
        size = (size - kernel) // stride + 1
    return size


class Encoder(nn.Module):
    """Nature-DQN CNN followed by LayerNorm over the whole feature map. Each stack is encoded alone."""

    def __init__(self, in_channels: int, channels: list[int], screen_size: int):
        super().__init__()
        c1, c2, c3 = channels
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        side = conv_out_size(screen_size)
        self.latent_shape = (c3, side, side)
        self.norm = nn.LayerNorm(self.latent_shape)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float() / 255.0
        return self.norm(self.convs(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.relu(h + self.conv2(F.relu(self.conv1(h))))


class Dynamics(nn.Module):
    """g(z, a) = LayerNorm(z + delta(z, a)); the action embedding is broadcast over the grid."""

    def __init__(self, latent_shape: tuple[int, int, int], num_actions: int, cfg: NetworkConfig):
        super().__init__()
        channels = latent_shape[0]
        self.action_embed = nn.Embedding(num_actions, cfg.action_embed_dim)
        self.conv_in = nn.Conv2d(channels + cfg.action_embed_dim, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(cfg.dynamics_blocks)])
        self.conv_out = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.LayerNorm(latent_shape)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        emb = self.action_embed(action)[:, :, None, None].expand(-1, -1, *z.shape[2:])
        h = F.relu(self.conv_in(torch.cat([z, emb], dim=1)))
        delta = self.conv_out(self.blocks(h))
        return self.norm(z + delta)


class QHead(nn.Module):
    def __init__(self, latent_dim: int, num_actions: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_actions))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.flatten(1))


class ActionHead(nn.Module):
    """MLP on [flatten(z), one_hot(a)]."""

    def __init__(self, latent_dim: int, num_actions: int, hidden: int, out: int):
        super().__init__()
        self.num_actions = num_actions
        self.net = nn.Sequential(nn.Linear(latent_dim + num_actions, hidden), nn.ReLU(), nn.Linear(hidden, out))

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(action, self.num_actions).to(z.dtype)
        return self.net(torch.cat([z.flatten(1), one_hot], dim=1))


class RewardHead(ActionHead):
    def __init__(self, latent_dim: int, num_actions: int, hidden: int):
        super().__init__(latent_dim, num_actions, hidden, len(REWARD_VALUES))


class ContinuationHead(ActionHead):
    def __init__(self, latent_dim: int, num_actions: int, hidden: int):
        super().__init__(latent_dim, num_actions, hidden, 1)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return super().forward(z, action).squeeze(-1)
