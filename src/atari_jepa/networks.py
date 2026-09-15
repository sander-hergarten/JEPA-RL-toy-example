"""Encoder, action-conditioned latent dynamics, and prediction heads.

Tensor contracts (defaults):

* ``Encoder``:   ``uint8 [B, 4, 84, 84]`` -> ``float [B, 64, 7, 7]`` (layer-normalized); with
  ``motion_channels`` the 3 signed differences of consecutive frames are appended to the input
* ``Dynamics``:  ``([B, 64, 7, 7], int64 [B])`` -> ``[B, 64, 7, 7]``
* ``QHead``:     ``[B, 64, 7, 7]`` -> ``[B, A]``
* ``RewardHead``: ``([B, 64, 7, 7], [B])`` -> logits ``[B, 3]`` over rewards ``[-1, 0, +1]``
* ``ContinuationHead``: ``([B, 64, 7, 7], [B])`` -> logits ``[B]``
* ``InverseDynamicsHead``: ``([B, 64, 7, 7], [B, 64, 7, 7])`` -> action logits ``[B, A]`` (optional)

Heads flatten their latent input, so they also work for other latent shapes (used by tests).

With ``conv_dtype="bfloat16"`` only the convolution stacks run under autocast; their outputs are cast
back to float32 before LayerNorm and the residual sum, so latents, heads and losses are float32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import NetworkConfig

REWARD_VALUES = (-1.0, 0.0, 1.0)  # class index -> clipped reward

_DTYPES = {"float32": None, "bfloat16": torch.bfloat16}


def conv_autocast(x: torch.Tensor, dtype: torch.dtype | None):
    """Autocast context for a conv trunk (a no-op for float32)."""
    return torch.autocast(device_type=x.device.type, dtype=dtype, enabled=dtype is not None)


def conv_out_size(size: int) -> int:
    for kernel, stride in ((8, 4), (4, 2), (3, 1)):
        size = (size - kernel) // stride + 1
    return size


class Encoder(nn.Module):
    """Nature-DQN CNN followed by LayerNorm over the whole feature map. Each stack is encoded alone."""

    def __init__(self, in_channels: int, channels: list[int], screen_size: int, conv_dtype: str = "float32",
                 motion_channels: bool = False):
        super().__init__()
        self.compute_dtype = _DTYPES[conv_dtype]
        self.motion_channels = motion_channels
        self.history = in_channels
        if motion_channels:
            in_channels += self.history - 1
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
        if self.motion_channels:  # signed frame differences: the ball is what moves
            x = torch.cat([x, x[:, 1:] - x[:, :-1]], dim=1)
        with conv_autocast(x, self.compute_dtype):
            h = self.convs(x)
        return self.norm(h.float())


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
        self.compute_dtype = _DTYPES[cfg.conv_dtype]
        self.action_embed = nn.Embedding(num_actions, cfg.action_embed_dim)
        self.conv_in = nn.Conv2d(channels + cfg.action_embed_dim, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(cfg.dynamics_blocks)])
        self.conv_out = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.LayerNorm(latent_shape)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        emb = self.action_embed(action)[:, :, None, None].expand(-1, -1, *z.shape[2:])
        with conv_autocast(z, self.compute_dtype):
            h = F.relu(self.conv_in(torch.cat([z, emb], dim=1)))
            delta = self.conv_out(self.blocks(h))
        return self.norm(z + delta.float())


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


class InverseDynamicsHead(nn.Module):
    """MLP on [flatten(z_t), flatten(z_t+1)] -> logits over the action taken between them."""

    def __init__(self, latent_dim: int, num_actions: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_actions))

    def forward(self, z: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z.flatten(1), z_next.flatten(1)], dim=1))


class MacroDynamics(nn.Module):
    """Jumpy ("hierarchical") dynamics: predict the latent H steps ahead in one shot.

    Conditioned on the whole action sequence rather than a single action, so planning can score any
    candidate sequence with one forward pass instead of unrolling the one-step model H times. This is
    the temporal-abstraction level of an H-JEPA-style hierarchy: level 1 predicts single steps, level 2
    skips over H of them.
    """

    def __init__(self, latent_shape: tuple[int, int, int], num_actions: int, horizon: int, cfg: NetworkConfig):
        super().__init__()
        channels = latent_shape[0]
        self.num_actions, self.horizon = num_actions, horizon
        self.compute_dtype = _DTYPES[cfg.conv_dtype]
        self.seq_embed = nn.Linear(num_actions * horizon, cfg.action_embed_dim)
        self.conv_in = nn.Conv2d(channels + cfg.action_embed_dim, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(cfg.dynamics_blocks)])
        self.conv_out = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.LayerNorm(latent_shape)

    def embed(self, actions: torch.Tensor) -> torch.Tensor:
        """``[B, H]`` actions -> ``[B, embed]``; the sequence is flattened one-hot."""
        one_hot = F.one_hot(actions, self.num_actions).flatten(1).to(self.seq_embed.weight.dtype)
        return self.seq_embed(one_hot)

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        emb = self.embed(actions)[:, :, None, None].expand(-1, -1, *z.shape[2:])
        with conv_autocast(z, self.compute_dtype):
            h = F.relu(self.conv_in(torch.cat([z, emb], dim=1)))
            delta = self.conv_out(self.blocks(h))
        return self.norm(z + delta.float())


class MacroHead(nn.Module):
    """MLP on [flatten(z), one_hot sequence] -> a scalar (macro return, or a continuation logit)."""

    def __init__(self, latent_dim: int, num_actions: int, horizon: int, hidden: int):
        super().__init__()
        self.num_actions, self.horizon = num_actions, horizon
        self.net = nn.Sequential(nn.Linear(latent_dim + num_actions * horizon, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(actions, self.num_actions).flatten(1).to(z.dtype)
        return self.net(torch.cat([z.flatten(1), one_hot], dim=1)).squeeze(-1)
