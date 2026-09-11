"""Online and target networks, EMA targets, and the optimizer step."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import Config
from .networks import ContinuationHead, Dynamics, Encoder, QHead, RewardHead


class WorldModel(nn.Module):
    """Online encoder/dynamics/heads plus EMA copies of the encoder and Q-head.

    There is no target dynamics network: targets for latent prediction come from the target encoder
    applied to real observations.
    """

    def __init__(self, cfg: Config, num_actions: int):
        super().__init__()
        env, net = cfg.env, cfg.network
        self.num_actions = num_actions
        self.encoder = Encoder(env.history, net.encoder_channels, env.screen_size)
        self.latent_shape = self.encoder.latent_shape
        latent_dim = int(np.prod(self.latent_shape))
        self.dynamics = Dynamics(self.latent_shape, num_actions, net)
        self.q_head = QHead(latent_dim, num_actions, net.q_hidden)
        self.reward_head = RewardHead(latent_dim, num_actions, net.reward_hidden)
        self.continuation_head = ContinuationHead(latent_dim, num_actions, net.continuation_hidden)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_q_head = copy.deepcopy(self.q_head)
        for p in self.target_parameters():
            p.requires_grad_(False)

    ONLINE = ("encoder", "dynamics", "q_head", "reward_head", "continuation_head")
    TARGET_PAIRS = (("target_encoder", "encoder"), ("target_q_head", "q_head"))

    def online_parameters(self) -> list[nn.Parameter]:
        return [p for name in self.ONLINE for p in getattr(self, name).parameters()]

    def target_parameters(self) -> list[nn.Parameter]:
        return [p for t, _ in self.TARGET_PAIRS for p in getattr(self, t).parameters()]

    @torch.no_grad()
    def update_targets(self, tau: float) -> None:
        """target = tau * target + (1 - tau) * online, for parameters and buffers."""
        for t_name, o_name in self.TARGET_PAIRS:
            target, online = getattr(self, t_name), getattr(self, o_name)
            t_params = [p for p in target.parameters()]
            o_params = [p for p in online.parameters()]
            torch._foreach_mul_(t_params, tau)
            torch._foreach_add_(t_params, o_params, alpha=1.0 - tau)
            for tb, ob in zip(target.buffers(), online.buffers()):
                tb.copy_(ob)

    @torch.no_grad()
    def q_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.encoder(obs))

    def parameter_counts(self) -> dict[str, int]:
        return {name: sum(p.numel() for p in getattr(self, name).parameters()) for name in self.ONLINE}


def trained_modules(cfg: Config) -> list[str]:
    """Online modules that receive gradients under the configured losses."""
    mods = ["encoder", "q_head"]
    if cfg.loss.needs_rollout:
        mods.append("dynamics")
    if cfg.loss.reward:
        mods.append("reward_head")
    if cfg.loss.continuation:
        mods.append("continuation_head")
    return mods


class Learner:
    """Owns the optimizer. One ``update`` = one optimizer step followed by one EMA target update."""

    def __init__(self, model: WorldModel, cfg: Config):
        from .losses import compute_losses

        self._compute_losses = compute_losses
        self.model = model
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(
            model.online_parameters(), lr=cfg.optim.lr, eps=cfg.optim.adam_eps
        )
        self.updates = 0

    def update(self, batch: Any) -> dict[str, float]:
        loss, metrics = self._compute_losses(self.model, batch, self.cfg.loss)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {self.updates}: {metrics}")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.online_parameters(), self.cfg.optim.grad_clip_norm
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at update {self.updates}")
        self.optimizer.step()
        self.model.update_targets(self.cfg.optim.tau)
        self.updates += 1
        metrics["grad_norm"] = float(grad_norm)
        return metrics
