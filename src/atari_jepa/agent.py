"""Online and target networks, EMA targets, and the optimizer step."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import Config
from .networks import (ContinuationHead, Dynamics, Encoder, FeatureProjection, InverseDynamicsHead,
                       LMUDynamics, MacroDynamics, MacroHead, QHead, RewardHead, SuccessorHead)


class WorldModel(nn.Module):
    """Online encoder/dynamics/heads plus EMA copies of the encoder and Q-head.

    There is no target dynamics network: targets for latent prediction come from the target encoder
    applied to real observations.
    """

    def __init__(self, cfg: Config, num_actions: int):
        super().__init__()
        env, net = cfg.env, cfg.network
        self.num_actions = num_actions
        self.encoder = Encoder(env.history, net.encoder_channels, env.screen_size, net.conv_dtype,
                               net.motion_channels)
        self.latent_shape = self.encoder.latent_shape
        latent_dim = int(np.prod(self.latent_shape))
        core = LMUDynamics if net.dynamics_kind == "lmu" else Dynamics
        self.dynamics = core(self.latent_shape, num_actions, net)
        self.q_head = QHead(latent_dim, num_actions, net.q_hidden)
        self.reward_head = RewardHead(latent_dim, num_actions, net.reward_hidden)
        self.continuation_head = ContinuationHead(latent_dim, num_actions, net.continuation_hidden)
        # Only built when used, so checkpoints from configs without it keep their exact layout.
        self.inverse_head = (
            InverseDynamicsHead(latent_dim, num_actions, net.inverse_hidden) if cfg.loss.inverse != "none" else None
        )
        h = cfg.loss.macro_horizon
        self.macro_horizon = h if cfg.loss.hierarchical else 0
        self.macro_dynamics = MacroDynamics(self.latent_shape, num_actions, h, net) if cfg.loss.hierarchical else None
        self.macro_return = MacroHead(latent_dim, num_actions, h, net.macro_hidden) if cfg.loss.hierarchical else None
        self.macro_continuation = MacroHead(latent_dim, num_actions, h, net.macro_hidden) if cfg.loss.hierarchical else None
        # Successor features. phi is a frozen random basis (see FeatureProjection); reward_weights is
        # the linear map w with r ~ w . phi(z'), so Q_sf(z, a) = w . psi(z, a) needs no rollout.
        self.phi = FeatureProjection(latent_dim, net.sf_dim, net.sf_seed) if cfg.loss.successor else None
        self.successor_head = (
            SuccessorHead(latent_dim, num_actions, net.successor_hidden, net.sf_dim) if cfg.loss.successor else None
        )
        self.reward_weights = nn.Linear(net.sf_dim, 1, bias=False) if cfg.loss.successor else None
        # psi carries the (1 - gamma) factor from the discounted-occupancy definition, so it stays on
        # phi's scale instead of growing like 1/(1 - gamma) (~100x here). Undone in q_from_successor,
        # which must return a value on the return scale for the planner to add it to a reward.
        self.sf_gamma = cfg.loss.gamma
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_q_head = copy.deepcopy(self.q_head)
        if self.successor_head is not None:
            self.target_successor_head = copy.deepcopy(self.successor_head)
        for p in self.target_parameters():
            p.requires_grad_(False)

    TARGET_PAIRS = (("target_encoder", "encoder"), ("target_q_head", "q_head"))

    @property
    def target_pairs(self) -> tuple[tuple[str, str], ...]:
        """TARGET_PAIRS plus the successor head when it exists (checkpoints without it are unchanged)."""
        if self.successor_head is None:
            return self.TARGET_PAIRS
        return self.TARGET_PAIRS + (("target_successor_head", "successor_head"),)

    def q_from_successor(self, z: torch.Tensor) -> torch.Tensor:
        """Q(z, a) = w . psi(z, a) / (1 - gamma), on the return scale; ``[B, ...] -> [B, A]``."""
        B = z.shape[0]
        zr = z.repeat_interleave(self.num_actions, dim=0)
        acts = torch.arange(self.num_actions, device=z.device).repeat(B)
        q = self.reward_weights(self.successor_head(zr, acts)).view(B, self.num_actions)
        return q / (1.0 - self.sf_gamma)

    @property
    def online_modules(self) -> tuple[str, ...]:
        names = ("encoder", "dynamics", "q_head", "reward_head", "continuation_head")
        if self.inverse_head is not None:
            names += ("inverse_head",)
        if self.macro_dynamics is not None:
            names += ("macro_dynamics", "macro_return", "macro_continuation")
        if self.successor_head is not None:
            names += ("successor_head", "reward_weights")
        return names

    def online_parameters(self) -> list[nn.Parameter]:
        return [p for name in self.online_modules for p in getattr(self, name).parameters()]

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Online parameters that still require gradients (the encoder may be frozen)."""
        return [p for p in self.online_parameters() if p.requires_grad]

    def freeze_encoder(self) -> None:
        """Keep the encoder fixed: RL then only learns on top of the pretrained representation."""
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    def target_parameters(self) -> list[nn.Parameter]:
        return [p for t, _ in self.target_pairs for p in getattr(self, t).parameters()]

    @torch.no_grad()
    def update_targets(self, tau: float) -> None:
        """target = tau * target + (1 - tau) * online, for parameters and buffers."""
        for t_name, o_name in self.target_pairs:
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
        return {name: sum(p.numel() for p in getattr(self, name).parameters()) for name in self.online_modules}


def check_precision_supported(cfg: Config, device: torch.device) -> None:
    """bfloat16 conv trunks need bf16 conv backward; some CPU backends (oneDNN on AVX2) lack it."""
    if cfg.network.conv_dtype != "bfloat16":
        return
    conv = nn.Conv2d(2, 2, 3).to(device)
    x = torch.randn(1, 2, 5, 5, device=device)
    try:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            y = conv(x)
        y.float().sum().backward()
    except RuntimeError as exc:
        raise RuntimeError(
            f"network.conv_dtype=bfloat16 cannot train on {device}: {exc}. Use a CUDA device or float32."
        ) from exc


def trained_modules(cfg: Config) -> list[str]:
    """Online modules that receive gradients under the configured losses."""
    mods = [] if cfg.train.freeze_encoder else ["encoder"]
    mods.append("q_head")
    if cfg.loss.needs_rollout:
        mods.append("dynamics")
    if cfg.loss.reward:
        mods.append("reward_head")
    if cfg.loss.continuation:
        mods.append("continuation_head")
    if cfg.loss.inverse != "none":
        mods.append("inverse_head")
    if cfg.loss.hierarchical:
        mods += ["macro_dynamics", "macro_return", "macro_continuation"]
    if cfg.loss.successor:
        mods += ["successor_head", "reward_weights"]
    return mods


class Learner:
    """Owns the optimizer. One ``update`` = one optimizer step followed by one EMA target update."""

    def __init__(self, model: WorldModel, cfg: Config):
        from .losses import compute_losses

        self._compute_losses = compute_losses
        self.model = model
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(
            model.trainable_parameters(), lr=cfg.optim.lr, eps=cfg.optim.adam_eps
        )
        self.updates = 0

    def update(self, batch: Any) -> dict[str, float]:
        """One optimizer step. All metrics come back in a single device-to-host transfer at the end.

        Non-finite losses/gradients and out-of-range rewards are detected after the step (checking
        earlier would force extra synchronizations); the exception aborts the run, and the last saved
        checkpoint predates the bad update.
        """
        from .losses import REWARD_RANGE_ERROR

        loss, metrics = self._compute_losses(self.model, batch, self.cfg.loss, as_tensors=True)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.trainable_parameters(), self.cfg.optim.grad_clip_norm
        )
        self.optimizer.step()
        self.model.update_targets(self.cfg.optim.tau)
        self.updates += 1
        metrics["grad_norm"] = grad_norm.detach()
        values = torch.stack([v.float() for v in metrics.values()]).tolist()
        out = dict(zip(metrics.keys(), values))
        if out.pop("_reward_out_of_range", 0.0):
            raise ValueError(REWARD_RANGE_ERROR)
        if not (np.isfinite(out["loss_total"]) and np.isfinite(out["grad_norm"])):
            raise FloatingPointError(f"non-finite loss or gradient at update {self.updates}: {out}")
        return out
