"""Masked training objective.

For a batch with ``K`` transitions per sequence (``k = 0..K-1``):

* ``m_k = valid_k``                       reward, continuation and Q mask (terminal step included)
* ``l_k = valid_k * (1 - terminated_k)``  next-latent (JEPA) mask

Every component is normalized by its own count of valid entries (``masked_mean``), and padded entries
are removed with ``torch.where`` so their values cannot leak into losses or gradients.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import LossConfig
from .networks import REWARD_VALUES


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``x`` over ``mask``; exactly 0 (finite, zero gradient) when the mask is empty."""
    mask = mask.bool()
    total = torch.where(mask, x, torch.zeros_like(x)).sum()
    return total / mask.sum().clamp(min=1)


def cosine_distance(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """D(u, v) = 1 - <u/|u|, v/|v|> over flattened latents; ``[N, ...] -> [N]``."""
    u = F.normalize(u.flatten(1), dim=1, eps=eps)
    v = F.normalize(v.flatten(1), dim=1, eps=eps)
    return 1.0 - (u * v).sum(dim=1)


def latent_std(z_flat: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(z_flat.var(dim=0, unbiased=False) + eps)


def variance_penalty(z_flat: torch.Tensor, floor: float, eps: float) -> torch.Tensor:
    """mean_j relu(floor - std_j) over a batch of flattened (un-normalized) latents ``[B, D]``."""
    return F.relu(floor - latent_std(z_flat, eps)).mean()


def covariance_penalty(z_flat: torch.Tensor) -> torch.Tensor:
    """VICReg-style sum of squared off-diagonal covariances / D, via the B x B Gram matrix.

    Uses ||Zc^T Zc||_F = ||Zc Zc^T||_F so the D x D covariance is never materialized.
    """
    B, D = z_flat.shape
    zc = z_flat - z_flat.mean(dim=0, keepdim=True)
    denom = max(B - 1, 1)
    gram = zc @ zc.T
    total = (gram**2).sum() / denom**2
    diag = ((zc**2).sum(dim=0) / denom) ** 2
    return (total - diag.sum()) / D


def reward_to_class(rewards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Map clipped rewards {-1, 0, +1} to classes {0, 1, 2}. Refuses any other value on valid steps."""
    ok = (rewards == -1) | (rewards == 0) | (rewards == 1)
    if not bool((ok | ~mask.bool()).all()):
        raise ValueError(
            "reward head received a reward outside {-1, 0, +1}; the 3-class head requires "
            "sign-clipped rewards (env.reward_transform='sign')"
        )
    return torch.where(mask.bool(), rewards + 1, torch.ones_like(rewards)).long()


def expected_reward(logits: torch.Tensor) -> torch.Tensor:
    """E[r] = sum_c p(c) * value(c) = p(+1) - p(-1)."""
    values = torch.tensor(REWARD_VALUES, dtype=logits.dtype, device=logits.device)
    return (logits.softmax(dim=-1) * values).sum(dim=-1)


def double_dqn_value(online_next_q: torch.Tensor, target_next_q: torch.Tensor) -> torch.Tensor:
    """Target-network value of the online network's greedy action (first index on exact ties)."""
    next_action = online_next_q.argmax(dim=-1, keepdim=True)
    return target_next_q.gather(-1, next_action).squeeze(-1)


def td_targets(
    rewards: torch.Tensor, terminated: torch.Tensor, next_value: torch.Tensor, gamma: float
) -> torch.Tensor:
    """y = r + gamma * (1 - terminated) * V(real next observation). Truncation still bootstraps."""
    return rewards + gamma * (1.0 - terminated.float()) * next_value


def unroll(dynamics: torch.nn.Module, z0: torch.Tensor, actions: torch.Tensor, steps: int) -> list[torch.Tensor]:
    """z_hat[0] = z0, z_hat[k+1] = g(z_hat[k], a_k). Predictions are never detached or replaced."""
    z_hat = [z0]
    for k in range(steps):
        z_hat.append(dynamics(z_hat[k], actions[:, k]))
    return z_hat


def compute_losses(model, batch, cfg: LossConfig) -> tuple[torch.Tensor, dict[str, float]]:
    obs, actions = batch.observations, batch.actions
    B, K = actions.shape
    valid = batch.valid.bool()
    terminated = batch.terminated.bool()
    rewards = batch.rewards.float()
    latent_mask = valid & ~terminated  # l_k

    q_depths = K if cfg.q_imagined else 1  # Q supervised on z_hat[0 .. q_depths-1]
    n_targets = max(q_depths, K if cfg.jepa else 0)  # target encodings needed for obs 1..n_targets
    steps = K if cfg.jepa else (K - 1 if (cfg.reward or cfg.continuation or cfg.q_imagined) else 0)

    def encode_seq(encoder, frames: torch.Tensor) -> torch.Tensor:
        n = frames.shape[1]
        return encoder(frames.flatten(0, 1)).unflatten(0, (B, n))

    # ---- targets from real observations, no gradients
    with torch.no_grad():
        next_obs = obs[:, 1 : n_targets + 1]
        target_z = encode_seq(model.target_encoder, next_obs)  # target_z[:, k] encodes obs k+1
        q_next_obs = next_obs[:, :q_depths]
        online_next_q = model.q_head(model.encoder(q_next_obs.flatten(0, 1))).unflatten(0, (B, q_depths))
        target_next_q = model.target_q_head(target_z[:, :q_depths].flatten(0, 1)).unflatten(0, (B, q_depths))
        next_value = double_dqn_value(online_next_q, target_next_q)
        y = td_targets(rewards[:, :q_depths], terminated[:, :q_depths], next_value, cfg.gamma)

    # ---- online root encoding and recursive rollout
    z0 = model.encoder(obs[:, 0])
    z_hat = unroll(model.dynamics, z0, actions, steps)

    metrics: dict[str, torch.Tensor | float] = {}
    total = torch.zeros((), device=obs.device)

    # Q-learning (root, plus imagined depths for the world model)
    q_pred = torch.stack(
        [model.q_head(z_hat[k]).gather(1, actions[:, k : k + 1]).squeeze(1) for k in range(q_depths)], dim=1
    )
    q_err = F.smooth_l1_loss(q_pred, y, reduction="none", beta=cfg.huber_delta)
    q_mask = valid[:, :q_depths]
    l_q = masked_mean(q_err, q_mask)
    total = total + cfg.lambda_q * l_q
    metrics["loss_q"] = l_q
    for k in range(q_depths):
        metrics[f"q_td_d{k}"] = masked_mean(q_err[:, k], q_mask[:, k])
    metrics["q_pred_mean"] = masked_mean(q_pred[:, 0], q_mask[:, 0])
    metrics["q_target_mean"] = masked_mean(y[:, 0], q_mask[:, 0])

    if cfg.jepa:
        dist = torch.stack(
            [cosine_distance(z_hat[k + 1], target_z[:, k], cfg.cosine_eps) for k in range(K)], dim=1
        )
        l_jepa = masked_mean(dist, latent_mask)
        total = total + cfg.lambda_jepa * l_jepa
        metrics["loss_jepa"] = l_jepa
        with torch.no_grad():
            for k in range(K):
                metrics[f"jepa_d{k + 1}"] = masked_mean(dist[:, k], latent_mask[:, k])
                persist = cosine_distance(z0, target_z[:, k], cfg.cosine_eps)
                metrics[f"persist_d{k + 1}"] = masked_mean(persist, latent_mask[:, k])

    if cfg.reward:
        logits = torch.stack([model.reward_head(z_hat[k], actions[:, k]) for k in range(K)], dim=1)
        classes = reward_to_class(rewards, valid)
        ce = F.cross_entropy(logits.flatten(0, 1), classes.flatten(), reduction="none").view(B, K)
        l_reward = masked_mean(ce, valid)
        total = total + cfg.lambda_reward * l_reward
        metrics["loss_reward"] = l_reward

    if cfg.continuation:
        logits = torch.stack([model.continuation_head(z_hat[k], actions[:, k]) for k in range(K)], dim=1)
        target = (~terminated).float()
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        l_cont = masked_mean(bce, valid)
        total = total + cfg.lambda_continue * l_cont
        metrics["loss_continue"] = l_cont

    z_root = z0.flatten(1)  # before any L2 normalization
    std = latent_std(z_root, cfg.variance_eps)
    if cfg.variance:
        l_var = F.relu(cfg.variance_floor - std).mean()
        total = total + cfg.lambda_var * l_var
        metrics["loss_var"] = l_var
    if cfg.covariance:
        l_cov = covariance_penalty(z_root)
        total = total + cfg.lambda_cov * l_cov
        metrics["loss_cov"] = l_cov

    with torch.no_grad():
        metrics["loss_total"] = total
        metrics["latent_std_mean"] = std.mean()
        metrics["latent_frac_below_floor"] = (std < cfg.variance_floor).float().mean()
        zn = F.normalize(z_root, dim=1)
        sim = zn @ zn.T
        metrics["latent_pairwise_cos"] = (sim.sum() - sim.diagonal().sum()) / max(B * (B - 1), 1)
        metrics["valid_transitions"] = valid.sum()
        metrics["terminal_transitions"] = (valid & terminated).sum()
        metrics["latent_targets"] = latent_mask.sum()

    out = {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in metrics.items()}
    return total, out
