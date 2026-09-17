"""Masked training objective.

For a batch with ``K`` transitions per sequence (``k = 0..K-1``):

* ``m_k = valid_k``                       reward, continuation and Q mask (terminal step included)
* ``l_k = valid_k * (1 - terminated_k)``  next-latent (JEPA) mask

The optional inverse-dynamics loss uses ``m_k``: the terminal transition's real final observation is a
valid "after" frame for the action that produced it.

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


REWARD_RANGE_ERROR = (
    "reward head received a reward outside {-1, 0, +1}; the 3-class head requires "
    "sign-clipped rewards (env.reward_transform='sign')"
)


def reward_out_of_range(rewards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """0-d bool tensor: whether any valid reward is outside {-1, 0, +1} (no host sync)."""
    ok = (rewards == -1) | (rewards == 0) | (rewards == 1)
    return (~ok & mask.bool()).any()


def mse_latent_distance(z_hat: list[torch.Tensor], target_z: torch.Tensor, target_z0: torch.Tensor | None,
                        mode: str = "absolute") -> torch.Tensor:
    """Per-step mean squared error between predicted and target latents, ``[B, K]``.

    The alternative to the cosine distance: it constrains the scale of the latent as well as its
    direction, which is why it needs an explicit anti-collapse term (a constant latent is an MSE
    optimum). ``mode`` matches :func:`jepa_distances`.
    """
    K = target_z.shape[1]
    if mode == "delta":
        if target_z0 is None:
            raise ValueError("delta target needs the target encoding of the root observation")
        prev = [target_z0] + [target_z[:, k] for k in range(K - 1)]
        return torch.stack([((z_hat[k + 1] - z_hat[k]) - (target_z[:, k] - prev[k])).flatten(1).pow(2).mean(1)
                            for k in range(K)], dim=1)
    if mode == "batch_centered":
        mu = target_z.flatten(0, 1).mean(0, keepdim=True).detach()
        return torch.stack([((z_hat[k + 1] - mu) - (target_z[:, k] - mu)).flatten(1).pow(2).mean(1)
                            for k in range(K)], dim=1)
    if mode == "absolute":
        return torch.stack([(z_hat[k + 1] - target_z[:, k]).flatten(1).pow(2).mean(1) for k in range(K)], dim=1)
    raise ValueError(f"unknown target mode {mode!r}")


def epps_pulley(y: torch.Tensor) -> torch.Tensor:
    """Epps-Pulley test statistic for standard normality, per column of ``[N, R]``.

    Compares the empirical characteristic function with the standard normal one under a Gaussian
    weight, in closed form (Epps & Pulley, 1983). Small for a standard normal sample, large otherwise
    (a constant sample is maximally non-Gaussian).
    """
    n = y.shape[0]
    d = y.unsqueeze(0) - y.unsqueeze(1)  # [N, N, R]
    pair = torch.exp(-0.5 * d.pow(2)).sum(dim=(0, 1))  # includes j == k
    single = torch.exp(-0.25 * y.pow(2)).sum(0)
    return 1.0 + n / 3.0**0.5 + (pair - n) / n - 2.0**0.5 * single


def sigreg_penalty(z_flat: torch.Tensor, n_directions: int = 64, generator: torch.Generator | None = None,
                   eps: float = 1e-6) -> torch.Tensor:
    """Sketched isotropic-Gaussian regularization: push the embedding toward an isotropic Gaussian.

    Random unit directions are drawn, the batch is projected onto each, each projection is standardized,
    and the Epps-Pulley normality statistic is averaged over directions. A collapsed embedding (all
    samples equal, or all variation in a few directions) is strongly non-Gaussian along most random
    directions, so this penalizes collapse without prescribing a per-dimension variance floor.

    Follows the SIGReg idea from LeJEPA (Balestriero & LeCun, 2025); the implementation here is a plain
    Epps-Pulley sketch, not their full method.
    """
    B, D = z_flat.shape
    dirs = torch.randn(D, n_directions, device=z_flat.device, dtype=z_flat.dtype, generator=generator)
    dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp(min=eps)
    centered = z_flat - z_flat.mean(0, keepdim=True)
    # One global scale, never per-direction: an isotropic Gaussian has the same variance along every
    # direction, so dividing each projection by its own std would make the test blind to a low-rank
    # (collapsed) cloud, whose projections are individually still Gaussian.
    scale = centered.pow(2).mean().sqrt().clamp(min=eps)
    return epps_pulley((centered @ dirs) / scale).mean()


def vicreg_penalty(z_flat: torch.Tensor, floor: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """VICReg variance hinge plus covariance term (the classic anti-collapse pair)."""
    return variance_penalty(z_flat, floor, eps) + covariance_penalty(z_flat)


def anti_collapse(name: str, z_flat: torch.Tensor, floor: float, eps: float,
                  n_directions: int = 64, generator: torch.Generator | None = None) -> torch.Tensor:
    if name == "sigreg":
        return sigreg_penalty(z_flat, n_directions, generator)
    if name == "vicreg":
        return vicreg_penalty(z_flat, floor, eps)
    if name == "variance":
        return variance_penalty(z_flat, floor, eps)
    if name == "none":
        return torch.zeros((), device=z_flat.device)
    raise ValueError(f"unknown anti-collapse term {name!r}")


def reward_to_class(rewards: torch.Tensor, mask: torch.Tensor, check: bool = True) -> torch.Tensor:
    """Map clipped rewards {-1, 0, +1} to classes {0, 1, 2}. Refuses any other value on valid steps.

    ``check=False`` skips the (host-synchronizing) check; the caller must then check
    ``reward_out_of_range`` itself.
    """
    if check and bool(reward_out_of_range(rewards, mask)):
        raise ValueError(REWARD_RANGE_ERROR)
    # clamp so an unchecked bad value can never become an illegal class index (a CUDA device assert)
    return torch.where(mask.bool(), rewards + 1, torch.ones_like(rewards)).long().clamp(0, 2)


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


def jepa_distances(z_hat: list[torch.Tensor], target_z: torch.Tensor, target_z0: torch.Tensor | None,
                   mode: str, eps: float) -> torch.Tensor:
    """Per-step temporal-prediction distances ``[B, K]`` under the configured target.

    * ``absolute``: D(z_hat[k+1], target_z[k+1]). The static background dominates the cosine.
    * ``batch_centered``: the same after subtracting the batch-mean target latent from both sides, so a
      shared constant component cannot carry the similarity.
    * ``delta``: D(z_hat[k+1] - z_hat[k], target_z[k+1] - target_z[k]), i.e. predict what *changes*.
      The predicted change is taken along the model's own rollout; the target change uses consecutive
      target encodings, with ``target_z0`` supplying the step before the first transition.
    """
    K = target_z.shape[1]
    if mode == "absolute":
        return torch.stack([cosine_distance(z_hat[k + 1], target_z[:, k], eps) for k in range(K)], dim=1)
    if mode == "batch_centered":
        mu = target_z.flatten(0, 1).mean(0, keepdim=True).detach()
        return torch.stack([cosine_distance(z_hat[k + 1] - mu, target_z[:, k] - mu, eps) for k in range(K)], dim=1)
    if mode == "delta":
        if target_z0 is None:
            raise ValueError("delta target needs the target encoding of the root observation")
        prev_target = [target_z0] + [target_z[:, k] for k in range(K - 1)]
        return torch.stack(
            [cosine_distance(z_hat[k + 1] - z_hat[k], target_z[:, k] - prev_target[k], eps) for k in range(K)], dim=1
        )
    raise ValueError(f"unknown jepa_target {mode!r}")


def n_step_targets(rewards: torch.Tensor, terminated: torch.Tensor, valid: torch.Tensor,
                   next_value: torch.Tensor, gamma: float, n: int, depths: int) -> torch.Tensor:
    """Truncated n-step Double DQN targets for each depth ``k < depths``; ``[B, depths]``.

    From root ``k`` the target accumulates up to ``min(n, K - k)`` real rewards and then bootstraps from
    the value of the last *real* observation reached. A termination ends the sum with no bootstrap; a
    padded (invalid) step means the window ended earlier, so the bootstrap stays at the last real step.
    ``n = 1`` reproduces the single-step target exactly.
    """
    B, K = rewards.shape
    zero = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    out = []
    for k in range(depths):
        acc, disc, boot = zero.clone(), torch.ones_like(zero), zero.clone()
        alive = valid[:, k].bool()
        for j in range(k, min(k + n, K)):
            step = valid[:, j].bool() & alive
            acc = acc + torch.where(step, disc * rewards[:, j], zero)
            boot = torch.where(step, disc * gamma * (~terminated[:, j].bool()).to(rewards.dtype) * next_value[:, j], boot)
            alive = step & ~terminated[:, j].bool()
            disc = disc * gamma
        out.append(acc + boot)
    return torch.stack(out, dim=1)


def depth_weights(K: int, mode: str, gamma: float, device, dtype) -> torch.Tensor:
    """Per-depth weights for the latent loss, normalized to mean 1 so the loss keeps its scale.

    Uniform weighting treats a K-step rollout as K equally important tasks. They are not independent:
    past K ~= 20 they begin to pull the shared encoder in opposing directions, and the feature they all
    agree on is one that barely changes -- predictable at every depth, and useless for control.
    """
    k = torch.arange(K, device=device, dtype=dtype)
    if mode == "uniform":
        w = torch.ones_like(k)
    elif mode == "inverse":
        w = 1.0 / (k + 1.0)
    elif mode == "discount":
        w = gamma**k
    else:
        raise ValueError(f"unknown depth weighting {mode!r}")
    return w / w.mean()


def dynamics_step(dynamics, z: torch.Tensor, action: torch.Tensor, state):
    """``(z', state')`` from a dynamics core. Anything without a ``step`` method is memoryless."""
    step = getattr(dynamics, "step", None)
    if step is None:
        return dynamics(z, action), None
    return step(z, action, state)


def unroll(dynamics: torch.nn.Module, z0: torch.Tensor, actions: torch.Tensor, steps: int) -> list[torch.Tensor]:
    """z_hat[0] = z0, z_hat[k+1] = g(z_hat[k], a_k). Predictions are never detached or replaced.

    The dynamics state is threaded through the rollout, so a recurrent core (LMUDynamics) sees the
    history of the rollout rather than restarting at every step. A memoryless core returns None for it.
    """
    z_hat, state = [z0], None
    for k in range(steps):
        z_next, state = dynamics_step(dynamics, z_hat[k], actions[:, k], state)
        z_hat.append(z_next)
    return z_hat


def compute_losses(model, batch, cfg: LossConfig, as_tensors: bool = False) -> tuple[torch.Tensor, dict]:
    """Total loss and per-component metrics.

    Metrics are floats by default. With ``as_tensors=True`` they are detached 0-d tensors and nothing
    synchronizes with the host, so the caller can fetch them all in one transfer (see ``Learner``). The
    caller must then raise ``REWARD_RANGE_ERROR`` if ``metrics["_reward_out_of_range"]`` is set.
    """
    obs, actions = batch.observations, batch.actions
    B, K = actions.shape
    valid = batch.valid.bool()
    terminated = batch.terminated.bool()
    rewards = batch.rewards.float()
    latent_mask = valid & ~terminated  # l_k

    # Q supervised on z_hat[0 .. q_depths-1]; q_imagined_depth caps that independently of K
    q_depths = min(cfg.q_imagined_depth or K, K) if cfg.q_imagined else 1
    n_targets = max(q_depths, K if cfg.jepa else 0)  # target encodings needed for obs 1..n_targets
    if cfg.jepa or cfg.inverse == "predicted":
        steps = K
    else:
        steps = K - 1 if (cfg.reward or cfg.continuation or cfg.q_imagined) else 0
    inverse_real = cfg.inverse == "real"

    def encode_seq(encoder, frames: torch.Tensor) -> torch.Tensor:
        n = frames.shape[1]
        return encoder(frames.flatten(0, 1)).unflatten(0, (B, n))

    # Online encodings of real next observations. With inverse="real" they need gradients (for all K
    # steps); otherwise they are only used, detached, for Double DQN action selection.
    n_online = K if inverse_real else q_depths
    with torch.set_grad_enabled(inverse_real and torch.is_grad_enabled()):
        online_next_z = encode_seq(model.encoder, obs[:, 1 : n_online + 1])  # [:, k] encodes obs k+1

    # ---- targets from real observations, no gradients
    with torch.no_grad():
        next_obs = obs[:, 1 : n_targets + 1]
        target_z = encode_seq(model.target_encoder, next_obs)  # target_z[:, k] encodes obs k+1
        target_z0 = model.target_encoder(obs[:, 0]) if (cfg.jepa and cfg.jepa_target == "delta") else None
        online_next_q = model.q_head(online_next_z[:, :q_depths].detach().flatten(0, 1)).unflatten(0, (B, q_depths))
        target_next_q = model.target_q_head(target_z[:, :q_depths].flatten(0, 1)).unflatten(0, (B, q_depths))
        next_value = double_dqn_value(online_next_q, target_next_q)
        if cfg.n_step == 1:
            y = td_targets(rewards[:, :q_depths], terminated[:, :q_depths], next_value, cfg.gamma)
        else:
            # n-step needs values at later observations too, so recompute over the full window
            all_online_q = model.q_head(model.encoder(obs[:, 1:].flatten(0, 1))).unflatten(0, (B, K))
            all_target_q = model.target_q_head(
                model.target_encoder(obs[:, 1:].flatten(0, 1))).unflatten(0, (B, K))
            values = double_dqn_value(all_online_q, all_target_q)
            y = n_step_targets(rewards, terminated, valid, values, cfg.gamma, cfg.n_step, q_depths)

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
        dist = jepa_distances(z_hat, target_z[:, :K], target_z0, cfg.jepa_target, cfg.cosine_eps)
        w_depth = depth_weights(K, cfg.jepa_depth_weight, cfg.jepa_depth_gamma, dist.device, dist.dtype)
        l_jepa = masked_mean(dist * w_depth, latent_mask)
        total = total + cfg.lambda_jepa * l_jepa
        metrics["loss_jepa"] = l_jepa
        with torch.no_grad():
            for k in range(K):
                metrics[f"jepa_d{k + 1}"] = masked_mean(dist[:, k], latent_mask[:, k])
                # persistence baseline under the same target convention (a delta of zero for "delta")
                persist_hat = [z0] * (K + 1)
                persist = jepa_distances(persist_hat, target_z[:, :K], target_z0, cfg.jepa_target,
                                         cfg.cosine_eps)[:, k]
                metrics[f"persist_d{k + 1}"] = masked_mean(persist, latent_mask[:, k])

    if cfg.reward:
        logits = torch.stack([model.reward_head(z_hat[k], actions[:, k]) for k in range(K)], dim=1)
        classes = reward_to_class(rewards, valid, check=not as_tensors)
        if as_tensors:
            metrics["_reward_out_of_range"] = reward_out_of_range(rewards, valid)
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

    if cfg.successor:
        # psi(z_t, a_t) = phi(z_{t+1}) + gamma * (not terminated) * psi_bar(z_{t+1}, a'), with a' the
        # online-greedy action (Double-DQN style selection, EMA evaluation). The horizon comes from the
        # bootstrap, so nothing is unrolled here: only step 0 of the window is used.
        z_root = z0.detach() if cfg.sf_detach else z0
        with torch.no_grad():
            phi_next = model.phi(target_z[:, 0])
            greedy_next = online_next_q[:, 0].argmax(dim=-1)
            psi_next = model.target_successor_head(target_z[:, 0], greedy_next)
            sf_target = (1.0 - cfg.gamma) * phi_next + cfg.gamma * (~terminated[:, 0]).float().unsqueeze(1) * psi_next
        psi = model.successor_head(z_root, actions[:, 0])
        sf_err = F.smooth_l1_loss(psi, sf_target, reduction="none").mean(dim=1)
        l_sf = masked_mean(sf_err, valid[:, 0])
        # w is fit against the reward actually collected, so Q_sf = w . psi is on the reward scale.
        pred_r = model.reward_weights(phi_next).squeeze(-1)
        l_sf_reward = masked_mean((pred_r - rewards[:, 0]) ** 2, valid[:, 0])
        total = total + cfg.lambda_sf * l_sf + cfg.lambda_sf_reward * l_sf_reward
        metrics["loss_sf"] = l_sf
        metrics["loss_sf_reward"] = l_sf_reward
        with torch.no_grad():
            metrics["sf_psi_norm"] = psi.norm(dim=1).mean()
            metrics["sf_target_norm"] = sf_target.norm(dim=1).mean()
            metrics["sf_q_mean"] = masked_mean(
                model.reward_weights(psi).squeeze(-1) / (1.0 - cfg.gamma), valid[:, 0])

    if cfg.inverse != "none":
        if inverse_real:
            z_real = torch.cat([z0.unsqueeze(1), online_next_z], dim=1)  # f(x_0) .. f(x_K)
            pairs = [(z_real[:, k], z_real[:, k + 1]) for k in range(K)]
        else:
            pairs = [(z_hat[k], z_hat[k + 1]) for k in range(K)]
        inv_logits = torch.stack([model.inverse_head(a, b) for a, b in pairs], dim=1)  # [B, K, A]
        ce = F.cross_entropy(inv_logits.flatten(0, 1), actions.flatten(), reduction="none").view(B, K)
        l_inv = masked_mean(ce, valid)
        total = total + cfg.lambda_inverse * l_inv
        metrics["loss_inverse"] = l_inv
        with torch.no_grad():
            metrics["inverse_acc"] = masked_mean((inv_logits.argmax(-1) == actions).float(), valid)

    if cfg.hierarchical:
        H = min(cfg.macro_horizon, K)
        macro_actions = actions[:, :H]
        # a macro step is valid only if all H one-step transitions inside it are
        macro_valid = valid[:, :H].all(dim=1)
        ended = terminated[:, :H].any(dim=1)
        z_in = z0.detach() if cfg.macro_detach else z0  # level 2 must not corrupt level 1
        z_macro = model.macro_dynamics(z_in, macro_actions)
        # jumpy latent target: the real observation H steps ahead (no target when the episode ended)
        macro_latent_mask = macro_valid & ~ended
        if cfg.jepa:
            m_dist = jepa_distances([z_in, z_macro], target_z[:, H - 1 : H], target_z0, cfg.jepa_target, cfg.cosine_eps)
        else:
            with torch.no_grad():
                t_macro = model.target_encoder(obs[:, H])
            m_dist = cosine_distance(z_macro, t_macro, cfg.cosine_eps).unsqueeze(1)
        l_macro_jepa = masked_mean(m_dist[:, 0], macro_latent_mask)
        # macro return: discounted sum of the clipped rewards actually collected inside the window
        with torch.no_grad():
            disc = torch.tensor([cfg.gamma**i for i in range(H)], device=obs.device)
            alive = torch.cumprod(torch.cat([torch.ones_like(terminated[:, :1]), ~terminated[:, :H - 1]], 1).float(), 1)
            macro_ret = (rewards[:, :H] * disc * alive * valid[:, :H]).sum(1)
        pred_ret = model.macro_return(z_in, macro_actions)
        l_macro_ret = masked_mean((pred_ret - macro_ret) ** 2, macro_valid)
        cont_logit = model.macro_continuation(z_in, macro_actions)
        l_macro_cont = masked_mean(
            F.binary_cross_entropy_with_logits(cont_logit, (~ended).float(), reduction="none"), macro_valid)
        l_macro = l_macro_jepa + l_macro_ret + l_macro_cont
        total = total + cfg.lambda_macro * l_macro
        metrics["loss_macro"] = l_macro
        metrics["macro_jepa"] = l_macro_jepa
        metrics["macro_return_mse"] = l_macro_ret
        metrics["macro_cont_bce"] = l_macro_cont

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

    if as_tensors:
        return total, {k: v.detach() if torch.is_tensor(v) else torch.tensor(v) for k, v in metrics.items()}
    values = torch.stack([torch.as_tensor(v, dtype=torch.float32, device=obs.device).detach().float()
                          for v in metrics.values()]).tolist()  # one host transfer
    return total, dict(zip(metrics.keys(), values))
