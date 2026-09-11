"""Gradient flow through the recursive rollout, and the gradient audit table."""

import torch
from conftest import random_batch, world_model_config

from atari_jepa.agent import WorldModel
from atari_jepa.losses import compute_losses, cosine_distance, unroll, variance_penalty


def nonzero_grad(module) -> bool:
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def no_grad(module) -> bool:
    return all(p.grad is None or p.grad.abs().sum() == 0 for p in module.parameters())


def test_depth_k_loss_reaches_encoder_and_every_dynamics_application(model):
    batch = random_batch()
    z0 = model.encoder(batch.observations[:, 0])
    z_hat = unroll(model.dynamics, z0, batch.actions, 3)
    for z in z_hat:
        z.retain_grad()
    target = torch.randn_like(z_hat[3])
    cosine_distance(z_hat[3], target).mean().backward()
    assert nonzero_grad(model.encoder)
    assert nonzero_grad(model.dynamics)
    for k in range(3):
        assert z_hat[k].grad is not None and z_hat[k].grad.abs().sum() > 0, f"no gradient at depth {k}"


def test_intermediate_prediction_affects_later_outputs(model):
    batch = random_batch()
    with torch.no_grad():
        z0 = model.encoder(batch.observations[:, 0])
        z1 = model.dynamics(z0, batch.actions[:, 0])
        z3 = model.dynamics(model.dynamics(z1, batch.actions[:, 1]), batch.actions[:, 2])
        z1p = z1 + 0.5 * torch.randn_like(z1)
        z3p = model.dynamics(model.dynamics(z1p, batch.actions[:, 1]), batch.actions[:, 2])
    assert (z3 - z3p).abs().max() > 1e-3


def test_full_loss_gradient_audit(model, cfg):
    loss, _ = compute_losses(model, random_batch(), cfg.loss)
    loss.backward()
    for name in ("encoder", "dynamics", "q_head", "reward_head", "continuation_head"):
        assert nonzero_grad(getattr(model, name)), name
    for p in model.target_parameters():
        assert p.grad is None and not p.requires_grad


def test_variance_penalty_does_not_reach_dynamics(model):
    batch = random_batch()
    z0 = model.encoder(batch.observations[:, 0])
    variance_penalty(z0.flatten(1), floor=10.0, eps=1e-4).backward()
    assert nonzero_grad(model.encoder)
    assert no_grad(model.dynamics)


def test_q_baseline_trains_only_encoder_and_q_head():
    cfg = world_model_config(q_imagined=False, jepa=False, reward=False, continuation=False, variance=False)
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4)
    loss, _ = compute_losses(model, random_batch(), cfg.loss)
    loss.backward()
    assert nonzero_grad(model.encoder) and nonzero_grad(model.q_head)
    assert no_grad(model.dynamics) and no_grad(model.reward_head) and no_grad(model.continuation_head)


def test_imagined_q_reaches_dynamics_and_bootstrap_has_no_graph():
    cfg = world_model_config(q_imagined=True, jepa=False, reward=False, continuation=False, variance=False)
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4)
    loss, _ = compute_losses(model, random_batch(), cfg.loss)
    loss.backward()
    assert nonzero_grad(model.dynamics) and nonzero_grad(model.encoder)
    assert all(p.grad is None for p in model.target_parameters())
