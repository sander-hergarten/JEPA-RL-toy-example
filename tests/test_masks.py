"""Mask normalization: padding cannot change losses or gradients; empty masks are finite zeros."""

import copy

import torch
from conftest import random_batch

from atari_jepa.losses import compute_losses, masked_mean


def grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def run(model, batch, cfg):
    model.zero_grad(set_to_none=True)
    loss, metrics = compute_losses(model, batch, cfg.loss)
    loss.backward()
    return loss.detach(), metrics, grads(model)


def test_padded_values_do_not_change_loss_or_gradients(model, cfg):
    batch = random_batch()
    loss_a, m_a, g_a = run(model, batch, cfg)

    other = copy.deepcopy(batch)
    pad = ~other.valid
    obs_pad = torch.zeros_like(other.valid[:, :1]).expand(-1, 1)
    obs_valid = torch.cat([torch.ones_like(obs_pad), other.valid], dim=1)
    noise = torch.randint(0, 256, other.observations.shape, dtype=torch.uint8)
    other.observations = torch.where(obs_valid[..., None, None, None], other.observations, noise)
    other.actions = torch.where(pad, torch.full_like(other.actions, 3), other.actions)
    other.rewards = torch.where(pad, torch.full_like(other.rewards, 7.5), other.rewards)  # not even a valid class
    other.terminated = other.terminated | pad
    other.truncated = other.truncated | pad
    loss_b, m_b, g_b = run(model, other, cfg)

    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=1e-6)
    for key in m_a:
        if key != "loss_total":
            assert abs(m_a[key] - m_b[key]) < 1e-5, key
    assert g_a.keys() == g_b.keys()
    for name in g_a:
        torch.testing.assert_close(g_a[name], g_b[name], rtol=1e-5, atol=1e-7, msg=name)


def test_all_masked_components_are_finite_and_zero(model, cfg):
    batch = random_batch()
    batch.valid[:] = False
    batch.terminated[:] = False
    loss, metrics, g = run(model, batch, cfg)
    for key in ("loss_q", "loss_jepa", "loss_reward", "loss_continue"):
        assert metrics[key] == 0.0, key
    assert torch.isfinite(loss)
    assert all(torch.isfinite(v).all() for v in g.values())
    # only the (root-based) variance term remains in the total
    assert abs(float(loss) - metrics["loss_var"]) < 1e-6


def test_masked_mean_normalizes_by_own_count():
    x = torch.tensor([[1.0, 2.0, float("nan")], [3.0, float("inf"), 5.0]])
    mask = torch.tensor([[True, True, False], [True, False, True]])
    assert masked_mean(x, mask).item() == (1 + 2 + 3 + 5) / 4
    empty = masked_mean(x, torch.zeros_like(mask))
    assert empty.item() == 0.0
    y = torch.ones(3, requires_grad=True)
    masked_mean(y, torch.zeros(3, dtype=torch.bool)).backward()
    assert torch.equal(y.grad, torch.zeros(3))
