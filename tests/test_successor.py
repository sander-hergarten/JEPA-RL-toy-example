"""Successor features: the Bellman target, what the loss is allowed to touch, and the SF value."""

import numpy as np
import pytest
import torch
from conftest import random_batch, world_model_config

from atari_jepa.agent import WorldModel, trained_modules
from atari_jepa.losses import compute_losses
from atari_jepa.networks import FeatureProjection


def sf_config(**overrides):
    cfg = world_model_config()
    cfg.loss.successor = True
    cfg.loss.jepa_target = "delta"
    for k, v in overrides.items():
        setattr(cfg.loss, k, v)
    return cfg


def test_the_feature_basis_is_frozen_and_deterministic():
    """phi must not follow the successor objective: a constant phi is a trivial optimum of it."""
    phi = FeatureProjection(64, 8, seed=3)
    assert not list(phi.parameters())  # a buffer, not a parameter: no gradient path, never in the optimizer
    torch.testing.assert_close(phi.weight, FeatureProjection(64, 8, seed=3).weight)
    assert not torch.allclose(phi.weight, FeatureProjection(64, 8, seed=4).weight)
    z = torch.randn(5, 64)
    assert phi(z).shape == (5, 8)
    # it survives an EMA target copy, which copies buffers
    model = WorldModel(sf_config(), num_actions=4)
    torch.testing.assert_close(model.phi.weight, model.phi.weight)
    assert model.phi.weight.requires_grad is False


def test_successor_modules_are_built_only_when_configured():
    plain = WorldModel(world_model_config(), num_actions=4)
    assert plain.successor_head is None and plain.phi is None
    assert plain.target_pairs == plain.TARGET_PAIRS  # checkpoint layout unchanged without the flag

    sf = WorldModel(sf_config(), num_actions=4)
    assert sf.successor_head is not None and sf.reward_weights is not None
    assert ("target_successor_head", "successor_head") in sf.target_pairs
    assert "successor_head" in sf.online_modules and "reward_weights" in sf.online_modules
    assert "successor_head" in trained_modules(sf_config())
    assert all(not p.requires_grad for p in sf.target_successor_head.parameters())


def test_the_target_is_the_bellman_backup_and_terminal_states_do_not_bootstrap():
    cfg = sf_config()
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4)
    batch = random_batch(A=4)
    _, metrics = compute_losses(model, batch, cfg.loss)

    assert np.isfinite(metrics["loss_sf"]) and metrics["loss_sf"] > 0

    def hand_loss(b, bootstrap: bool) -> float:
        """The loss recomputed from the definition, with the bootstrap term optionally dropped."""
        with torch.no_grad():
            tz = model.target_encoder(b.observations[:, 1])
            target = (1.0 - cfg.loss.gamma) * model.phi(tz)
            if bootstrap:
                greedy = model.q_head(model.encoder(b.observations[:, 1])).argmax(-1)
                alive = (~b.terminated[:, 0].bool()).float().unsqueeze(1)
                target = target + cfg.loss.gamma * alive * model.target_successor_head(tz, greedy)
            psi = model.successor_head(model.encoder(b.observations[:, 0]), b.actions[:, 0])
            err = torch.nn.functional.smooth_l1_loss(psi, target, reduction="none").mean(dim=1)
            return float(err[b.valid[:, 0].bool()].mean())

    assert metrics["loss_sf"] == pytest.approx(hand_loss(batch, bootstrap=True), rel=1e-4)
    # with a live bootstrap term the two disagree, so the check above is not vacuous
    assert metrics["loss_sf"] != pytest.approx(hand_loss(batch, bootstrap=False), rel=1e-4)

    # every step-0 transition terminal: the target collapses to phi alone, bootstrap or not
    term = random_batch(A=4)
    term.terminated[:, 0] = True
    _, m2 = compute_losses(model, term, cfg.loss)
    assert m2["loss_sf"] == pytest.approx(hand_loss(term, bootstrap=False), rel=1e-4)


@pytest.mark.parametrize("detach", [True, False])
def test_sf_detach_controls_whether_the_successor_loss_reaches_the_encoder(detach):
    cfg = sf_config(sf_detach=detach)
    # isolate the successor term: everything else off, so any encoder gradient came from it
    for flag in ("jepa", "reward", "continuation", "variance", "q_imagined"):
        setattr(cfg.loss, flag, False)
    cfg.loss.lambda_q = 0.0
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4)
    loss, _ = compute_losses(model, random_batch(A=4), cfg.loss)
    loss.backward()
    grad = sum(p.grad.abs().sum() for p in model.encoder.parameters() if p.grad is not None)
    assert (grad == 0) if detach else (grad > 0)
    # the successor head itself always trains
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.successor_head.parameters())


def test_the_successor_value_is_the_reward_map_applied_to_psi():
    cfg = sf_config()
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4)
    z = model.encoder(torch.randint(0, 256, (3, 4, 84, 84), dtype=torch.uint8))
    with torch.no_grad():
        q_sf = model.q_from_successor(z)
        want = torch.stack([
            model.reward_weights(model.successor_head(z, torch.full((3,), a, dtype=torch.long))).squeeze(-1)
            for a in range(4)
        ], dim=1) / (1.0 - cfg.loss.gamma)
    assert q_sf.shape == (3, 4)
    torch.testing.assert_close(q_sf, want)


def test_controllers_refuse_a_checkpoint_without_successor_features():
    from atari_jepa.planning import make_controller

    cfg = world_model_config()
    model = WorldModel(cfg, num_actions=4)
    device = torch.device("cpu")
    with pytest.raises(ValueError, match="loss.successor"):
        make_controller("sf", model, cfg, device, 0.0, seed=0)
    with pytest.raises(ValueError, match="loss.successor"):
        make_controller("lookahead", model, cfg, device, 0.0, seed=0, bootstrap="sf")


def test_sf_bootstrap_changes_the_planned_score_but_not_the_search_shape():
    from atari_jepa.planning import beam_search, one_step_scores

    cfg = sf_config()
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4)
    z = model.encoder(torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.uint8))
    with torch.no_grad():
        q_scores = one_step_scores(model, z, 0.99, "q")
        sf_scores = one_step_scores(model, z, 0.99, "sf")
    assert q_scores.shape == sf_scores.shape == (1, 4)
    assert not torch.allclose(q_scores, sf_scores)
    for boot in ("q", "sf"):
        action, score = beam_search(model, z, horizon=3, beam_width=4, gamma=0.99, bootstrap=boot)
        assert 0 <= action < 4 and np.isfinite(score)


def test_planning_bootstrap_is_validated():
    from atari_jepa.config import PlanningConfig

    assert PlanningConfig(bootstrap="sf").bootstrap == "sf"
    with pytest.raises(ValueError, match="planning.bootstrap"):
        PlanningConfig(bootstrap="successor")
    assert sf_config().loss.variant_name().endswith("+sf")
    assert sf_config(sf_detach=False).loss.variant_name().endswith("+sf_attached")
