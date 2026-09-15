"""H-JEPA level 2: jumpy dynamics over an action sequence, its heads, and the hierarchical planner."""

import numpy as np
import pytest
import torch
from conftest import random_batch, world_model_config

from atari_jepa.agent import WorldModel, trained_modules
from atari_jepa.losses import compute_losses
from atari_jepa.networks import MacroDynamics
from atari_jepa.planning import candidate_sequences, hierarchical_scores, make_controller


def hierarchical_cfg(horizon=3):
    cfg = world_model_config()
    cfg.loss.hierarchical = True
    cfg.loss.macro_horizon = horizon
    return cfg


def test_macro_dynamics_depends_on_the_whole_sequence_not_just_the_first_action():
    m = MacroDynamics((64, 7, 7), 4, 3, world_model_config().network)
    z = torch.randn(1, 64, 7, 7)
    a = torch.tensor([[1, 2, 3]])
    same_first = torch.tensor([[1, 0, 0]])
    with torch.no_grad():
        assert not torch.allclose(m(z, a), m(z, same_first))  # later actions matter
        assert torch.allclose(m(z, a), m(z, a.clone()))       # deterministic


def test_level_two_trains_and_leaves_the_targets_alone():
    cfg = hierarchical_cfg()
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4)
    loss, metrics = compute_losses(model, random_batch(K=3), cfg.loss)
    loss.backward()
    for key in ("loss_macro", "macro_jepa", "macro_return_mse", "macro_cont_bce"):
        assert np.isfinite(metrics[key]), key
    for name in ("macro_dynamics", "macro_return", "macro_continuation", "encoder"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters()), name
    assert all(p.grad is None for p in model.target_parameters())
    assert "macro_dynamics" in trained_modules(cfg)


def test_a_flat_config_builds_no_level_two():
    cfg = world_model_config()
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4)
    assert model.macro_dynamics is None and model.macro_horizon == 0
    _, metrics = compute_losses(model, random_batch(), cfg.loss)
    assert "loss_macro" not in metrics


def test_candidate_sequences_are_exhaustive_when_they_fit_and_sampled_otherwise():
    exhaustive = candidate_sequences(4, 3, 128, None, "cpu")
    assert exhaustive.shape == (64, 3) and len({tuple(r.tolist()) for r in exhaustive}) == 64
    sampled = candidate_sequences(4, 5, 128, torch.Generator().manual_seed(0), "cpu")
    assert sampled.shape == (128, 5)
    constants = {tuple([a] * 5) for a in range(4)}
    assert constants <= {tuple(r.tolist()) for r in sampled}  # the "repeat one action" options are always there


def test_hierarchical_scores_and_controller_pick_the_best_sequence():
    cfg = hierarchical_cfg()
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4).eval()
    obs = np.random.default_rng(0).integers(0, 256, (4, 84, 84), dtype=np.uint8)
    with torch.no_grad():  # score the same latent the controller will encode, not an unrelated one
        z = model.encoder(torch.from_numpy(obs).unsqueeze(0))
    seqs = candidate_sequences(4, 3, 128, None, "cpu")
    scores = hierarchical_scores(model, z, cfg.loss.gamma, seqs)
    assert scores.shape == (len(seqs),) and torch.isfinite(scores).all()
    ctrl = make_controller("hierarchical", model, cfg, torch.device("cpu"), 0.0, seed=0)
    action, info = ctrl.act(obs)
    assert action == int(seqs[int(scores.argmax()), 0])  # executes the first action of the best sequence
    assert 0 <= info["q_action"] < 4 and ctrl.stats()["decisions"] == 1


def test_hierarchical_controller_refuses_a_flat_checkpoint():
    cfg = world_model_config()
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4)
    with pytest.raises(ValueError, match="loss.hierarchical"):
        make_controller("hierarchical", model, cfg, torch.device("cpu"), 0.0, seed=0)
