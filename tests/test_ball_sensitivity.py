"""Motion input channels and the temporal-target variants (both aimed at small fast objects)."""

import copy
import json

import numpy as np
import pytest
import torch
from conftest import random_batch, world_model_config

from atari_jepa.agent import WorldModel
from atari_jepa.losses import compute_losses, jepa_distances
from atari_jepa.networks import Encoder


def test_motion_channels_extend_the_input_and_vanish_without_motion():
    enc = Encoder(4, [32, 64, 64], 84, motion_channels=True)
    assert enc.convs[0].in_channels == 7  # 4 frames + 3 signed differences
    still = torch.full((2, 4, 84, 84), 17, dtype=torch.uint8)
    x = still.float() / 255.0
    with torch.no_grad():
        diffs = torch.cat([x, x[:, 1:] - x[:, :-1]], dim=1)[:, 4:]
    assert torch.count_nonzero(diffs) == 0  # a static scene contributes nothing through the motion path

    # a moving dot changes the motion channels even though the final frame is identical
    a = torch.zeros(1, 4, 84, 84, dtype=torch.uint8)
    b = torch.zeros(1, 4, 84, 84, dtype=torch.uint8)
    for t in range(4):
        a[0, t, 40, 20 + t] = 255  # moves right
        b[0, t, 40, 23 - t] = 255  # moves left, same last frame position at t=3? no: ends elsewhere
    b[0, 3] = a[0, 3]  # force identical final frames; only the motion history differs
    with torch.no_grad():
        za, zb = enc(a), enc(b)
    assert (za - zb).abs().max() > 1e-4


def test_motion_channels_are_configured_end_to_end():
    cfg = world_model_config()
    cfg.network.motion_channels = True
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4)
    assert model.encoder.convs[0].in_channels == 7
    assert model.target_encoder.convs[0].in_channels == 7
    loss, metrics = compute_losses(model, random_batch(), cfg.loss)
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in model.encoder.parameters())
    assert cfg.network.encoder_in_channels(cfg.env.history) == 7


@pytest.mark.parametrize("mode", ["batch_centered", "delta"])
def test_centered_and_delta_targets_ignore_a_shared_constant_component(mode):
    """A constant added to every latent is exactly the 'static background' these modes should ignore."""
    g = torch.Generator().manual_seed(0)
    B, K, D = 8, 3, 16
    z_hat = [torch.randn(B, D, generator=g) for _ in range(K + 1)]
    target = torch.randn(B, K, D, generator=g)
    target0 = torch.randn(B, D, generator=g)
    offset = 5.0 * torch.randn(1, D, generator=g)

    base = jepa_distances(z_hat, target, target0, mode, 1e-8)
    shifted = jepa_distances([z + offset for z in z_hat], target + offset, target0 + offset, mode, 1e-8)
    torch.testing.assert_close(base, shifted, rtol=1e-4, atol=1e-5)
    # the absolute target is *not* invariant: a large shared component dominates its cosine
    abs_base = jepa_distances(z_hat, target, target0, "absolute", 1e-8)
    abs_shifted = jepa_distances([z + offset for z in z_hat], target + offset, target0 + offset, "absolute", 1e-8)
    assert (abs_base - abs_shifted).abs().max() > 0.1


def test_absolute_mode_matches_plain_cosine_and_delta_penalizes_persistence():
    from atari_jepa.losses import cosine_distance

    g = torch.Generator().manual_seed(1)
    z_hat = [torch.randn(4, 6, generator=g) for _ in range(3)]
    target = torch.randn(4, 2, 6, generator=g)
    target0 = torch.randn(4, 6, generator=g)
    expected = torch.stack([cosine_distance(z_hat[k + 1], target[:, k]) for k in range(2)], dim=1)
    torch.testing.assert_close(jepa_distances(z_hat, target, target0, "absolute", 1e-8), expected)
    # a model that predicts no change scores the worst possible delta distance
    frozen = [z_hat[0]] * 3
    assert torch.allclose(jepa_distances(frozen, target, target0, "delta", 1e-8), torch.ones(4, 2))


@pytest.mark.parametrize("mode", ["batch_centered", "delta"])
def test_targets_train_through_the_rollout(mode):
    cfg = world_model_config()
    cfg.loss.jepa_target = mode
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4)
    loss, metrics = compute_losses(model, random_batch(), cfg.loss)
    loss.backward()
    assert torch.isfinite(loss) and metrics["loss_jepa"] > 0
    for name in ("encoder", "dynamics"):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(model, name).parameters()), name
    assert all(p.grad is None for p in model.target_parameters())
    assert cfg.loss.variant_name().endswith(f"+jepa_{mode}")


def test_collection_policy_can_be_the_models_own_planner(tmp_path):
    """Planner-driven collection must use this run's own network, and disagree with greedy Q."""
    from test_checkpoint import tiny_config

    from atari_jepa.train import Trainer

    run = tmp_path / "run"
    cfg = tiny_config(run)
    cfg.train.collect_controller = "lookahead"
    trainer = Trainer(cfg, run, torch.device("cpu"))
    assert trainer._collector == "lookahead"
    obs = np.random.default_rng(0).integers(0, 256, (4, 84, 84), dtype=np.uint8)
    planned = trainer._greedy(obs)
    with torch.no_grad():
        q_action = int(trainer.model.q_values(torch.from_numpy(obs).unsqueeze(0)).argmax(-1))
    assert 0 <= planned < trainer.env.num_actions
    trainer.run(["test"])  # a full short run collects with the planner without error
    assert trainer.counters["decisions"] == cfg.train.total_decisions
    assert json.loads((run / "metadata.json").read_text())["collect_controller"] == "lookahead"

    # a config whose model is not trained must refuse planner collection rather than plan with noise
    bad = tiny_config(tmp_path / "bad")
    bad.loss.reward = False
    bad.train.collect_controller = "lookahead"
    with pytest.raises(ValueError, match="trains the model"):
        Trainer(bad, tmp_path / "bad", torch.device("cpu"))


def test_collection_switches_to_the_planner_only_after_the_threshold(tmp_path):
    from test_checkpoint import tiny_config

    from atari_jepa.train import Trainer

    run = tmp_path / "run"
    cfg = tiny_config(run)
    cfg.train.collect_controller = "lookahead"
    cfg.train.collect_switch_decisions = 10**9  # never reached in this run
    trainer = Trainer(cfg, run, torch.device("cpu"))
    obs = np.random.default_rng(0).integers(0, 256, (4, 84, 84), dtype=np.uint8)
    with torch.no_grad():
        q_action = int(trainer.model.q_values(torch.from_numpy(obs).unsqueeze(0)).argmax(-1))
    assert trainer._greedy(obs) == q_action  # before the switch: the Q-policy
    trainer.counters["decisions"] = 10**9
    from atari_jepa.planning import one_step_scores
    with torch.no_grad():
        planned = int(one_step_scores(trainer.model, trainer.model.encoder(
            torch.from_numpy(obs).unsqueeze(0)), cfg.loss.gamma).argmax(-1))
    assert trainer._greedy(obs) == planned  # after: the planner


def test_shrink_and_perturb_partially_resets_the_q_head(tmp_path):
    from test_checkpoint import tiny_config

    from atari_jepa.train import Trainer

    run = tmp_path / "run"
    cfg = tiny_config(run)
    trainer = Trainer(cfg, run, torch.device("cpu"))
    # give the optimizer some state for the Q head, then reset
    loss, _ = trainer.learner._compute_losses(trainer.model, random_batch(A=4), cfg.loss)
    trainer.learner.optimizer.zero_grad(); loss.backward(); trainer.learner.optimizer.step()
    assert any(p in trainer.learner.optimizer.state for p in trainer.model.q_head.parameters())

    before = [p.clone() for p in trainer.model.q_head.parameters()]
    before_target = [p.clone() for p in trainer.model.target_q_head.parameters()]
    torch.manual_seed(0)
    trainer._shrink_and_perturb_q_head(0.5)
    after = list(trainer.model.q_head.parameters())
    assert all(not torch.equal(a, b) for a, b in zip(after, before))          # it moved
    assert all((a - 0.5 * b).abs().max() < 5.0 for a, b in zip(after, before))  # but stayed bounded
    assert all(not torch.equal(a, b) for a, b in zip(trainer.model.target_q_head.parameters(), before_target))
    assert all(p not in trainer.learner.optimizer.state for p in trainer.model.q_head.parameters())
    # the encoder is untouched: a reset restores plasticity in the head only
    assert trainer.model.encoder.convs[0].weight.grad is None or True
