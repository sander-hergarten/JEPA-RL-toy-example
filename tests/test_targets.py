"""EMA convention, reward classes, and Double DQN targets."""

import copy

import numpy as np

import pytest
import torch

from atari_jepa.losses import double_dqn_value, expected_reward, reward_to_class, td_targets


def test_ema_initialization_and_update(model):
    for t_name, o_name in model.TARGET_PAIRS:
        for pt, po in zip(getattr(model, t_name).parameters(), getattr(model, o_name).parameters()):
            assert torch.equal(pt, po)
    old_targets = [p.clone() for p in model.target_parameters()]
    with torch.no_grad():
        for p in list(model.encoder.parameters()) + list(model.q_head.parameters()):
            p.add_(torch.randn_like(p))
    online = [p.clone() for p in list(model.encoder.parameters()) + list(model.q_head.parameters())]
    tau = 0.9
    model.update_targets(tau)
    for new, old, on in zip(model.target_parameters(), old_targets, online):
        torch.testing.assert_close(new, tau * old + (1 - tau) * on)
    # the online network is untouched and the other online modules are not averaged in
    for p, on in zip(list(model.encoder.parameters()) + list(model.q_head.parameters()), online):
        assert torch.equal(p, on)


def test_reward_classes_and_expected_reward():
    r = torch.tensor([-1.0, 0.0, 1.0, 0.0])
    assert reward_to_class(r, torch.ones(4, dtype=torch.bool)).tolist() == [0, 1, 2, 1]
    logits = torch.full((3, 3), -30.0)
    logits[0, 0] = logits[1, 1] = logits[2, 2] = 30.0
    torch.testing.assert_close(expected_reward(logits), torch.tensor([-1.0, 0.0, 1.0]))
    mixed = torch.log(torch.tensor([[0.2, 0.5, 0.3]]))
    torch.testing.assert_close(expected_reward(mixed), torch.tensor([0.1]))
    with pytest.raises(ValueError):
        reward_to_class(torch.tensor([2.0]), torch.ones(1, dtype=torch.bool))
    # an out-of-range value on a masked step is ignored
    assert reward_to_class(torch.tensor([2.0]), torch.zeros(1, dtype=torch.bool)).tolist() == [1]


def test_double_dqn_uses_online_selection_and_target_evaluation():
    online = torch.tensor([[1.0, 5.0, 2.0], [3.0, 0.0, 3.0]])
    target = torch.tensor([[9.0, 4.0, 7.0], [1.0, 8.0, 2.0]])
    v = double_dqn_value(online, target)
    # row 0: online argmax = 1 -> target 4 (not target max 9); row 1: tie -> first index 0 -> 1
    assert v.tolist() == [4.0, 1.0]
    rewards = torch.tensor([1.0, 0.0])
    y = td_targets(rewards, torch.tensor([True, False]), v, gamma=0.5)
    assert y.tolist() == [1.0, 0.5]  # terminal: no bootstrap; truncation/continuing: bootstrap


def test_td_targets_have_no_graph_when_built_under_no_grad(model):
    obs = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    with torch.no_grad():
        v = double_dqn_value(model.q_head(model.encoder(obs)), model.target_q_head(model.target_encoder(obs)))
        y = td_targets(torch.zeros(2), torch.zeros(2, dtype=torch.bool), v, 0.99)
    assert not y.requires_grad


def test_target_q_consumes_target_encoder(model):
    obs = torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8)
    m2 = copy.deepcopy(model)
    with torch.no_grad():
        for p in m2.encoder.parameters():
            p.add_(1.0)
    # changing only the online encoder must not change target-network values
    with torch.no_grad():
        a = model.target_q_head(model.target_encoder(obs))
        b = m2.target_q_head(m2.target_encoder(obs))
    torch.testing.assert_close(a, b)


def test_n_step_targets_match_hand_computation_and_reduce_to_one_step():
    from atari_jepa.losses import n_step_targets, td_targets

    g = 0.5
    rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    terminated = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]).bool()
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]).bool()  # row 2: window ends early
    values = torch.tensor([[10.0, 20.0, 30.0, 40.0]] * 3)

    one = n_step_targets(rewards, terminated, valid, values, g, 1, 2)
    torch.testing.assert_close(one, td_targets(rewards[:, :2], terminated[:, :2], values[:, :2], g))

    three = n_step_targets(rewards, terminated, valid, values, g, 3, 1)[:, 0]
    # row 0: r + g r + g^2 r + g^3 V(x_3) = 1 + .5 + .25 + .125*30
    # row 1: terminates at step 1 -> 1 + .5*(-1), no bootstrap
    # row 2: step 2 is padding -> bootstrap at the last real step: 1 + .5*1 + .25*V(x_2)
    torch.testing.assert_close(three, torch.tensor([1 + 0.5 + 0.25 + 0.125 * 30, 1 - 0.5, 1 + 0.5 + 0.25 * 20]))

    # n larger than the window is clipped to the available steps, never reading past it
    assert torch.equal(n_step_targets(rewards, terminated, valid, values, g, 99, 1),
                       n_step_targets(rewards, terminated, valid, values, g, 4, 1))


def test_n_step_flows_through_compute_losses():
    from conftest import random_batch, world_model_config

    from atari_jepa.agent import WorldModel
    from atari_jepa.losses import compute_losses

    cfg = world_model_config()
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4)
    batch = random_batch()
    _, m1 = compute_losses(model, batch, cfg.loss)
    cfg.loss.n_step = 3
    _, m3 = compute_losses(model, batch, cfg.loss)
    assert m1["q_target_mean"] != m3["q_target_mean"]  # the target actually changed
    assert np.isfinite(m3["loss_q"]) and m3["loss_q"] > 0


@pytest.mark.parametrize("cap,expected", [(0, 6), (2, 2), (99, 6)])
def test_q_imagined_depth_caps_the_value_supervision_independently_of_k(cap, expected):
    """The Q-loss covers min(cap or K, K) imagined depths, while the rollout stays K steps long."""
    from conftest import random_batch, world_model_config

    from atari_jepa.agent import WorldModel
    from atari_jepa.losses import compute_losses

    K = 6
    cfg = world_model_config()
    cfg.replay.rollout_steps = K
    cfg.loss.q_imagined_depth = cap
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4)
    loss, metrics = compute_losses(model, random_batch(K=K), cfg.loss)
    loss.backward()

    depths = sorted(int(k[len("q_td_d"):]) for k in metrics if k.startswith("q_td_d"))
    assert depths == list(range(expected))
    # the rollout itself is untouched: the JEPA loss still supervises all K steps
    assert sorted(int(k[len("jepa_d"):]) for k in metrics if k.startswith("jepa_d")) == list(range(1, K + 1))
    assert torch.isfinite(loss) and any(p.grad is not None for p in model.q_head.parameters())


def test_q_imagined_depth_is_rejected_when_negative_and_named_in_the_variant():
    from atari_jepa.config import LossConfig

    with pytest.raises(ValueError, match="q_imagined_depth"):
        LossConfig(q_imagined_depth=-1)
    named = LossConfig(q_imagined=True, jepa=True, reward=True, continuation=True,
                       jepa_target="delta", q_imagined_depth=5)
    assert named.variant_name().endswith("+qdepth5")
    assert "qdepth" not in LossConfig(q_imagined=True, jepa=True, reward=True,
                                      continuation=True).variant_name()
