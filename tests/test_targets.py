"""EMA convention, reward classes, and Double DQN targets."""

import copy

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
