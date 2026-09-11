"""Termination vs truncation semantics, end to end through replay and the loss."""

import numpy as np
import torch
from conftest import frame, ids
from test_replay import fill

from atari_jepa.losses import compute_losses, double_dqn_value
from atari_jepa.replay import SequenceReplay


def q_targets(model, batch, gamma):
    """Recompute the Double DQN target for step 0 by hand."""
    nxt = batch.observations[:, 1]
    with torch.no_grad():
        v = double_dqn_value(model.q_head(model.encoder(nxt)), model.target_q_head(model.target_encoder(nxt)))
    return batch.rewards[:, 0] + gamma * (1 - batch.terminated[:, 0].float()) * v


def test_terminal_transition_trains_reward_and_disables_bootstrap(model, cfg):
    buf = SequenceReplay(100, (84, 84), history=4)
    fill(buf, [3, 5], end="terminated")
    # episode 0: raw reward +1 at t=1; make the terminal transition itself rewarding
    buf.raw_rewards[2] = -1.0
    batch = buf.gather(np.array([2]), horizon=3).to_torch(torch.device("cpu"))
    assert batch.valid[0].tolist() == [True, False, False]
    assert batch.terminated[0].tolist() == [True, False, False]
    assert batch.rewards[0, 0] == -1.0
    _, m = compute_losses(model, batch, cfg.loss)
    assert m["q_target_mean"] == -1.0  # y = r, no bootstrap
    assert m["loss_reward"] > 0 and m["loss_continue"] > 0  # the terminal step is trained
    assert m["latent_targets"] == 0  # no latent consistency on the terminal observation
    assert m["valid_transitions"] == 1


def test_truncation_bootstraps_from_final_observation_not_reset(model, cfg):
    buf = SequenceReplay(100, (84, 84), history=4)
    fill(buf, [3, 4], end="truncated")
    batch_np = buf.gather(np.array([2]), horizon=3)
    # next observation is the truncated episode's final frame, not episode 1's reset frame
    assert ids(batch_np.observations[0, 1]) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert batch_np.truncated[0].tolist() == [True, False, False]
    assert batch_np.valid[0].tolist() == [True, False, False]
    batch = batch_np.to_torch(torch.device("cpu"))
    y = q_targets(model, batch, cfg.loss.gamma)
    _, m = compute_losses(model, batch, cfg.loss)
    assert np.isclose(m["q_target_mean"], float(y[0]), atol=1e-6)
    assert float(y[0]) != float(batch.rewards[0, 0])  # it bootstrapped
    # continuation target for a truncated step is 1, and its next latent is supervised
    assert m["latent_targets"] == 1


def test_padded_steps_after_termination_are_ignored(model, cfg):
    buf = SequenceReplay(100, (84, 84), history=4)
    fill(buf, [2, 6], end="terminated")
    batch = buf.gather(np.array([0]), horizon=3).to_torch(torch.device("cpu"))
    assert batch.valid[0].tolist() == [True, True, False]
    _, m = compute_losses(model, batch, cfg.loss)
    assert m["valid_transitions"] == 2
    assert m["latent_targets"] == 1  # step 0 only; step 1 terminates
    # the depth-2 Q TD metric is over zero valid entries and reports exactly 0
    assert m["q_td_d2"] == 0.0


def test_frame_fixture_distinguishes_reset():
    # guard for the tests above: final and reset frames differ in their identifying pixels
    assert ids(frame(0, 3)[None])[0] != ids(frame(1, 0)[None])[0]
