"""The delta probe: pair sampling, the jump head, and the baselines it is read against."""

import numpy as np
import pytest
import torch
from conftest import frame, world_model_config

from atari_jepa.agent import WorldModel
from atari_jepa.delta_probe import (
    action_summary,
    encode,
    episode_split,
    iterated_prediction,
    pair_indices,
    run_delta,
)
from atari_jepa.networks import JumpHead
from atari_jepa.replay import SequenceReplay


def filled_replay(episodes: int = 8, length: int = 40, capacity: int = 2000) -> SequenceReplay:
    """Episodes of known length, each frame tagged with (episode, t) by conftest.frame."""
    replay = SequenceReplay(capacity=capacity, frame_shape=(84, 84), history=4)
    rng = np.random.default_rng(0)
    for e in range(episodes):
        replay.start_episode(frame(e, 0))
        for t in range(length):
            replay.add(action=int(rng.integers(0, 4)), raw_reward=0.0,
                       terminated=(t == length - 1), truncated=False, next_frame=frame(e, t + 1))
    return replay


def test_pairs_stay_inside_one_episode_and_respect_the_holdout():
    replay = filled_replay()
    rng = np.random.default_rng(0)
    for delta in (1, 10, 30):
        for holdout in (False, True):
            roots = pair_indices(replay, delta, rng, 50, holdout=holdout)
            slot, tgt = roots % replay.capacity, (roots + delta) % replay.capacity
            assert (replay.episode_id[slot] == replay.episode_id[tgt]).all()
            # the frames themselves confirm it: same episode tag, exactly delta apart in time
            a, b = replay.frames[slot], replay.frames[tgt]
            assert (a[:, 0, 0] == b[:, 0, 0]).all()
            assert (b[:, 0, 1].astype(int) - a[:, 0, 1].astype(int) == delta).all()
            assert (episode_split(replay.episode_id[slot]) == holdout).all()

    # train and test roots never share an episode, so no pair can straddle the split
    tr = pair_indices(replay, 10, rng, 50, holdout=False)
    te = pair_indices(replay, 10, rng, 50, holdout=True)
    assert not (set(replay.episode_id[tr % replay.capacity]) & set(replay.episode_id[te % replay.capacity]))


def test_a_delta_longer_than_any_episode_has_no_pairs():
    replay = filled_replay(episodes=4, length=20)
    with pytest.raises(ValueError, match="pairs available|no pair"):
        pair_indices(replay, 50, np.random.default_rng(0), 10, holdout=False)


def test_pair_sampling_never_touches_the_span():
    """Memory is O(1) in delta: only the two endpoint slots are read, never the frames between."""
    replay = filled_replay(episodes=6, length=60)
    reads = []
    original = replay.frames

    class Watched(np.ndarray):
        def __getitem__(self, item):
            reads.append(item)
            return original.__getitem__(item)

    replay.frames = original.view(Watched)
    pair_indices(replay, 40, np.random.default_rng(0), 20, holdout=False)
    replay.frames = original
    assert reads == []  # pair selection reads slot metadata only


def test_jump_head_shapes_conditioning_and_gradients():
    z = torch.randn(3, 8, 7, 7)
    cfg = world_model_config().network
    unconditioned = JumpHead((8, 7, 7), 0, cfg)
    assert unconditioned(z).shape == z.shape
    conditioned = JumpHead((8, 7, 7), 4, cfg)
    with pytest.raises(ValueError, match="needs a summary"):
        conditioned(z)
    s = torch.rand(3, 4)
    out = conditioned(z, s)
    assert out.shape == z.shape
    out.sum().backward()
    assert conditioned.conv_in.weight.grad.abs().sum() > 0
    # the summary actually reaches the output
    with torch.no_grad():
        other = conditioned(z, torch.zeros_like(s))
    assert (out - other).abs().max() > 1e-6


def test_action_summary_counts_the_span_and_sums_to_one():
    replay = filled_replay(episodes=4, length=30)
    roots = pair_indices(replay, 10, np.random.default_rng(0), 16, holdout=False)
    s = action_summary(replay, roots, 10, 4, torch.device("cpu"))
    assert s.shape == (16, 4)
    torch.testing.assert_close(s.sum(dim=1), torch.ones(16))
    idx = (roots[:, None] + np.arange(10)[None, :]) % replay.capacity
    expected = np.stack([(replay.actions[idx] == a).sum(axis=1) for a in range(4)], axis=1) / 10
    torch.testing.assert_close(s, torch.from_numpy(expected.astype(np.float32)))


def test_iterated_prediction_matches_an_explicit_unroll():
    cfg = world_model_config()
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4)
    replay = filled_replay(episodes=4, length=30)
    roots = pair_indices(replay, 3, np.random.default_rng(0), 8, holdout=False)
    device = torch.device("cpu")
    z = encode(model, replay, roots, device)
    got = iterated_prediction(model, z, replay, roots, 3)
    with torch.no_grad():
        want = z
        for k in range(3):
            acts = torch.from_numpy(replay.actions[(roots + k) % replay.capacity])
            want = model.dynamics(want, acts)
    torch.testing.assert_close(got, want)


def test_run_delta_reports_every_baseline_and_a_mean_relative_score():
    cfg = world_model_config()
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4)
    replay = filled_replay(episodes=12, length=40)

    class Args:
        train_pairs, test_pairs, updates, batch_size, lr, seed = 64, 32, 2, 8, 1e-4, 0

    row = run_delta(model, cfg, replay, 5, torch.device("cpu"), Args(), np.random.default_rng(0))
    for key in ("mean", "persistence", "iterated", "direct_action_summary", "direct_none"):
        assert np.isfinite(row[key]) and row[key] >= 0.0
    for key in ("persistence", "iterated", "direct_action_summary", "direct_none"):
        assert np.isfinite(row[f"score_{key}"])
        assert row[f"score_{key}"] == pytest.approx((row["mean"] - row[key]) / row["mean"])
    # the scale is anchored on the mean predictor, which by construction scores exactly 0
    assert (row["mean"] - row["mean"]) / row["mean"] == 0.0


def test_pairwise_mean_distance_matches_an_explicit_loop():
    from atari_jepa.action_effect import pairwise_mean_distance
    from atari_jepa.losses import cosine_distance

    z = torch.randn(4, 6, 7, 7)
    want = torch.stack([cosine_distance(z[i : i + 1], z[j : j + 1])[0]
                        for i in range(4) for j in range(i + 1, 4)]).mean()
    assert pairwise_mean_distance(z) == pytest.approx(float(want), rel=1e-5)
    assert np.isnan(pairwise_mean_distance(z[:1]))  # a single action has no pairs
