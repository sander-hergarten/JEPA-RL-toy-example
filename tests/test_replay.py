"""Replay causality, padding, episode boundaries and eviction."""

import numpy as np
import pytest
from conftest import frame, ids

from atari_jepa.replay import SequenceReplay


def fill(buf: SequenceReplay, lengths: list[int], end: str = "terminated", first_episode: int = 0):
    """Write episodes of the given lengths; each ends terminated/truncated (last one may be left open)."""
    for e, length in enumerate(lengths, start=first_episode):
        buf.start_episode(frame(e, 0))
        for t in range(length):
            last = t == length - 1
            buf.add(t % 4, float(t == 1), last and end == "terminated", last and end == "truncated", frame(e, t + 1))


def test_stacks_are_causal_and_padded_with_first_observation():
    buf = SequenceReplay(100, (84, 84), history=4)
    fill(buf, [6])
    batch = buf.gather(np.array([0, 1, 2, 4]), horizon=1)
    assert ids(batch.observations[0, 0]) == [(0, 0)] * 4
    assert ids(batch.observations[1, 0]) == [(0, 0), (0, 0), (0, 0), (0, 1)]
    assert ids(batch.observations[2, 0]) == [(0, 0), (0, 0), (0, 1), (0, 2)]
    assert ids(batch.observations[3, 0]) == [(0, 1), (0, 2), (0, 3), (0, 4)]
    # next observation stack is the actual next time step
    assert ids(batch.observations[3, 1]) == [(0, 2), (0, 3), (0, 4), (0, 5)]


def test_sequences_never_cross_episode_boundaries():
    buf = SequenceReplay(100, (84, 84), history=4)
    fill(buf, [3, 4])
    # root at t=1 of episode 0: transitions 1, 2 (terminal) valid, then padding
    batch = buf.gather(np.array([1]), horizon=5)
    assert batch.valid[0].tolist() == [True, True, False, False, False]
    assert batch.terminated[0].tolist() == [False, True, False, False, False]
    # observation after the terminal transition is the real final frame of episode 0
    assert ids(batch.observations[0, 2])[-1] == (0, 3)
    # everything later is zero padding, never the next episode's reset frame
    assert (batch.observations[0, 3:] == 0).all()
    assert (batch.actions[0, 2:] == 0).all() and (batch.rewards[0, 2:] == 0).all()


def test_sampled_batches_are_internally_consistent():
    rng = np.random.default_rng(0)
    buf = SequenceReplay(500, (84, 84), history=4)
    fill(buf, list(rng.integers(1, 12, size=30)))
    batch = buf.sample(256, horizon=5, rng=rng)
    assert len(set(batch.roots.tolist())) > 50  # spread across the buffer
    for b in range(256):
        ep0, t0 = ids(batch.observations[b, 0])[-1]
        n_valid = int(batch.valid[b].sum())
        assert batch.valid[b, :n_valid].all() and not batch.valid[b, n_valid:].any()
        for k in range(n_valid + 1):
            stack = ids(batch.observations[b, k])
            assert all(e == ep0 for e, _ in stack), "stack mixes episodes"
            ts = [t for _, t in stack]
            assert ts[-1] == t0 + k and ts == sorted(ts), "non-causal history"
        ended = batch.terminated[b] | batch.truncated[b]
        if ended.any():
            assert int(np.argmax(ended)) == n_valid - 1


def test_eviction_keeps_order_and_rejects_roots_with_evicted_history():
    buf = SequenceReplay(20, (84, 84), history=4)
    fill(buf, [30])  # one long episode, 31 frames written into 20 slots
    assert buf.oldest == 11 and len(buf) == 20
    roots = buf.valid_roots()
    # the root must have its 3 previous frames still stored: abs index >= oldest + 3
    assert roots.min() == 14 and roots.max() == 29
    batch = buf.gather(roots, horizon=2)
    for b, r in enumerate(roots):
        assert [t for _, t in ids(batch.observations[b, 0])] == [r - 3, r - 2, r - 1, r]
    rng = np.random.default_rng(1)
    assert set(buf.sample(200, 2, rng).roots.tolist()) <= set(roots.tolist())


def test_eviction_of_episode_start_rejects_padded_roots():
    buf = SequenceReplay(10, (84, 84), history=4)
    fill(buf, [2, 20])
    roots = buf.valid_roots()
    with pytest.raises(ValueError):
        buf.gather(np.array([buf.oldest]), horizon=1)
    assert all(r - min(buf.step_in_episode[r % 10], 3) >= buf.oldest for r in roots)


def test_open_episode_and_collection_boundary():
    buf = SequenceReplay(100, (84, 84), history=4)
    buf.start_episode(frame(0, 0))
    for t in range(3):
        buf.add(1, 0.0, False, False, frame(0, t + 1))
    # a partial episode can be sampled; transitions that do not exist yet are invalid
    batch = buf.gather(np.array([1]), horizon=4)
    assert batch.valid[0].tolist() == [True, True, False, False]
    # a collection boundary (e.g. resume) starts a new episode without closing the old one
    buf.start_episode(frame(1, 0))
    buf.add(2, 0.0, False, False, frame(1, 1))
    batch = buf.gather(np.array([2]), horizon=4)
    assert batch.valid[0].tolist() == [True, False, False, False]
    assert (batch.observations[0, 2:] == 0).all()
    assert 3 not in buf.valid_roots().tolist()  # the unfinished episode's last frame has no transition


def test_persistence_round_trip(tmp_path):
    buf = SequenceReplay(50, (84, 84), history=4)
    fill(buf, [10, 30, 8])
    buf.save(tmp_path / "replay.npz")
    loaded = SequenceReplay.load(tmp_path / "replay.npz", (84, 84))
    assert loaded.n_written == buf.n_written and loaded.n_episodes == buf.n_episodes
    np.testing.assert_array_equal(loaded.valid_roots(), buf.valid_roots())
    roots = buf.valid_roots()
    a, b = buf.gather(roots, 5), loaded.gather(roots, 5)
    for name in ("observations", "actions", "rewards", "terminated", "truncated", "valid"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    with pytest.raises(RuntimeError):
        loaded.add(0, 0.0, False, False, frame(9, 9))  # loaded episodes stay closed


def test_stacks_at_matches_gather_and_rejects_out_of_range():
    buf = SequenceReplay(100, (84, 84), history=4)
    fill(buf, [6])
    roots = np.array([0, 2, 4])
    np.testing.assert_array_equal(buf.stacks_at(roots), buf.gather(roots, 1).observations[:, 0])
    # the episode's final observation has no transition, so it is not a valid root but is still stored
    assert 6 not in buf.valid_roots().tolist()
    assert ids(buf.stacks_at(np.array([6]))[0]) == [(0, 3), (0, 4), (0, 5), (0, 6)]
    with pytest.raises(ValueError):
        buf.stacks_at(np.array([buf.n_written]))
