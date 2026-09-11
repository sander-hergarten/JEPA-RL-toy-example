"""Planning arithmetic on a hand-constructed toy latent model with known outcomes."""

import itertools

import torch
from torch import nn

from atari_jepa.planning import LookaheadController, beam_search, one_step_scores

BIG = 40.0


class ToyModel(nn.Module):
    """Latent = one-hot over S states. Tables give next state, reward, continuation and Q per state."""

    def __init__(self, next_state, reward, cont, q):
        super().__init__()
        self.next_state = torch.tensor(next_state)  # [S, A]
        self.reward = torch.tensor(reward)  # [S, A] in {-1, 0, 1}
        self.cont = torch.tensor(cont, dtype=torch.float32)  # [S, A] in {0, 1}
        self.q = torch.tensor(q, dtype=torch.float32)  # [S, A]
        self.S, self.num_actions = self.next_state.shape

    def _state(self, z):
        return z.argmax(dim=-1)

    def encoder(self, obs):
        return nn.functional.one_hot(obs.flatten(1)[:, 0].long(), self.S).float()

    def dynamics(self, z, a):
        return nn.functional.one_hot(self.next_state[self._state(z), a], self.S).float()

    def reward_head(self, z, a):
        cls = (self.reward[self._state(z), a] + 1).long()
        return nn.functional.one_hot(cls, 3).float() * 2 * BIG - BIG

    def continuation_head(self, z, a):
        return (self.cont[self._state(z), a] * 2 - 1) * BIG

    def q_head(self, z):
        return self.q[self._state(z)]


def root(model, s=0):
    return nn.functional.one_hot(torch.tensor([s]), model.S).float()


def two_action_model(terminal_on_a1: bool):
    # state 0: action 0 -> state 1 (reward 0), action 1 -> state 2 (reward +1)
    # Q(state 1) max 5, Q(state 2) max 100 -> lookahead prefers action 1 unless it terminates
    return ToyModel(
        next_state=[[1, 2], [1, 1], [2, 2]],
        reward=[[0, 1], [0, 0], [0, 0]],
        cont=[[1, 0 if terminal_on_a1 else 1], [1, 1], [1, 1]],
        q=[[0, 0], [5, 1], [100, 0]],
    )


def test_one_step_lookahead_picks_higher_return_action():
    m = two_action_model(terminal_on_a1=False)
    scores = one_step_scores(m, root(m), gamma=0.9)
    torch.testing.assert_close(scores, torch.tensor([[0.9 * 5, 1 + 0.9 * 100]]))
    assert int(scores.argmax()) == 1


def test_continuation_suppresses_value_after_termination():
    m = two_action_model(terminal_on_a1=True)
    scores = one_step_scores(m, root(m), gamma=0.9)
    torch.testing.assert_close(scores, torch.tensor([[4.5, 1.0]]))
    assert int(scores.argmax()) == 0


def test_ties_break_to_lowest_action_index():
    m = ToyModel(next_state=[[0, 0, 0]], reward=[[0, 0, 0]], cont=[[1, 1, 1]], q=[[1, 1, 1]])
    assert int(one_step_scores(m, root(m), 0.9).argmax()) == 0
    assert beam_search(m, root(m), horizon=2, beam_width=9, gamma=0.9)[0] == 0


def brute_force(m, horizon, gamma):
    best, best_first = -float("inf"), None
    for seq in itertools.product(range(m.num_actions), repeat=horizon):
        s, J, S = 0, 0.0, 1.0
        for k, a in enumerate(seq):
            J += gamma**k * S * float(m.reward[s, a])
            S *= float(m.cont[s, a])
            s = int(m.next_state[s, a])
        J += gamma**horizon * S * float(m.q[s].max())
        if J > best:
            best, best_first = J, seq[0]
    return best_first, best


def test_exhaustive_beam_search_matches_brute_force():
    # a trap: action 1 gives +1 now but leads to a -1 chain; action 0 delays a +1
    m = ToyModel(
        next_state=[[1, 2], [3, 3], [2, 2], [3, 3]],
        reward=[[0, 1], [1, 1], [-1, -1], [0, 0]],
        cont=[[1, 1], [1, 1], [1, 1], [0, 0]],
        q=[[0, 0], [0, 0], [0, 0], [50, 50]],
    )
    for horizon in (1, 2, 3):
        first, score = brute_force(m, horizon, gamma=0.9)
        got_first, got_score = beam_search(m, root(m), horizon, beam_width=m.num_actions**horizon, gamma=0.9)
        assert got_first == first, horizon
        assert abs(got_score - score) < 1e-4
    assert beam_search(m, root(m), 1, 16, 0.9)[0] == int(one_step_scores(m, root(m), 0.9).argmax())


def test_controller_replans_from_real_observation_and_logs_disagreement():
    m = two_action_model(terminal_on_a1=False)
    ctrl = LookaheadController(m, torch.device("cpu"), epsilon=0.0, seed=0, gamma=0.9)
    obs = torch.zeros(1, 2, 2, dtype=torch.uint8).numpy()  # encodes state 0
    action, info = ctrl.act(obs)
    assert action == 1 and info["q_action"] == 0
    obs1 = obs.copy()
    obs1[0, 0, 0] = 1  # a *real* observation of state 1 is encoded afresh
    action, info = ctrl.act(obs1)
    assert action == 0
    stats = ctrl.stats()
    assert stats["decisions"] == 2 and stats["disagreement_with_q"] == 0.5
