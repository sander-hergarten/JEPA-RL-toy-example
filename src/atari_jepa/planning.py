"""Controllers: greedy Q-policy and model-based lookahead from the same checkpoint.

Both controllers encode the current *real* observation history at every decision; a predicted latent is
never carried forward once a real observation is available.

Tie-breaking: exact score ties are broken toward the lowest action index (``torch.argmax`` returns the
first maximal index; beam pruning uses a stable sort over candidates ordered by (beam, action)).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .losses import expected_reward


def leaf_values(model, z: torch.Tensor, bootstrap: str) -> torch.Tensor:
    """max_b of the value used at a search leaf; ``"q"`` is the Q head, ``"sf"`` is w . psi(z, b).

    The successor value has an unbounded horizon obtained by bootstrapping rather than unrolling, so
    swapping it in at the leaf is the cleanest test of whether a longer *value* horizon helps planning.
    """
    q = model.q_from_successor(z) if bootstrap == "sf" else model.q_head(z)
    return q.max(dim=-1).values


@torch.no_grad()
def one_step_scores(model, z: torch.Tensor, gamma: float, bootstrap: str = "q") -> torch.Tensor:
    """score(a) = E[r | z, a] + gamma * P(continue | z, a) * max_b Q(g(z, a))[b]; ``[B, ...] -> [B, A]``."""
    B, A = z.shape[0], model.num_actions
    zr = z.repeat_interleave(A, dim=0)
    acts = torch.arange(A, device=z.device).repeat(B)
    next_z = model.dynamics(zr, acts)
    reward = expected_reward(model.reward_head(zr, acts))
    cont = torch.sigmoid(model.continuation_head(zr, acts))
    value = leaf_values(model, next_z, bootstrap)
    return (reward + gamma * cont * value).view(B, A)


@torch.no_grad()
def beam_search(model, z: torch.Tensor, horizon: int, beam_width: int, gamma: float,
                bootstrap: str = "q") -> tuple[int, float]:
    """Approximate H-step search from a single root latent ``z`` ``[1, ...]``.

    J = sum_{k<H} gamma^k S_k E[R(z_k, a_k)] + gamma^H S_H max_a Q(z_H, a),  S_k = prod_{j<k} C(z_j, a_j).
    Beams are pruned by the bootstrapped partial score J_k + gamma^k S_k max Q(z_k). With
    ``beam_width >= A**(H-1)`` this is an exhaustive search. Returns (first action, best score).
    """
    if z.shape[0] != 1:
        raise ValueError("beam_search plans for one root at a time")
    A = model.num_actions
    device = z.device
    zs = z  # [N, ...] beam latents
    J = torch.zeros(1, device=device)
    S = torch.ones(1, device=device)
    first = torch.full((1,), -1, dtype=torch.long, device=device)
    for k in range(horizon):
        N = zs.shape[0]
        zr = zs.repeat_interleave(A, dim=0)
        acts = torch.arange(A, device=device).repeat(N)
        reward = expected_reward(model.reward_head(zr, acts))
        cont = torch.sigmoid(model.continuation_head(zr, acts))
        next_z = model.dynamics(zr, acts)
        Jr, Sr = J.repeat_interleave(A), S.repeat_interleave(A)
        J = Jr + gamma**k * Sr * reward
        S = Sr * cont
        first = acts if k == 0 else first.repeat_interleave(A)
        value = leaf_values(model, next_z, bootstrap)
        score = J + gamma ** (k + 1) * S * value
        if k == horizon - 1:
            best = int(torch.argmax(score))
            return int(first[best]), float(score[best])
        order = torch.sort(score, descending=True, stable=True).indices[:beam_width]
        zs, J, S, first = next_z[order], J[order], S[order], first[order]
    raise ValueError("horizon must be >= 1")


@torch.no_grad()
def hierarchical_scores(model, z: torch.Tensor, gamma: float, sequences: torch.Tensor) -> torch.Tensor:
    """Score candidate action sequences with the level-2 jumpy model; ``[N]`` for a single root.

    J(seq) = E[R_macro(z, seq)] + gamma^H * P(continue through the window) * max_b Q(g_H(z, seq))

    One forward pass per candidate over the whole window, instead of H unrolls of the one-step model.
    """
    n = sequences.shape[0]
    zr = z.expand(n, *z.shape[1:])
    ret = model.macro_return(zr, sequences)
    cont = torch.sigmoid(model.macro_continuation(zr, sequences))
    value = model.q_head(model.macro_dynamics(zr, sequences)).max(dim=-1).values  # level 2 always bootstraps on Q
    return ret + gamma ** sequences.shape[1] * cont * value


def candidate_sequences(num_actions: int, horizon: int, budget: int, generator: torch.Generator | None,
                        device) -> torch.Tensor:
    """All ``A**H`` sequences when that fits in the budget, else the constant ones plus random samples."""
    total = num_actions**horizon
    if total <= budget:
        grid = torch.cartesian_prod(*[torch.arange(num_actions) for _ in range(horizon)])
        return grid.reshape(total, horizon).to(device)
    constant = torch.arange(num_actions).repeat_interleave(horizon).view(num_actions, horizon)
    sampled = torch.randint(0, num_actions, (budget - num_actions, horizon), generator=generator)
    return torch.cat([constant, sampled]).to(device)


class Controller:
    name = "base"

    def __init__(self, model, device: torch.device, epsilon: float, seed: int):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.latencies: list[float] = []
        self.disagreements = 0
        self.decisions = 0

    def _choose(self, z: torch.Tensor) -> tuple[int, int]:
        """Return (controller action, greedy Q action) for a latent ``[1, ...]``."""
        raise NotImplementedError

    @torch.no_grad()
    def act(self, history: np.ndarray) -> tuple[int, dict[str, Any]]:
        start = time.perf_counter()
        obs = torch.from_numpy(history).to(self.device).unsqueeze(0)
        z = self.model.encoder(obs)
        action, q_action = self._choose(z)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.latencies.append(time.perf_counter() - start)
        self.decisions += 1
        self.disagreements += int(action != q_action)
        explored = bool(self.epsilon > 0 and self.rng.random() < self.epsilon)
        if explored:
            action = int(self.rng.integers(self.model.num_actions))
        return action, {"q_action": q_action, "explored": explored}

    def stats(self) -> dict[str, float]:
        lat = np.asarray(self.latencies) * 1e3 if self.latencies else np.zeros(1)
        return {
            "decisions": self.decisions,
            "latency_ms_mean": float(lat.mean()),
            "latency_ms_p95": float(np.percentile(lat, 95)),
            "disagreement_with_q": self.disagreements / max(self.decisions, 1),
        }


class QController(Controller):
    name = "q"

    def _choose(self, z):
        q_action = int(self.model.q_head(z).argmax(dim=-1))
        return q_action, q_action


class LookaheadController(Controller):
    name = "lookahead"

    def __init__(self, model, device, epsilon, seed, gamma: float, horizon: int = 1, beam_width: int = 16,
                 bootstrap: str = "q"):
        super().__init__(model, device, epsilon, seed)
        self.gamma = gamma
        self.horizon = horizon
        self.beam_width = beam_width
        self.bootstrap = bootstrap

    def _choose(self, z):
        q_action = int(self.model.q_head(z).argmax(dim=-1))
        if self.horizon == 1:
            action = int(one_step_scores(self.model, z, self.gamma, self.bootstrap).argmax(dim=-1))
        else:
            action, _ = beam_search(self.model, z, self.horizon, self.beam_width, self.gamma, self.bootstrap)
        return action, q_action


class SuccessorController(Controller):
    """Greedy on Q_sf(z, a) = w . psi(z, a): an unbounded value horizon, no rollout at all."""

    name = "sf"

    def _choose(self, z):
        q_action = int(self.model.q_head(z).argmax(dim=-1))
        return int(self.model.q_from_successor(z).argmax(dim=-1)), q_action


class HierarchicalController(Controller):
    """Plan with the level-2 jumpy model over action sequences, execute the first action, replan."""

    name = "hierarchical"

    def __init__(self, model, device, epsilon, seed, gamma, budget=128):
        super().__init__(model, device, epsilon, seed)
        self.gamma = gamma
        gen = torch.Generator().manual_seed(seed)
        self.sequences = candidate_sequences(model.num_actions, model.macro_horizon, budget, gen, device)

    def _choose(self, z):
        q_action = int(self.model.q_head(z).argmax(dim=-1))
        scores = hierarchical_scores(self.model, z, self.gamma, self.sequences)
        return int(self.sequences[int(torch.argmax(scores)), 0]), q_action


def make_controller(name: str, model, cfg, device: torch.device, epsilon: float, seed: int,
                    horizon: int | None = None, bootstrap: str | None = None):
    bootstrap = bootstrap or cfg.planning.bootstrap
    if bootstrap == "sf" and getattr(model, "successor_head", None) is None:
        raise ValueError("bootstrap 'sf' needs a checkpoint trained with loss.successor")
    if name == "hierarchical":
        if getattr(model, "macro_dynamics", None) is None:
            raise ValueError("controller 'hierarchical' needs a checkpoint trained with loss.hierarchical")
        return HierarchicalController(model, device, epsilon, seed, cfg.loss.gamma, cfg.planning.macro_candidates)
    if name == "q":
        return QController(model, device, epsilon, seed)
    if name == "sf":
        if getattr(model, "successor_head", None) is None:
            raise ValueError("controller 'sf' needs a checkpoint trained with loss.successor")
        return SuccessorController(model, device, epsilon, seed)
    if name == "lookahead":
        return LookaheadController(
            model,
            device,
            epsilon,
            seed,
            gamma=cfg.loss.gamma,
            horizon=horizon or cfg.planning.horizon,
            beam_width=cfg.planning.beam_width,
            bootstrap=bootstrap,
        )
    raise ValueError(f"unknown controller {name!r}; choose 'q', 'sf', 'lookahead' or 'hierarchical'")
