"""Does credit from a deep rollout step actually reach the early steps?

The LMU was motivated by gradient decay: with K=30, the gradient of a depth-30 term has to travel back
through 30 applications of g, and a memoryless chain attenuates it. The LMU adds a linear path through
the memory (m_K <- m_k, Jacobian A^(K-k)) that skips the intervening nonlinear steps, so if decay is
the binding constraint it should show up as a flatter profile here.

Measures ||d L_K / d z_hat[k]|| for k = 0..K, where L_K is *only* the deepest JEPA term, so the number
is credit travelling backwards through the rollout and nothing else.
"""
import json
import sys

import numpy as np
import torch

from atari_jepa.checkpoint import load_checkpoint, model_from_checkpoint
from atari_jepa.losses import dynamics_step, jepa_distances
from atari_jepa.replay import SequenceReplay

BATCH = 64


def profile(run: str, seed: int) -> dict:
    device = torch.device("cuda")
    d = f"runs/{run}/seed{seed}"
    ckpt = load_checkpoint(f"{d}/checkpoint.pt", device)
    cfg, model = model_from_checkpoint(ckpt, device)
    model.eval()
    replay = SequenceReplay.load(f"{d}/replay.npz", (cfg.env.screen_size, cfg.env.screen_size))
    K = cfg.replay.rollout_steps
    batch = replay.sample(BATCH, K, np.random.default_rng(0)).to_torch(device)

    z0 = model.encoder(batch.observations[:, 0])
    z_hat, state = [z0], None
    for k in range(K):
        z_next, state = dynamics_step(model.dynamics, z_hat[k], batch.actions[:, k], state)
        z_hat.append(z_next)
    for z in z_hat:
        z.retain_grad()

    with torch.no_grad():
        target_z = model.target_encoder(batch.observations[:, 1:].flatten(0, 1)).unflatten(0, (BATCH, K))
        target_z0 = model.target_encoder(batch.observations[:, 0])
    dist = jepa_distances(z_hat, target_z, target_z0, cfg.loss.jepa_target, cfg.loss.cosine_eps)
    deepest = dist[:, K - 1].mean()  # only the depth-K term
    model.zero_grad(set_to_none=True)
    deepest.backward()

    norms = [float(z.grad.norm()) for z in z_hat]
    return {"run": run, "seed": seed, "K": K, "kind": cfg.network.dynamics_kind, "norms": norms}


if __name__ == "__main__":
    out = []
    for run in sys.argv[1:]:
        for seed in range(3):
            try:
                out.append(profile(run, seed))
            except FileNotFoundError as exc:
                print(f"{run} seed{seed}: skipped ({exc})")
    for run in {r["run"] for r in out}:
        rows = [r for r in out if r["run"] == run]
        K = rows[0]["K"]
        mean = np.mean([r["norms"] for r in rows], axis=0)
        top = mean[-1]
        # report as a fraction of the gradient at the deepest latent, so runs are comparable
        frac = mean / max(top, 1e-12)
        idx = [0, max(K // 4, 1), max(K // 2, 1), max(3 * K // 4, 1), K - 1, K]
        idx = sorted(set(min(i, K) for i in idx))
        print(f"{run:32s} ({rows[0]['kind']}, K={K})  " +
              "  ".join(f"k={i}:{frac[i]:.2e}" for i in idx))
        print(f"{'':32s}  reach(k=0/k=K) = {frac[0]:.3e}")
    json.dump(out, open("/tmp/gradflow.json", "w"), indent=1)
