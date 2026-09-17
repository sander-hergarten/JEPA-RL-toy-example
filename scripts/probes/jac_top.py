"""Top singular values of one dynamics step by power iteration, plus how many exceed 1.

The randomized probe estimate measures the gain in a *random* direction (~0.87 at K=30). That cannot
explain a backward gradient that grows 5.4x over thirty steps, so either the estimate misses the top of
the spectrum or something else amplifies. Power iteration on J^T J finds the real top; deflating finds
how many directions exceed unit gain, i.e. the dimension of the subspace that survives the rollout.
"""
import sys

import numpy as np
import torch

from atari_jepa.checkpoint import load_checkpoint, model_from_checkpoint
from atari_jepa.losses import dynamics_step
from atari_jepa.replay import SequenceReplay

ITERS, NVEC = 30, 12


def top_singular_values(run: str, seed: int = 0):
    device = torch.device("cuda")
    d = f"runs/{run}/seed{seed}"
    ckpt = load_checkpoint(f"{d}/checkpoint.pt", device)
    cfg, model = model_from_checkpoint(ckpt, device)
    model.eval()
    replay = SequenceReplay.load(f"{d}/replay.npz", (cfg.env.screen_size, cfg.env.screen_size))
    K = cfg.replay.rollout_steps
    batch = replay.sample(1, K, np.random.default_rng(0)).to_torch(device)
    with torch.no_grad():
        z = model.encoder(batch.observations[:, 0])
    shape, action = z.shape, batch.actions[:, 0]
    D = int(np.prod(shape[1:]))
    base = z.flatten().detach().clone()

    def fwd(flat):
        zk, _ = dynamics_step(model.dynamics, flat.view(shape), action, None)
        return zk.flatten()

    def jvp(v):
        return torch.func.jvp(fwd, (base,), (v,))[1].detach()

    def vjp(u):
        _, f = torch.func.vjp(fwd, base)
        return f(u)[0].detach()

    # simultaneous (block) power iteration on J^T J, with re-orthonormalization each step
    V = torch.linalg.qr(torch.randn(D, NVEC, device=device))[0]
    for _ in range(ITERS):
        W = torch.stack([vjp(jvp(V[:, j])) for j in range(NVEC)], dim=1)
        V = torch.linalg.qr(W)[0]
    sv = []
    for j in range(NVEC):
        Jv = jvp(V[:, j])
        sv.append(float(Jv.norm()))
    return sorted(sv, reverse=True), K


if __name__ == "__main__":
    for run in sys.argv[1:]:
        try:
            sv, K = top_singular_values(run)
        except FileNotFoundError as exc:
            print(f"{run}: skipped ({exc})")
            continue
        above = sum(1 for s in sv if s > 1.0)
        print(f"{run:28s} K={K:2d}  top-{NVEC} sigma: " + " ".join(f"{s:.3f}" for s in sv))
        print(f"{'':28s}         sigma_1 {sv[0]:.3f}   above 1.0: {above} of {NVEC} probed")
