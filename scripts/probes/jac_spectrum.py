"""Singular-value spectrum of one dynamics step, via a randomized range finder.

If the rollout filtered credit spectrally (the "low-pass" story), the per-step Jacobian would have a
few dominant singular values and a long tail: repeated application would then act like power
iteration, keeping the top direction and suppressing the rest. If instead the singular values cluster
near 1, the rollout is an approximately isometric channel -- it neither attenuates nor filters, which
is what a residual map with LayerNorm is built to be.
"""
import sys

import numpy as np
import torch

from atari_jepa.checkpoint import load_checkpoint, model_from_checkpoint
from atari_jepa.losses import dynamics_step
from atari_jepa.replay import SequenceReplay

R = 48  # probe directions


def spectrum(run: str, seed: int, steps=(0, None)):
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
    shape = z.shape
    D = int(np.prod(shape[1:]))
    out = {}
    for k in [s for s in steps if s is not None] + [K - 1]:
        action = batch.actions[:, k]

        def step(flat):
            zk, _ = dynamics_step(model.dynamics, flat.view(shape), action, None)
            return zk.flatten()

        omega = torch.randn(D, R, device=device)
        omega = omega / omega.norm(dim=0, keepdim=True)
        cols = []
        base = z.flatten().detach().clone()
        for j in range(R):
            _, jv = torch.func.jvp(step, (base,), (omega[:, j],))
            cols.append(jv.detach())
        Y = torch.stack(cols, dim=1)  # [D, R] ~ J @ omega
        sv = torch.linalg.svdvals(Y.double()).cpu().numpy()
        out[k] = sv
    return out, K


if __name__ == "__main__":
    for run in sys.argv[1:]:
        try:
            out, K = spectrum(run, 0)
        except FileNotFoundError as exc:
            print(f"{run}: skipped ({exc})")
            continue
        for k, sv in out.items():
            top = sv[0]
            ratios = [sv[i] / top for i in (0, 1, 3, 7, 15, 31, R - 1) if i < len(sv)]
            print(f"{run:28s} K={K:2d} step {k:2d}  top sigma {top:6.3f}   "
                  f"sigma_i/sigma_1: " + " ".join(f"{r:.3f}" for r in ratios))
            # how isometric: spread of the probed singular values around 1
            print(f"{'':28s}          median sigma {np.median(sv):6.3f}  "
                  f"sigma_max/sigma_min over probes {sv[0] / max(sv[-1], 1e-9):8.2f}")
