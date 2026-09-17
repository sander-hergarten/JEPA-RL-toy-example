"""Where does the one-step planner's action preference come from, and does it survive long K?

score(a) = E[r|z,a] + gamma * P(cont|z,a) * max_b Q(g(z,a))[b]. Decompose the across-action range of
each term, and measure whether the Q head's input gradient is aligned with the direction the dynamics
moves the latent when the action changes. If the value term's range collapses while the latent
displacement does not, the world model and the value function have stopped sharing a subspace.
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from atari_jepa.checkpoint import load_checkpoint, model_from_checkpoint
from atari_jepa.losses import expected_reward
from atari_jepa.replay import SequenceReplay

N = 512


def analyse(run, seed):
    device = torch.device("cuda")
    d = f"runs/{run}/seed{seed}"
    ckpt = load_checkpoint(f"{d}/checkpoint.pt", device)
    cfg, model = model_from_checkpoint(ckpt, device)
    model.eval()
    replay = SequenceReplay.load(f"{d}/replay.npz", (cfg.env.screen_size, cfg.env.screen_size))
    roots = replay.sample_roots(N, np.random.default_rng(1))
    obs = torch.from_numpy(replay.stacks_at(roots)).to(device)
    A, g = model.num_actions, cfg.loss.gamma
    with torch.no_grad():
        z = model.encoder(obs)
        q = model.q_head(z)
        zr = z.repeat_interleave(A, 0)
        acts = torch.arange(A, device=device).repeat(N)
        nz = model.dynamics(zr, acts)
        r = expected_reward(model.reward_head(zr, acts)).view(N, A)
        c = torch.sigmoid(model.continuation_head(zr, acts)).view(N, A)
        v = model.q_head(nz).max(-1).values.view(N, A)
        score = r + g * c * v
        rng_ = lambda x: float((x.max(1).values - x.min(1).values).mean())
        # displacement of the predicted next latent between actions, vs the Q head's sensitivity direction
        nz4 = nz.view(N, A, -1)
        disp = (nz4 - nz4.mean(1, keepdim=True)).flatten(1)  # action-driven part of the prediction
        disp_norm = float(disp.norm(dim=1).mean())
        z_flat = z.flatten(1)
        z_norm = float(z_flat.norm(dim=1).mean())
    # Q head input gradient at the predicted next latents (direction along which Q changes)
    nz_req = nz.detach().clone().requires_grad_(True)
    qn = model.q_head(nz_req).max(-1).values.sum()
    grad = torch.autograd.grad(qn, nz_req)[0].view(N, A, -1)
    # alignment: how much of the action-driven displacement lies along the Q gradient
    gdir = F.normalize(grad.mean(1), dim=1)
    ddir = F.normalize((nz4[:, 0] - nz4[:, 1]).flatten(1), dim=1)
    align = float((gdir * ddir).sum(1).abs().mean())
    # random-direction baseline for that alignment
    rnd = F.normalize(torch.randn_like(ddir), dim=1)
    align_rand = float((gdir * rnd).sum(1).abs().mean())
    model.zero_grad(set_to_none=True)
    return {
        "K": cfg.replay.rollout_steps, "q_range": rng_(q), "score_range": rng_(score),
        "reward_range": rng_(r), "cont_range": rng_(c), "value_range": rng_(v),
        "disp_over_z": disp_norm / z_norm, "q_grad_norm": float(grad.norm(dim=2).mean()),
        "align_disp_qgrad": align, "align_random": align_rand,
        "argmax_agree": float((score.argmax(1) == q.argmax(1)).float().mean()),
    }


if __name__ == "__main__":
    for run in sys.argv[1:]:
        rows = []
        for s in range(3):
            try:
                rows.append(analyse(run, s))
            except FileNotFoundError:
                pass
        if not rows:
            continue
        m = lambda k: float(np.mean([r[k] for r in rows]))
        print(f"{run:30s} K={rows[0]['K']:2d}  Q-range {m('q_range'):.4f}  planner-range {m('score_range'):.4f}  "
              f"[reward {m('reward_range'):.4f} cont {m('cont_range'):.4f} value {m('value_range'):.4f}]  "
              f"disp/|z| {m('disp_over_z'):.4f}  |dQ/dz| {m('q_grad_norm'):.4f}  "
              f"align {m('align_disp_qgrad'):.4f} (rand {m('align_random'):.4f})  agree {m('argmax_agree'):.2f}")
