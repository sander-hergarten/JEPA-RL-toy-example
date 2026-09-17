"""Read-only write-in ablation (prints only). H=1 controllers on the same checkpoint, emulator clones:
  proj : model successor with its action displacement projected ONTO the real successors' action
         subspace U = span{f(x'_a) - mean_a f}:   score = max Q( mean_a g + P_U (g(z,a) - mean_a g) )
  orth : only the OFF-manifold part of the model's action displacement kept:
                                                 score = max Q( mean_a g + (I - P_U)(g(z,a) - mean_a g) )
Reward/continuation terms are kept as in the standard planner. EPISODES per seed, eps 0.01, seeds 10000+.
"""
import sys
import numpy as np
import torch

from atari_jepa.checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from atari_jepa.envs import FrameStacker, make_env
from atari_jepa.losses import expected_reward

EPISODES, CAP, EPS = 3, 2500, 0.01


def projector(dt):
    """Orthonormal basis of the row space of dt [A, D] (rank A-1)."""
    U, S, Vh = torch.linalg.svd(dt, full_matrices=False)
    keep = S > 1e-4 * S.max()
    return Vh[keep]  # [r, D]


@torch.no_grad()
def play(model, cfg, ckpt, device, controller, seed):
    env = make_env(cfg.env)
    check_compatible(ckpt["env"], env.metadata())
    stacker = FrameStacker(cfg.env.history)
    A, g = env.num_actions, cfg.loss.gamma
    rng = np.random.default_rng(1234 + seed)
    acts = torch.arange(A, device=device)
    returns, lengths, frac_in_U = [], [], []
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=10000 + ep)
        history = stacker.reset(obs)
        total, n = 0.0, 0
        for t in range(CAP):
            z = model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))
            zr = z.expand(A, *z.shape[1:])
            snap = env.clone_state()
            succ = []
            for a in range(A):
                env.restore_state(snap)
                nxt, r, term, trunc, _ = env.step(a)
                succ.append(np.concatenate([history[1:], nxt[None]], 0))
            env.restore_state(snap)
            zt = model.encoder(torch.from_numpy(np.stack(succ)).to(device)).flatten(1)
            dt = zt - zt.mean(0, keepdim=True)
            B = projector(dt)  # [r, D]
            zp = model.dynamics(zr, acts).flatten(1)
            gbar = zp.mean(0, keepdim=True)
            dp = zp - gbar
            dp_in = (dp @ B.T) @ B
            frac_in_U.append(float((dp_in.norm(dim=1) ** 2).sum() / (dp.norm(dim=1) ** 2).sum()))
            leaf = gbar + dp_in if controller == "proj" else gbar + (dp - dp_in)
            score = expected_reward(model.reward_head(zr, acts)) + g * torch.sigmoid(model.continuation_head(zr, acts)) * model.q_head(leaf).max(-1).values
            action = int(score.argmax())
            if rng.random() < EPS:
                action = int(rng.integers(A))
            obs, r, term, trunc, _ = env.step(action)
            history = stacker.push(obs)
            total += float(r); n += 1
            if term or trunc:
                break
        returns.append(total); lengths.append(n)
    env.close()
    return returns, lengths, float(np.mean(frac_in_U))


if __name__ == "__main__":
    device = torch.device("cuda")
    for name in sys.argv[1:]:
        summary = {}
        for s in range(3):
            d = f"runs/{name}/seed{s}"
            try:
                ckpt = load_checkpoint(f"{d}/checkpoint.pt", device)
            except FileNotFoundError:
                continue
            cfg, model = model_from_checkpoint(ckpt, device)
            model.eval()
            for ctrl in ("proj", "orth"):
                rets, lens, fu = play(model, cfg, ckpt, device, ctrl, s)
                summary.setdefault(ctrl, []).append(float(np.mean(rets)))
                print(f"[{name} s{s} {ctrl}] returns={rets} mean={np.mean(rets):.1f} lengths={lens} frac_dp_in_U={fu:.3f}", flush=True)
        for ctrl, v in summary.items():
            print(f"== {name} {ctrl}: per-seed {np.round(v,1).tolist()} mean {np.mean(v):.2f} +- {np.std(v):.2f}", flush=True)
    print("ABL_DONE")
