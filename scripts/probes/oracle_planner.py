"""Read-only oracle-successor evaluation (prints only). Three H=1 controllers on the same checkpoint:
  model   : standard planner  E[r] + g*c*max Q(g(z,a))                (what the dossier evaluates)
  critic  : real successors   sign(r_a) + g*(1-term_a)*max Q(f(x'_a))  (emulator clone, no model)
  hybrid  : real ball + model paddle: max Q(mean_a f(x'_a) + (g(z,a) - mean_a g(z,a)))
Episodes: EPISODES per seed, eps 0.01, reset seeds 10000+, capped at CAP decisions.
"""
import sys
import numpy as np
import torch

from atari_jepa.checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from atari_jepa.envs import FrameStacker, make_env
from atari_jepa.losses import expected_reward

EPISODES, CAP, EPS = 3, 2500, 0.01


@torch.no_grad()
def play(model, cfg, ckpt, device, controller, seed):
    env = make_env(cfg.env)
    check_compatible(ckpt["env"], env.metadata())
    stacker = FrameStacker(cfg.env.history)
    A, g = env.num_actions, cfg.loss.gamma
    rng = np.random.default_rng(1234 + seed)
    acts = torch.arange(A, device=device)
    returns, lengths, disagree = [], [], []
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=10000 + ep)
        history = stacker.reset(obs)
        total, n, dis = 0.0, 0, 0
        for t in range(CAP):
            z = model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))
            q_action = int(model.q_head(z).argmax(-1))
            zr = z.expand(A, *z.shape[1:])
            if controller == "model":
                zp = model.dynamics(zr, acts)
                score = expected_reward(model.reward_head(zr, acts)) + g * torch.sigmoid(model.continuation_head(zr, acts)) * model.q_head(zp).max(-1).values
            else:
                snap = env.clone_state()
                succ, rs, terms = [], np.zeros(A), np.zeros(A)
                for a in range(A):
                    env.restore_state(snap)
                    nxt, r, term, trunc, _ = env.step(a)
                    succ.append(np.concatenate([history[1:], nxt[None]], 0)); rs[a] = np.sign(r); terms[a] = float(term)
                env.restore_state(snap)
                zt = model.encoder(torch.from_numpy(np.stack(succ)).to(device)).flatten(1)
                if controller == "critic":
                    v = model.q_head(zt).max(-1).values.cpu().numpy()
                    score = torch.from_numpy(rs + g * (1 - terms) * v)
                else:  # hybrid: real ball (common mode), model paddle (differential)
                    zp = model.dynamics(zr, acts).flatten(1)
                    hyb = zt.mean(0, keepdim=True) + (zp - zp.mean(0, keepdim=True))
                    score = expected_reward(model.reward_head(zr, acts)) + g * torch.sigmoid(model.continuation_head(zr, acts)) * model.q_head(hyb).max(-1).values
            action = int(torch.as_tensor(score).argmax())
            dis += int(action != q_action)
            if rng.random() < EPS:
                action = int(rng.integers(A))
            obs, r, term, trunc, _ = env.step(action)
            history = stacker.push(obs)
            total += float(r); n += 1
            if term or trunc:
                break
        returns.append(total); lengths.append(n); disagree.append(dis / max(n, 1))
    env.close()
    return returns, lengths, float(np.mean(disagree))


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
            for ctrl in ("critic", "model", "hybrid"):
                rets, lens, dis = play(model, cfg, ckpt, device, ctrl, s)
                summary.setdefault(ctrl, []).append(float(np.mean(rets)))
                print(f"[{name} s{s} {ctrl}] returns={rets} mean={np.mean(rets):.1f} lengths={lens} disagree_q={dis:.2f}", flush=True)
        for ctrl, v in summary.items():
            print(f"== {name} {ctrl}: per-seed {np.round(v,1).tolist()} mean {np.mean(v):.2f} +- {np.std(v):.2f}", flush=True)
    print("ORACLE_DONE")
