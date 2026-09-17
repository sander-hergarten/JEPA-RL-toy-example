"""Does the model rank actions the way the real successors do, under the same Q head?

From one cloned emulator state, take every action for real, encode each true successor, and score it
with the checkpoint's own Q head: v_real[a] = max_b Q(f(x'_a)). Score the model's predicted successor
the same way: v_model[a] = max_b Q(g(z, a)). If the two orderings agree, the planner's one-step
lookahead is grounded; if they diverge while the latent-space prediction error is low, the model got
better on its metric and worse on what the planner consumes.
"""
import sys

import numpy as np
import torch

from atari_jepa.checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from atari_jepa.envs import FrameStacker, make_env
from atari_jepa.losses import cosine_distance

PROBES, EVERY, EPS = 300, 5, 0.1


def kendall(a, b):
    n = len(a)
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return s / (n * (n - 1) / 2)


@torch.no_grad()
def run(run_name, seed):
    device = torch.device("cuda")
    d = f"runs/{run_name}/seed{seed}"
    ckpt = load_checkpoint(f"{d}/checkpoint.pt", device)
    cfg, model = model_from_checkpoint(ckpt, device)
    model.eval()
    env = make_env(cfg.env)
    check_compatible(ckpt["env"], env.metadata())
    stacker = FrameStacker(cfg.env.history)
    A = env.num_actions
    rng = np.random.default_rng(31_000 + seed)
    agree, taus, err_taken, err_cf, qgrad_err, best_real_picked = [], [], [], [], [], []
    probes, ep = 0, 0
    while probes < PROBES and ep < 6:
        obs, _ = env.reset(seed=31_000 + seed * 10 + ep)
        history = stacker.reset(obs)
        for t in range(3000):
            if t % EVERY == 0 and probes < PROBES:
                snap = env.clone_state()
                succ = []
                for a in range(A):
                    env.restore_state(snap)
                    nxt, _, _, _, _ = env.step(a)
                    succ.append(np.concatenate([history[1:], nxt[None]], 0))
                env.restore_state(snap)
                z = model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))
                z_true = model.encoder(torch.from_numpy(np.stack(succ)).to(device))
                acts = torch.arange(A, device=device)
                z_pred = model.dynamics(z.expand(A, *z.shape[1:]), acts)
                v_real = model.q_head(z_true).max(-1).values.cpu().numpy()
                v_model = model.q_head(z_pred).max(-1).values.cpu().numpy()
                agree.append(float(v_real.argmax() == v_model.argmax()))
                taus.append(kendall(v_real, v_model))
                # the model's real one-step error on every counterfactual branch (not only the taken action)
                err_cf.append(float(cosine_distance(z_pred, z_true).mean()))
                # regret in real value units of following the model's pick
                best_real_picked.append(float(v_real[v_model.argmax()] - v_real.max()))
                probes += 1
            z = model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))
            action = int(model.q_head(z).argmax(-1))
            if rng.random() < EPS:
                action = int(rng.integers(A))
            obs, _, term, trunc, _ = env.step(action)
            history = stacker.push(obs)
            if term or trunc:
                break
        ep += 1
    env.close()
    return {"K": cfg.replay.rollout_steps, "n": probes, "argmax_agree": float(np.mean(agree)),
            "kendall_tau": float(np.mean(taus)), "cf_one_step_err": float(np.mean(err_cf)),
            "regret_vs_real": float(np.mean(best_real_picked))}


if __name__ == "__main__":
    for name in sys.argv[1:]:
        rows = []
        for s in range(3):
            try:
                rows.append(run(name, s))
            except FileNotFoundError:
                pass
        if not rows:
            continue
        m = lambda k: float(np.mean([r[k] for r in rows]))
        print(f"{name:30s} K={rows[0]['K']:2d} n={rows[0]['n']}  argmax agree {m('argmax_agree'):.3f}  "
              f"kendall tau {m('kendall_tau'):+.3f}  counterfactual 1-step err {m('cf_one_step_err'):.4f}  "
              f"regret {m('regret_vs_real'):+.4f}")
