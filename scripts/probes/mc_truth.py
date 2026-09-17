"""Ground truth for the one-step decision: Monte-Carlo return of each counterfactual branch.

From one cloned emulator state, take each action for real, then roll the arm's own eps-greedy Q policy
forward H decisions with a shared RNG, and record the discounted return. That is the actual value of
choosing that action now, independent of any learned head. Then score three estimators against it:

  actor   Q(z, a)                      what the Q-controller ranks by
  critic  r_a + gamma * max_b Q(f(x'_a)) the Q head applied to the REAL successor (no model involved)
  model   r_hat + gamma * max_b Q(g(z,a)) what the one-step planner ranks by

Agreement with MC (Kendall tau, and the regret of following each estimator's argmax) separates
"the value head got worse" from "the model got worse" from "both are fine and something else is".
"""
import copy
import json
import sys

import numpy as np
import torch

from atari_jepa.checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from atari_jepa.envs import FrameStacker, make_env
from atari_jepa.losses import expected_reward

PROBES, EVERY, EPS, H = 150, 8, 0.05, 60


def kendall(a, b):
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return s / (n * (n - 1) / 2)


@torch.no_grad()
def policy_action(model, history, device, rng, A):
    z = model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))
    a = int(model.q_head(z).argmax(-1))
    return int(rng.integers(A)) if rng.random() < EPS else a


@torch.no_grad()
def mc_return(env, model, stacker, history, first_action, device, A, gamma, seed):
    """Take first_action, then follow the policy for H decisions; discounted clipped return."""
    rng = np.random.default_rng(seed)
    hist = history.copy()
    total, disc = 0.0, 1.0
    obs, r, term, trunc, _ = env.step(first_action)
    total += disc * float(np.sign(r)); disc *= gamma
    hist = stacker_push(stacker, hist, obs)
    if term or trunc:
        return total
    for _ in range(H - 1):
        a = policy_action(model, hist, device, rng, A)
        obs, r, term, trunc, _ = env.step(a)
        total += disc * float(np.sign(r)); disc *= gamma
        hist = stacker_push(stacker, hist, obs)
        if term or trunc:
            break
    return total


def stacker_push(stacker, hist, obs):
    return np.concatenate([hist[1:], obs[None]], 0)


@torch.no_grad()
def run(name, seed):
    device = torch.device("cuda")
    d = f"runs/{name}/seed{seed}"
    ckpt = load_checkpoint(f"{d}/checkpoint.pt", device)
    cfg, model = model_from_checkpoint(ckpt, device)
    model.eval()
    env = make_env(cfg.env)
    check_compatible(ckpt["env"], env.metadata())
    stacker = FrameStacker(cfg.env.history)
    A, g = env.num_actions, cfg.loss.gamma
    rng = np.random.default_rng(41_000 + seed)
    tau = {"actor": [], "critic": [], "model": []}
    regret = {"actor": [], "critic": [], "model": []}
    mc_spread = []
    probes, ep = 0, 0
    while probes < PROBES and ep < 8:
        obs, _ = env.reset(seed=41_000 + seed * 10 + ep)
        history = stacker.reset(obs)
        for t in range(2500):
            if t % EVERY == 0 and probes < PROBES:
                snap = env.clone_state()
                mc = np.zeros(A)
                succ, real_r = [], np.zeros(A)
                for a in range(A):
                    env.restore_state(snap)
                    # real successor + reward for the critic estimator
                    nxt, r, _, _, _ = env.step(a)
                    succ.append(np.concatenate([history[1:], nxt[None]], 0)); real_r[a] = np.sign(r)
                    env.restore_state(snap)
                    mc[a] = mc_return(env, model, stacker, history, a, device, A, g, seed=probes * 7 + 1)
                env.restore_state(snap)
                z = model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))
                acts = torch.arange(A, device=device)
                q_root = model.q_head(z)[0].cpu().numpy()
                z_true = model.encoder(torch.from_numpy(np.stack(succ)).to(device))
                critic = real_r + g * model.q_head(z_true).max(-1).values.cpu().numpy()
                zr = z.expand(A, *z.shape[1:])
                z_pred = model.dynamics(zr, acts)
                r_hat = expected_reward(model.reward_head(zr, acts)).cpu().numpy()
                c_hat = torch.sigmoid(model.continuation_head(zr, acts)).cpu().numpy()
                model_v = r_hat + g * c_hat * model.q_head(z_pred).max(-1).values.cpu().numpy()
                if mc.max() - mc.min() > 1e-9:  # only states where the choice mattered at all
                    for key, est in (("actor", q_root), ("critic", critic), ("model", model_v)):
                        tau[key].append(kendall(mc, est))
                        regret[key].append(float(mc[est.argmax()] - mc.max()))
                    mc_spread.append(float(mc.max() - mc.min()))
                probes += 1
            action = policy_action(model, history, device, rng, A)
            obs, _, term, trunc, _ = env.step(action)
            history = stacker.push(obs)
            if term or trunc:
                break
        ep += 1
    env.close()
    n = len(mc_spread)
    return {"K": cfg.replay.rollout_steps, "probes": probes, "decisive_states": n,
            "mc_spread": float(np.mean(mc_spread)) if n else float("nan"),
            **{f"tau_{k}": float(np.mean(v)) if v else float("nan") for k, v in tau.items()},
            **{f"regret_{k}": float(np.mean(v)) if v else float("nan") for k, v in regret.items()}}


if __name__ == "__main__":
    out = {}
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
        out[name] = {k: m(k) for k in rows[0] if k != "K"} | {"K": rows[0]["K"]}
        print(f"{name:28s} K={rows[0]['K']:2d} decisive {m('decisive_states'):.0f}/{m('probes'):.0f} "
              f"MC spread {m('mc_spread'):.3f} | tau actor {m('tau_actor'):+.3f} critic {m('tau_critic'):+.3f} "
              f"model {m('tau_model'):+.3f} | regret actor {m('regret_actor'):+.4f} critic {m('regret_critic'):+.4f} "
              f"model {m('regret_model'):+.4f}", flush=True)
    json.dump(out, open("/tmp/mc_truth.json", "w"), indent=1)
    print("MC_DONE")
