"""Decisive-state catch oracle, binned by lead time (evaluation-only; no training, no project files).

From cloned emulator states visited by the arm's own eps-greedy Q policy, take each action for real,
then follow the arm's greedy Q policy (eps 0) for up to LOOK decisions with the shared sticky RNG.
Record for each branch when (if ever) a life is lost. A state is DECISIVE when at least one branch
loses the ball at lead time tau and at least one branch survives well past tau. catch_a = 1 for the
surviving branches. Score four estimators by whether their argmax is a catch branch:

  actor   Q(z, a)                                      (the Q controller)
  critic  r_a + gamma * max_b Q(f(x'_a))               (Q head on the REAL successor, no model)
  model   r_hat + gamma * c_hat * max_b Q(g(z, a))     (the H=1 planner)
  beam5   best 5-step beam-search score per first action (the H=5 planner, width 16)

Also records, on decisive states only: tau(model, critic), the one-step counterfactual error, and the
across-action range of the critic's and the model's value term.
"""
import json
import sys
import time

import numpy as np
import torch

from atari_jepa.checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from atari_jepa.envs import FrameStacker, make_env
from atari_jepa.losses import cosine_distance, dynamics_step, expected_reward

PROBES, EVERY, EPS_VISIT, LOOK, GAP, BEAM_H, BEAM_W = 300, 4, 0.05, 40, 6, 5, 16


def kendall(a, b):
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return s / (n * (n - 1) / 2)


@torch.no_grad()
def greedy(model, hist, device):
    z = model.encoder(torch.from_numpy(hist).to(device).unsqueeze(0))
    return int(model.q_head(z).argmax(-1))


@torch.no_grad()
def beam_scores_per_first(model, z, horizon, width, gamma):
    A, device = model.num_actions, z.device
    zs, J, S = z, torch.zeros(1, device=device), torch.ones(1, device=device)
    first, state = None, None
    for k in range(horizon):
        N = zs.shape[0]
        zr = zs.repeat_interleave(A, 0)
        acts = torch.arange(A, device=device).repeat(N)
        reward = expected_reward(model.reward_head(zr, acts))
        cont = torch.sigmoid(model.continuation_head(zr, acts))
        sr = None if state is None else state.repeat_interleave(A, 0)
        next_z, state = dynamics_step(model.dynamics, zr, acts, sr)
        Jr, Sr = J.repeat_interleave(A), S.repeat_interleave(A)
        J = Jr + gamma**k * Sr * reward
        S = Sr * cont
        first = acts if k == 0 else first.repeat_interleave(A)
        value = model.q_head(next_z).max(-1).values
        score = J + gamma ** (k + 1) * S * value
        if k == horizon - 1:
            out = np.full(A, -1e9)
            for a in range(A):
                m = first == a
                if bool(m.any()):
                    out[a] = float(score[m].max())
            return out
        order = torch.sort(score, descending=True, stable=True).indices[:width]
        zs, J, S, first = next_z[order], J[order], S[order], first[order]
        state = None if state is None else state[order]


def branch(env, model, snap, hist0, a, device):
    """Take a, then greedy Q for LOOK decisions. Returns (successor stack, clipped r, t_loss or None)."""
    env.restore_state(snap)
    lives0 = env._ale.lives()
    obs, r, term, trunc, _ = env.step(a)
    succ = np.concatenate([hist0[1:], obs[None]], 0)
    hist = succ
    if env._ale.lives() < lives0 or term or trunc:
        return succ, float(np.sign(r)), 0
    for j in range(1, LOOK + 1):
        act = greedy(model, hist, device)
        obs, _, term, trunc, _ = env.step(act)
        hist = np.concatenate([hist[1:], obs[None]], 0)
        if env._ale.lives() < lives0 or term or trunc:
            return succ, float(np.sign(r)), j
    return succ, float(np.sign(r)), None


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
    rng = np.random.default_rng(51_000 + seed)
    recs = []
    probes, ep = 0, 0
    t0 = time.time()
    while probes < PROBES and ep < 10:
        obs, _ = env.reset(seed=51_000 + seed * 10 + ep)
        history = stacker.reset(obs)
        for t in range(3000):
            if t % EVERY == 0 and probes < PROBES:
                snap = env.clone_state()
                ram = env.state_vector()
                succ, real_r, t_loss = [], np.zeros(A), []
                for a in range(A):
                    s, r, tl = branch(env, model, snap, history, a, device)
                    succ.append(s); real_r[a] = r; t_loss.append(tl)
                env.restore_state(snap)
                losses = [tl for tl in t_loss if tl is not None]
                if losses:
                    tau = min(losses)
                    catch = np.array([1.0 if (tl is None or tl > tau + GAP) else 0.0 for tl in t_loss])
                    decisive = 0 < catch.sum() < A
                else:
                    tau, catch, decisive = None, np.ones(A), False
                if decisive:
                    z = model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))
                    acts = torch.arange(A, device=device)
                    q_root = model.q_head(z)[0].cpu().numpy()
                    z_true = model.encoder(torch.from_numpy(np.stack(succ)).to(device))
                    v_true = model.q_head(z_true).max(-1).values.cpu().numpy()
                    critic = real_r + g * v_true
                    zr = z.expand(A, *z.shape[1:])
                    z_pred = model.dynamics(zr, acts)
                    r_hat = expected_reward(model.reward_head(zr, acts)).cpu().numpy()
                    c_hat = torch.sigmoid(model.continuation_head(zr, acts)).cpu().numpy()
                    v_pred = model.q_head(z_pred).max(-1).values.cpu().numpy()
                    model_v = r_hat + g * c_hat * v_pred
                    beam = beam_scores_per_first(model, z, BEAM_H, BEAM_W, g)
                    err = cosine_distance(z_pred, z_true).cpu().numpy()
                    recs.append({
                        "tau": int(tau), "catch": catch.tolist(), "chance": float(catch.mean()),
                        "ball_y": int(ram[101]), "ball_x": int(ram[99]), "paddle_x": int(ram[72]),
                        "actor_ok": float(catch[int(q_root.argmax())]),
                        "critic_ok": float(catch[int(critic.argmax())]),
                        "model_ok": float(catch[int(model_v.argmax())]),
                        "beam5_ok": float(catch[int(beam.argmax())]),
                        "tau_model_critic": float(kendall(model_v, critic)),
                        "tau_actor_critic": float(kendall(q_root, critic)),
                        "tau_critic_catch": float(kendall(critic, catch)),
                        "tau_model_catch": float(kendall(model_v, catch)),
                        "tau_actor_catch": float(kendall(q_root, catch)),
                        "tau_beam_catch": float(kendall(beam, catch)),
                        "cf_err": float(err.mean()),
                        "v_true_range": float(v_true.max() - v_true.min()),
                        "v_pred_range": float(v_pred.max() - v_pred.min()),
                        "q_range": float(q_root.max() - q_root.min()),
                    })
                probes += 1
            z = model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))
            action = int(model.q_head(z).argmax(-1))
            if rng.random() < EPS_VISIT:
                action = int(rng.integers(A))
            obs, _, term, trunc, _ = env.step(action)
            history = stacker.push(obs)
            if term or trunc:
                break
        ep += 1
    env.close()
    n = len(recs)
    m = lambda k, rr=recs: float(np.mean([r[k] for r in rr])) if rr else float("nan")
    summary = {"K": cfg.replay.rollout_steps, "seed": seed, "probes": probes, "decisive": n,
               "elapsed_s": time.time() - t0}
    for k in ("chance", "actor_ok", "critic_ok", "model_ok", "beam5_ok", "tau_model_critic", "tau_actor_critic",
              "tau_critic_catch", "tau_model_catch", "tau_actor_catch", "tau_beam_catch", "cf_err",
              "v_true_range", "v_pred_range", "q_range", "tau"):
        summary[k] = m(k)
    bins = [(0, 2), (3, 5), (6, 9), (10, 40)]
    for lo, hi in bins:
        rr = [r for r in recs if lo <= r["tau"] <= hi]
        summary[f"bin_{lo}_{hi}"] = {"n": len(rr), **{k: m(k, rr) for k in ("chance", "actor_ok", "critic_ok", "model_ok", "beam5_ok")}}
    print(f"{name} seed{seed} K={summary['K']:2d} decisive {n}/{probes} chance {summary['chance']:.2f} | "
          f"catch-acc actor {summary['actor_ok']:.3f} critic {summary['critic_ok']:.3f} model {summary['model_ok']:.3f} "
          f"beam5 {summary['beam5_ok']:.3f} | tau(model,critic) {summary['tau_model_critic']:+.3f} "
          f"cf_err {summary['cf_err']:.4f} vrange true {summary['v_true_range']:.3f} pred {summary['v_pred_range']:.3f} "
          f"| mean lead {summary['tau']:.1f} | {summary['elapsed_s']:.0f}s", flush=True)
    for lo, hi in bins:
        b = summary[f"bin_{lo}_{hi}"]
        print(f"    lead {lo:2d}-{hi:2d}: n={b['n']:3d} chance {b['chance']:.2f} actor {b['actor_ok']:.2f} "
              f"critic {b['critic_ok']:.2f} model {b['model_ok']:.2f} beam5 {b['beam5_ok']:.2f}", flush=True)
    return summary, recs


if __name__ == "__main__":
    out = {}
    for name in sys.argv[1:]:
        out[name] = []
        for s in range(3):
            try:
                summary, recs = run(name, s)
            except FileNotFoundError as exc:
                print("skip", name, s, exc, flush=True)
                continue
            out[name].append({"summary": summary, "records": recs})
            json.dump(out, open("/tmp/decisive_probe.json", "w"))
    print("PROBE_DONE", flush=True)
