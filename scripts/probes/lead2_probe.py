"""Decisive-state catch oracle restricted to the last two decisions before the ball reaches the paddle.

State selector (RAM, evaluation-only): ball descending (ball_y rose since the last decision) and
161 <= ball_y <= 176 (catches peak at ball_y 175-177; the ball drops 4-8 per decision). From a cloned
state take each action for real, then the greedy Q policy for up to LOOK decisions; a branch MISSES if
a life is lost inside the window. Decisive = the four branches do not all agree. Score each estimator
by whether its argmax is a catch branch:
  actor Q(z,a) | critic r + g max Q(f(x'_a)) [real successor] | model r^ + g c^ max Q(g(z,a)) [H=1 planner]
  | beam5: best 5-step beam score per first action [H=5 planner].
"""
import json
import sys
import time

import numpy as np
import torch

from atari_jepa.checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from atari_jepa.envs import FrameStacker, make_env
from atari_jepa.losses import cosine_distance, dynamics_step, expected_reward

MAX_STATES, EPS_VISIT, LOOK, BEAM_H, BEAM_W, MAX_EP = 220, 0.05, 12, 5, 16, 12
Y_LO, Y_HI = 161, 176


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
    env.restore_state(snap)
    lives0 = env._ale.lives()
    obs, r, term, trunc, _ = env.step(a)
    succ = np.concatenate([hist0[1:], obs[None]], 0)
    hist = succ
    if env._ale.lives() < lives0 or term or trunc:
        return succ, float(np.sign(r)), False
    for j in range(1, LOOK + 1):
        act = greedy(model, hist, device)
        obs, _, term, trunc, _ = env.step(act)
        hist = np.concatenate([hist[1:], obs[None]], 0)
        if env._ale.lives() < lives0 or term or trunc:
            return succ, float(np.sign(r)), False
    return succ, float(np.sign(r)), True


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
    rng = np.random.default_rng(61_000 + seed)
    recs, n_states, ep = [], 0, 0
    t0 = time.time()
    while n_states < MAX_STATES and ep < MAX_EP:
        obs, _ = env.reset(seed=61_000 + seed * 20 + ep)
        history = stacker.reset(obs)
        prev_y = None
        for t in range(3000):
            ram = env.state_vector()
            by, bx, px = int(ram[101]), int(ram[99]), int(ram[72])
            if prev_y is not None and by > prev_y and Y_LO <= by <= Y_HI and n_states < MAX_STATES:
                snap = env.clone_state()
                succ, real_r, catch = [], np.zeros(A), np.zeros(A)
                for a in range(A):
                    s, r, c = branch(env, model, snap, history, a, device)
                    succ.append(s); real_r[a] = r; catch[a] = float(c)
                env.restore_state(snap)
                n_states += 1
                if 0 < catch.sum() < A:
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
                        "ball_y": by, "gap": bx - px - 9, "chance": float(catch.mean()), "catch": catch.tolist(),
                        "actor_ok": float(catch[int(q_root.argmax())]), "critic_ok": float(catch[int(critic.argmax())]),
                        "model_ok": float(catch[int(model_v.argmax())]), "beam5_ok": float(catch[int(beam.argmax())]),
                        "tau_model_critic": float(kendall(model_v, critic)), "tau_model_catch": float(kendall(model_v, catch)),
                        "tau_critic_catch": float(kendall(critic, catch)), "tau_actor_catch": float(kendall(q_root, catch)),
                        "tau_beam_catch": float(kendall(beam, catch)),
                        "cf_err": float(err.mean()), "v_true_range": float(v_true.max() - v_true.min()),
                        "v_pred_range": float(v_pred.max() - v_pred.min()), "q_range": float(q_root.max() - q_root.min()),
                        # value separation between catch and miss branches, real vs predicted successor
                        "v_true_sep": float(v_true[catch == 1].mean() - v_true[catch == 0].mean()),
                        "v_pred_sep": float(v_pred[catch == 1].mean() - v_pred[catch == 0].mean()),
                        "q_sep": float(q_root[catch == 1].mean() - q_root[catch == 0].mean()),
                    })
            prev_y = by
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
    m = lambda k: float(np.mean([r[k] for r in recs])) if recs else float("nan")
    keys = ("chance", "actor_ok", "critic_ok", "model_ok", "beam5_ok", "tau_model_critic", "tau_model_catch",
            "tau_critic_catch", "tau_actor_catch", "tau_beam_catch", "cf_err", "v_true_range", "v_pred_range",
            "q_range", "v_true_sep", "v_pred_sep", "q_sep")
    summary = {"K": cfg.replay.rollout_steps, "seed": seed, "states": n_states, "decisive": n, "episodes": ep,
               "elapsed_s": time.time() - t0, **{k: m(k) for k in keys}}
    print(f"{name} seed{seed} K={summary['K']:2d} decisive {n}/{n_states} ({ep} eps) chance {summary['chance']:.2f} | "
          f"catch-acc actor {summary['actor_ok']:.3f} critic {summary['critic_ok']:.3f} model {summary['model_ok']:.3f} "
          f"beam5 {summary['beam5_ok']:.3f} | tau-vs-catch actor {summary['tau_actor_catch']:+.2f} critic {summary['tau_critic_catch']:+.2f} "
          f"model {summary['tau_model_catch']:+.2f} beam5 {summary['tau_beam_catch']:+.2f} | tau(model,critic) {summary['tau_model_critic']:+.2f} "
          f"| sep(catch-miss) q {summary['q_sep']:+.3f} v_true {summary['v_true_sep']:+.3f} v_pred {summary['v_pred_sep']:+.3f} "
          f"| cf_err {summary['cf_err']:.3f} | {summary['elapsed_s']:.0f}s", flush=True)
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
            json.dump(out, open("/tmp/lead2_probe.json", "w"))
    print("LEAD2_DONE", flush=True)
