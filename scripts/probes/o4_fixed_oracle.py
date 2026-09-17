"""Corrected-harness one-step planners. READ-ONLY: prints only, writes nothing.

oracle_eval2.py's docstring: ALE cloneSystemState restores RAM/CPU/RNG but NOT the last executed
action, which sticky actions repeat with p=0.25. Every earlier branching harness therefore ran each
counterfactual branch with the sticky context of the *previous branch* (asc: 0,1,2,3), not the real
previous action. Fix used here, before EVERY branch and before the real step:
    restore(snap); step(a_prev); restore(snap)
The intermediate step's outcome is discarded; only its side effect (last action := a_prev) persists,
and restore(snap) resets RAM/CPU/RNG. So every branch and the real step share the true sticky context
and the same RNG draw. Transparency check: mode=model_fixed must reproduce eval_lookahead_h1 exactly
(K=5: +40.43, per-seed 40.0 / 43.3 / 38.0) and report fidelity 1.0.
"""
import sys
import time

import numpy as np
import torch

from atari_jepa.checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from atari_jepa.envs import FrameStacker, clip_reward, find_action, make_env
from atari_jepa.losses import expected_reward

EPISODES, SEED_BASE, EPS, MAX_DEC = 10, 10000, 0.01, 10000


@torch.no_grad()
def run(name, seed, device, mode):
    d = f"runs/{name}/seed{seed}"
    ckpt = load_checkpoint(f"{d}/checkpoint.pt", device)
    cfg, model = model_from_checkpoint(ckpt, device)
    model.eval()
    env = make_env(cfg.env)
    check_compatible(ckpt["env"], env.metadata())
    stacker = FrameStacker(cfg.env.history)
    A, g = env.num_actions, cfg.loss.gamma
    fire = find_action(env.action_meanings, "FIRE")
    noop = find_action(env.action_meanings, "NOOP")
    rng = np.random.default_rng(SEED_BASE)
    rets, lens = [], []
    disagree = n_dec = fid_hits = fid_n = 0
    t0 = time.perf_counter()

    def set_context(snap, a_prev):
        env.restore_state(snap)
        env.step(a_prev)          # only side effect wanted: ALE last action := a_prev
        env.restore_state(snap)   # RAM/CPU/RNG back; last action stays a_prev

    for i in range(EPISODES):
        s = SEED_BASE + i
        frame, info = env.reset(seed=s)
        history = stacker.reset(frame)
        a_prev = fire if cfg.env.fire_on_reset else noop
        raw, decisions = 0.0, 0
        terminated = truncated = False
        while True:
            z = model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))
            q_action = int(model.q_head(z).argmax(-1))
            snap = env.clone_state()
            succ, r_a, term_a = [None] * A, np.zeros(A), np.zeros(A)
            for a in range(A):
                set_context(snap, a_prev)
                nxt, r, term, trunc, _ = env.step(a)
                succ[a] = nxt
                r_a[a] = float(clip_reward(r))
                term_a[a] = float(term)
            set_context(snap, a_prev)  # real step below gets the true context too
            if mode == "oracle_fixed":
                stacks = np.stack([np.concatenate([history[1:], f[None]], 0) for f in succ])
                v = model.q_head(model.encoder(torch.from_numpy(stacks).to(device))).max(-1).values.cpu().numpy()
                score = r_a + g * (1.0 - term_a) * v
            else:
                acts = torch.arange(A, device=device)
                zr = z.expand(A, *z.shape[1:])
                z_pred = model.dynamics(zr, acts)
                r_hat = expected_reward(model.reward_head(zr, acts)).cpu().numpy()
                c_hat = torch.sigmoid(model.continuation_head(zr, acts)).cpu().numpy()
                v = model.q_head(z_pred).max(-1).values.cpu().numpy()
                score = r_hat + g * c_hat * v
            action = int(np.argmax(score))
            disagree += int(action != q_action)
            n_dec += 1
            if EPS > 0 and rng.random() < EPS:
                action = int(rng.integers(A))
            frame, reward, terminated, truncated, sinfo = env.step(action)
            fid_hits += int(np.array_equal(frame, succ[action]))
            fid_n += 1
            a_prev = fire if sinfo.get("fire_frames", 0) else action
            history = stacker.push(frame)
            raw += reward
            decisions += 1
            if terminated or truncated or decisions >= MAX_DEC:
                break
        rets.append(raw)
        lens.append(decisions)
        print(f"{mode} {name} s{seed} ep{i} seed {s}: return {raw:+.0f} in {decisions} decisions "
              f"fid {fid_hits}/{fid_n} ({time.perf_counter() - t0:.0f}s)", flush=True)
    env.close()
    rets, lens = np.array(rets), np.array(lens)
    return dict(K=cfg.replay.rollout_steps, seed=seed, ret=float(rets.mean()), sd=float(rets.std()),
                len=float(lens.mean()), fid=fid_hits / max(fid_n, 1), dis=disagree / max(n_dec, 1))


if __name__ == "__main__":
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plan = [("model_fixed", "breakout_se_nstep_500k", [0, 1, 2]),
            ("oracle_fixed", "breakout_se_nstep_500k", [0, 1, 2]),
            ("oracle_fixed", "breakout_se_k30_500k", [0, 1, 2]),
            ("model_fixed", "breakout_se_k30_500k", [0])]
    for mode, name, seeds in plan:
        rows = [run(name, s, device, mode) for s in seeds]
        m = np.array([r["ret"] for r in rows])
        print(f"RESULT {mode} {name} K={rows[0]['K']} return {m.mean():+.2f} +- {m.std():.2f} "
              f"per-seed {' / '.join(f'{x:.1f}' for x in m)} | len {np.mean([r['len'] for r in rows]):.0f} | "
              f"fidelity {np.mean([r['fid'] for r in rows]):.3f} | disagree {np.mean([r['dis'] for r in rows]):.2f}",
              flush=True)
    print("O4_DONE", flush=True)
