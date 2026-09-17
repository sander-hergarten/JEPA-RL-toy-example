"""Directional correctness of each controller during the ball's approach, by lead time (eval-only).

Run the arm's Q controller, H=1 planner and H=5 beam planner (eps 0.01, the eval protocol) for a few
episodes each, logging per decision the chosen action, the greedy-Q action, and RAM paddle_x (72),
ball_x (99), ball_y (101) and lives. Post-process each ball descent: the arrival is the decision at
which ball_y peaks (catch: it then decreases; miss: a life is lost). For every decision in the descent,
lead = arrival - t and landing = ball_x at arrival. A move is CORRECT when it takes the paddle toward
the landing point (NOOP/FIRE correct when the paddle is already within the dead zone). Report the
correctness rate by lead bin, overall, and on the decisions where the planner overrides Q -- who is
right, the planner or Q?
"""
import json
import sys
import time

import numpy as np
import torch

from atari_jepa.checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from atari_jepa.envs import FrameStacker, make_env
from atari_jepa.planning import LookaheadController, QController

EPISODES, EPS, MAX_DEC = 3, 0.01, 2500
BINS = [(1, 2), (3, 5), (6, 9), (10, 15), (16, 60)]


def play(model, cfg, device, ctrl, seed_base, meanings):
    env = make_env(cfg.env)
    stacker = FrameStacker(cfg.env.history)
    rows = []
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=seed_base + ep)
        hist = stacker.reset(obs)
        for t in range(MAX_DEC):
            ram = env.state_vector()
            lives = env._ale.lives()
            a, info = ctrl.act(hist)
            rows.append((ep, t, a, info["q_action"], int(ram[72]), int(ram[99]), int(ram[101]), lives))
            obs, r, term, trunc, _ = env.step(a)
            hist = stacker.push(obs)
            if term or trunc:
                break
    env.close()
    return np.array(rows, dtype=np.int64)


def descents(rows):
    """Yield (indices of the descent, arrival index, miss flag) for each ball approach."""
    ep, t, a, qa, px, bx, by, lives = rows.T
    out = []
    n = len(rows)
    i = 0
    while i < n - 1:
        # a descent starts when ball_y starts increasing
        if ep[i + 1] == ep[i] and by[i + 1] > by[i]:
            j = i + 1
            while j < n - 1 and ep[j + 1] == ep[j] and by[j + 1] >= by[j] and lives[j + 1] == lives[j]:
                j += 1
            miss = j < n - 1 and ep[j + 1] == ep[j] and lives[j + 1] < lives[j]
            if j - i >= 3:
                out.append((list(range(i, j + 1)), j, bool(miss)))
            i = j + 1
        else:
            i += 1
    return out


def analyse(rows, meanings, sign_right, dead):
    ep, t, a, qa, px, bx, by, lives = rows.T
    move = np.zeros(len(meanings), dtype=np.int64)
    for k, m in enumerate(meanings):
        if "RIGHT" in m:
            move[k] = sign_right
        elif "LEFT" in m:
            move[k] = -sign_right
    recs = []
    for idx, arr, miss in descents(rows):
        landing = bx[arr]
        for i in idx[:-1]:
            lead = arr - i
            gap = landing - px[i]
            need = 0 if abs(gap) <= dead else int(np.sign(gap))
            ok = lambda act: (move[act] == need) if need != 0 else (move[act] == 0)
            recs.append((lead, int(ok(a[i])), int(ok(qa[i])), int(a[i] != qa[i]), int(miss), abs(gap)))
    return np.array(recs, dtype=np.int64)


def calibrate(rows):
    """Dead zone and sign convention from the data: paddle-ball offset at catches vs misses; d(paddle)/RIGHT."""
    ep, t, a, qa, px, bx, by, lives = rows.T
    catch_off, miss_off = [], []
    for idx, arr, miss in descents(rows):
        off = bx[arr] - px[arr]
        (miss_off if miss else catch_off).append(off)
    return np.array(catch_off), np.array(miss_off)


def run(name, seed, device):
    d = f"runs/{name}/seed{seed}"
    ckpt = load_checkpoint(f"{d}/checkpoint.pt", device)
    cfg, model = model_from_checkpoint(ckpt, device)
    model.eval()
    env = make_env(cfg.env)
    check_compatible(ckpt["env"], env.metadata())
    meanings = env.action_meanings
    env.close()
    g = cfg.loss.gamma
    ctrls = {
        "q": QController(model, device, EPS, seed),
        "h1": LookaheadController(model, device, EPS, seed, g, horizon=1, beam_width=16),
        "h5": LookaheadController(model, device, EPS, seed, g, horizon=5, beam_width=16),
    }
    out = {"K": cfg.replay.rollout_steps, "seed": seed}
    traces = {}
    for cname, ctrl in ctrls.items():
        t0 = time.time()
        rows = play(model, cfg, device, ctrl, 10000, meanings)
        traces[cname] = rows
        out[cname] = {"decisions": int(len(rows)), "episodes": EPISODES, "secs": time.time() - t0}
    return cfg, meanings, out, traces


if __name__ == "__main__":
    device = torch.device("cuda")
    all_rows = {}
    results = {}
    for name in sys.argv[1:]:
        for seed in range(3):
            try:
                cfg, meanings, out, traces = run(name, seed, device)
            except FileNotFoundError as exc:
                print("skip", name, seed, exc, flush=True)
                continue
            all_rows[(name, seed)] = (meanings, out, traces)
            print(f"played {name} seed{seed}: " + " ".join(f"{c}={out[c]['decisions']}dec/{out[c]['secs']:.0f}s" for c in ("q", "h1", "h5")), flush=True)
    # calibration pooled over everything: sign of RIGHT, dead zone from catch offsets
    catch_all, miss_all, dpx = [], [], []
    for (name, seed), (meanings, out, traces) in all_rows.items():
        for cname, rows in traces.items():
            c, m = calibrate(rows)
            catch_all += list(c); miss_all += list(m)
            ep, t, a, qa, px, bx, by, lives = rows.T
            same = ep[1:] == ep[:-1]
            right = np.array(["RIGHT" in meanings[k] for k in a[:-1]])
            dpx += list((px[1:] - px[:-1])[same & right])
    catch_all, miss_all, dpx = np.array(catch_all), np.array(miss_all), np.array(dpx)
    sign_right = int(np.sign(np.median(dpx))) or 1
    lo, hi = np.percentile(catch_all, [5, 95])
    center = float(np.median(catch_all))
    dead = float(max(np.percentile(np.abs(catch_all - center), 75), 4))
    print(f"calibration: RIGHT moves paddle_x by median {np.median(dpx):+.1f} (sign {sign_right}); catch offset ball_x-paddle_x "
          f"median {center:.1f} p5..p95 [{lo:.0f},{hi:.0f}] n={len(catch_all)}; miss offset median {np.median(miss_all):.1f} "
          f"|miss offset| median {np.median(np.abs(miss_all - center)):.1f} n={len(miss_all)}; dead zone +-{dead:.1f}", flush=True)
    # analysis
    summary = {}
    for (name, seed), (meanings, out, traces) in all_rows.items():
        K = out["K"]
        for cname, rows in traces.items():
            ep, t, a, qa, px, bx, by, lives = rows.T
            px_c = px + int(round(center))  # shift paddle by the catch-offset so gap = landing - paddle centre
            rows2 = rows.copy(); rows2[:, 4] = px_c
            recs = analyse(rows2, meanings, sign_right, dead)
            if len(recs) == 0:
                continue
            lead, ok_c, ok_q, over, miss, gap = recs.T
            key = (K, cname)
            summary.setdefault(key, []).append(recs)
    print("\nK  ctrl | n_dec  overall: ctrl  q   | override-frac  on overrides: ctrl  q  | by lead bin (ctrl/q, n)", flush=True)
    for (K, cname), lst in sorted(summary.items()):
        recs = np.concatenate(lst)
        lead, ok_c, ok_q, over, miss, gap = recs.T
        ov = over == 1
        line = (f"K={K:2d} {cname:2s} | {len(recs):5d}  {ok_c.mean():.3f} {ok_q.mean():.3f} | {ov.mean():.2f}  "
                f"{ok_c[ov].mean() if ov.any() else float('nan'):.3f} {ok_q[ov].mean() if ov.any() else float('nan'):.3f} |")
        for lo_, hi_ in BINS:
            m = (lead >= lo_) & (lead <= hi_)
            line += f" [{lo_}-{hi_}] {ok_c[m].mean():.2f}/{ok_q[m].mean():.2f} ({m.sum()})"
        print(line, flush=True)
    # per-seed overall for spread
    print("\nper-seed overall correctness (ctrl): ", flush=True)
    for (K, cname), lst in sorted(summary.items()):
        print(f"K={K:2d} {cname:2s}: " + " ".join(f"{r[:,1].mean():.3f}" for r in lst), flush=True)
    # miss statistics
    print("\nmisses per 1000 decisions and descents per controller:", flush=True)
    for (name, seed), (meanings, out, traces) in all_rows.items():
        for cname, rows in traces.items():
            ds = descents(rows)
            nm = sum(m for _, _, m in ds)
            print(f"{name} seed{seed} {cname}: descents {len(ds)} misses {nm} decisions {len(rows)}", flush=True)
    print("APPROACH_DONE", flush=True)
