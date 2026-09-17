"""Where in the approach does a controller lose the catch? Per-descent geometry from real play (eval-only).

Same play as approach_probe.py (Q, H=1, H=5; eps 0.01; reset seeds 10000..), traces saved. For each ball
descent: gap(tau) = landing_x - (paddle_x(tau) + centre) at lead tau before arrival. Reports per (K, ctrl):
  * mean |gap| and P(|gap| <= window) by lead (is the paddle already under the landing point early?)
  * miss geometry: |offset| at arrival for misses, near (<= 1 paddle move) vs far
  * idle fraction (NOOP/FIRE) and direction-flip fraction by lead (dithering)
  * for misses, the largest lead at which the race was still winnable (|gap| <= speed*lead + window)
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
CENTRE, WINDOW, SPEED = 9, 12, 11  # from approach_probe calibration (RAM units)
LEADS = [1, 2, 3, 4, 5, 6, 8, 10, 12]


def play(cfg, ctrl):
    env = make_env(cfg.env)
    stacker = FrameStacker(cfg.env.history)
    rows = []
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=10000 + ep)
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
    ep, t, a, qa, px, bx, by, lives = rows.T
    out, n, i = [], len(rows), 0
    while i < n - 1:
        if ep[i + 1] == ep[i] and by[i + 1] > by[i]:
            j = i + 1
            while j < n - 1 and ep[j + 1] == ep[j] and by[j + 1] >= by[j] and lives[j + 1] == lives[j]:
                j += 1
            miss = j < n - 1 and ep[j + 1] == ep[j] and lives[j + 1] < lives[j]
            if j - i >= 3:
                out.append((i, j, bool(miss)))
            i = j + 1
        else:
            i += 1
    return out


def analyse(rows, meanings):
    ep, t, a, qa, px, bx, by, lives = rows.T
    move = np.zeros(len(meanings), dtype=np.int64)
    for k, m in enumerate(meanings):
        move[k] = 1 if "RIGHT" in m else (-1 if "LEFT" in m else 0)
    gap_by_lead = {L: [] for L in LEADS}
    idle_by_lead = {L: [] for L in LEADS}
    flip_by_lead = {L: [] for L in LEADS}
    toward_by_lead = {L: [] for L in LEADS}
    miss_off, catch_off, winnable_lead, n_desc, n_miss = [], [], [], 0, 0
    for i, j, miss in descents(rows):
        landing = bx[j]
        n_desc += 1
        n_miss += int(miss)
        off = landing - (px[j] + CENTRE)
        (miss_off if miss else catch_off).append(abs(off))
        last_win = None
        for k in range(i, j):
            lead = j - k
            gap = landing - (px[k] + CENTRE)
            if abs(gap) <= SPEED * lead + WINDOW:
                last_win = lead if last_win is None else min(last_win, lead)
            if lead in gap_by_lead:
                gap_by_lead[lead].append(abs(gap))
                idle_by_lead[lead].append(float(move[a[k]] == 0))
                toward_by_lead[lead].append(float(move[a[k]] == np.sign(gap)) if abs(gap) > WINDOW else float("nan"))
                if k > i and move[a[k]] != 0 and move[a[k - 1]] != 0:
                    flip_by_lead[lead].append(float(move[a[k]] != move[a[k - 1]]))
        if miss:
            # the largest lead from which the race was already lost for good
            lost = None
            for k in range(j - 1, i - 1, -1):
                lead = j - k
                gap = landing - (px[k] + CENTRE)
                if abs(gap) > SPEED * lead + WINDOW:
                    lost = lead
                else:
                    break
            winnable_lead.append(-1 if lost is None else lost)
    nm = lambda x: float(np.nanmean(x)) if len(x) else float("nan")
    return {
        "descents": n_desc, "misses": n_miss,
        "gap": {L: nm(v) for L, v in gap_by_lead.items()},
        "in_window": {L: nm([g <= WINDOW for g in v]) for L, v in gap_by_lead.items()},
        "idle": {L: nm(v) for L, v in idle_by_lead.items()},
        "toward": {L: nm(v) for L, v in toward_by_lead.items()},
        "flip": {L: nm(v) for L, v in flip_by_lead.items()},
        "miss_off_median": float(np.median(miss_off)) if miss_off else float("nan"),
        "miss_near_frac": nm([o <= WINDOW + SPEED for o in miss_off]),
        "catch_off_median": float(np.median(catch_off)) if catch_off else float("nan"),
        "miss_lost_lead_hist": np.bincount(np.array([w for w in winnable_lead if w >= 0], dtype=int), minlength=12)[:12].tolist(),
        "miss_lost_lead_mean": nm([w for w in winnable_lead if w >= 0]),
    }


if __name__ == "__main__":
    device = torch.device("cuda")
    traces, results = {}, {}
    for name in sys.argv[1:]:
        for seed in range(3):
            d = f"runs/{name}/seed{seed}"
            try:
                ckpt = load_checkpoint(f"{d}/checkpoint.pt", device)
            except FileNotFoundError:
                continue
            cfg, model = model_from_checkpoint(ckpt, device)
            model.eval()
            env = make_env(cfg.env)
            check_compatible(ckpt["env"], env.metadata())
            meanings = env.action_meanings
            env.close()
            g, K = cfg.loss.gamma, cfg.replay.rollout_steps
            for cname, ctrl in (("q", QController(model, device, EPS, seed)),
                                ("h1", LookaheadController(model, device, EPS, seed, g, horizon=1, beam_width=16)),
                                ("h5", LookaheadController(model, device, EPS, seed, g, horizon=5, beam_width=16))):
                rows = play(cfg, ctrl)
                traces[f"{name}_s{seed}_{cname}"] = rows
                results.setdefault((K, cname), []).append(analyse(rows, meanings))
            print(f"played {name} seed{seed}", flush=True)
    np.savez_compressed("/tmp/approach_traces.npz", **traces)
    agg = lambda lst, key, L=None: float(np.nanmean([r[key][L] if L is not None else r[key] for r in lst]))
    print("\nmean |gap| (landing - paddle centre) by lead; window=+-12, paddle speed 11/decision", flush=True)
    for (K, c), lst in sorted(results.items()):
        print(f"K={K:2d} {c:2s} desc/miss {sum(r['descents'] for r in lst)}/{sum(r['misses'] for r in lst)} | " +
              " ".join(f"L{L}:{agg(lst,'gap',L):5.1f}" for L in LEADS), flush=True)
    print("\nP(paddle already within window) by lead", flush=True)
    for (K, c), lst in sorted(results.items()):
        print(f"K={K:2d} {c:2s} | " + " ".join(f"L{L}:{agg(lst,'in_window',L):.2f}" for L in LEADS), flush=True)
    print("\nP(move toward landing | outside window) by lead", flush=True)
    for (K, c), lst in sorted(results.items()):
        print(f"K={K:2d} {c:2s} | " + " ".join(f"L{L}:{agg(lst,'toward',L):.2f}" for L in LEADS), flush=True)
    print("\nidle fraction (NOOP/FIRE) by lead", flush=True)
    for (K, c), lst in sorted(results.items()):
        print(f"K={K:2d} {c:2s} | " + " ".join(f"L{L}:{agg(lst,'idle',L):.2f}" for L in LEADS), flush=True)
    print("\ndirection-flip fraction (consecutive moves) by lead", flush=True)
    for (K, c), lst in sorted(results.items()):
        print(f"K={K:2d} {c:2s} | " + " ".join(f"L{L}:{agg(lst,'flip',L):.2f}" for L in LEADS), flush=True)
    print("\nmiss geometry: median |offset| at arrival, near-miss fraction (<= 23), mean lead at which the race was lost, hist[lead 0..11]", flush=True)
    for (K, c), lst in sorted(results.items()):
        hist = np.sum([r["miss_lost_lead_hist"] for r in lst], axis=0).tolist()
        print(f"K={K:2d} {c:2s} | miss |off| {agg(lst,'miss_off_median'):5.1f} near {agg(lst,'miss_near_frac'):.2f} "
              f"lost-lead {agg(lst,'miss_lost_lead_mean'):.2f} hist {hist} | catch |off| {agg(lst,'catch_off_median'):.1f}", flush=True)
    print("APPROACH2_DONE", flush=True)
