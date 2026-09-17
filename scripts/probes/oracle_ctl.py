"""Oracle-successor controller: the one-step planner with the learned dynamics replaced by the TRUE
one-step transition. At every decision: clone the emulator (incl. RNG), step each action for real,
encode each real successor with the arm's own online encoder, restore, and act on
    argmax_a [ r_a + gamma * (1 - terminated_a) * max_b Q(f(x'_a)) ].
Same protocol as the rest of the project: 10 episodes, reset seeds 10000-10009, eps = 0.01, cap 10000.
No learned dynamics, no training. Read-only on the checkpoint; writes only to /tmp.
"""
import json
import sys
import time

import numpy as np
import torch

from atari_jepa.checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from atari_jepa.envs import FrameStacker, clip_reward, make_env
from atari_jepa.losses import expected_reward

EPISODES, SEED_BASE, EPS, MAX_DEC = 10, 10000, 0.01, 10000


@torch.no_grad()
def run(name, seed, device):
    d = f"runs/{name}/seed{seed}"
    ckpt = load_checkpoint(f"{d}/checkpoint.pt", device)
    cfg, model = model_from_checkpoint(ckpt, device)
    model.eval()
    env = make_env(cfg.env)
    check_compatible(ckpt["env"], env.metadata())
    stacker = FrameStacker(cfg.env.history)
    A, g = env.num_actions, cfg.loss.gamma
    rng = np.random.default_rng(SEED_BASE)
    records = []
    disagree = n_dec = 0
    t0 = time.perf_counter()
    for i in range(EPISODES):
        s = SEED_BASE + i
        frame, info = env.reset(seed=s)
        history = stacker.reset(frame)
        raw, decisions = 0.0, 0
        terminated = truncated = capped = False
        while True:
            snap = env.clone_state()
            succ, r_a, term_a = [], np.zeros(A), np.zeros(A)
            for a in range(A):
                env.restore_state(snap)
                nxt, r, term, trunc, _ = env.step(a)
                succ.append(np.concatenate([history[1:], nxt[None]], 0))
                r_a[a] = float(clip_reward(r))
                term_a[a] = float(term)
            env.restore_state(snap)
            z = model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))
            q_action = int(model.q_head(z).argmax(-1))
            acts = torch.arange(A, device=device); zr = z.expand(A, *z.shape[1:]); z_pred = model.dynamics(zr, acts); r_hat = expected_reward(model.reward_head(zr, acts)).cpu().numpy(); c_hat = torch.sigmoid(model.continuation_head(zr, acts)).cpu().numpy()
            v = model.q_head(z_pred).max(-1).values.cpu().numpy()
            score = r_hat + g * c_hat * v  # CONTROL: learned-model successor, same clone/restore harness
            action = int(np.argmax(score))
            disagree += int(action != q_action)
            n_dec += 1
            if EPS > 0 and rng.random() < EPS:
                action = int(rng.integers(A))
            frame, reward, terminated, truncated, _ = env.step(action)
            history = stacker.push(frame)
            raw += reward
            decisions += 1
            if terminated or truncated:
                break
            if decisions >= MAX_DEC:
                capped = True
                break
        records.append({"episode": i, "reset_seed": s, "raw_return": raw, "decisions": decisions,
                        "terminated": terminated, "truncated": truncated, "capped": capped})
        print(f"{name} s{seed} ep{i} seed {s}: return {raw:+.0f} in {decisions} decisions "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    env.close()
    rets = np.array([r["raw_return"] for r in records])
    lens = np.array([r["decisions"] for r in records])
    return {"K": cfg.replay.rollout_steps, "seed": seed, "return_mean": float(rets.mean()),
            "return_std": float(rets.std()), "len_mean": float(lens.mean()),
            "disagreement_with_q": disagree / max(n_dec, 1), "wall_s": time.perf_counter() - t0,
            "records": records}


if __name__ == "__main__":
    name = sys.argv[1]
    seeds = [int(x) for x in sys.argv[2:]] or [0, 1, 2]
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"arm": name, "seeds": {}}
    for s in seeds:
        out["seeds"][str(s)] = run(name, s, device)
        json.dump(out, open(f"/tmp/oraclectl_{name}.json", "w"), indent=1)
    m = np.array([v["return_mean"] for v in out["seeds"].values()])
    L = np.array([v["len_mean"] for v in out["seeds"].values()])
    print(f"ORACLECTL {name} K={out['seeds'][str(seeds[0])]['K']} return {m.mean():+.2f} +- {m.std():.2f} "
          f"per-seed {' / '.join(f'{x:.1f}' for x in m)} | len {L.mean():.0f} | "
          f"disagree {np.mean([v['disagreement_with_q'] for v in out['seeds'].values()]):.2f}", flush=True)
    print("ORACLE_DONE", flush=True)
