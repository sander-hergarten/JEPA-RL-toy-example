"""Closed-loop paddle kinematics of the Q policy vs the H=1 planner (read-only, frozen checkpoints).

Per decision: action, ball (x,y), paddle x, per-decision ball vy, reward. Derived per controller:
catch rate at paddle contacts, paddle-ball offset at catches vs misses, tracking-error profile vs
time-to-contact during descending legs, command-reversal rate and closing speed during the approach.
"""
import json
import sys

import numpy as np
import torch

from atari_jepa.checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from atari_jepa.envs import FrameStacker, make_env
from atari_jepa.planning import one_step_scores

EPISODES, EPS, MAXDEC = 3, 0.01, 3000
PADDLE_Y = 179
RIGHT, LEFT = 2, 3


@torch.no_grad()
def play(model, cfg, env, stacker, device, controller, seed_base):
    A, g = env.num_actions, cfg.loss.gamma
    rng = np.random.default_rng(seed_base)
    logs = []
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=seed_base + ep)
        history = stacker.reset(obs)
        rows = []
        for t in range(MAXDEC):
            ram = env.state_vector()
            bx, by, px = int(ram[99]), int(ram[101]), int(ram[72])
            z = model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))
            if controller == "q":
                action = int(model.q_head(z).argmax(-1))
            else:
                action = int(one_step_scores(model, z, g).argmax(-1))
            if rng.random() < EPS:
                action = int(rng.integers(A))
            obs, r, term, trunc, _ = env.step(action)
            history = stacker.push(obs)
            rows.append((t, action, bx, by, px, float(r)))
            if term or trunc:
                break
        logs.append(rows)
    return logs


def analyse(logs):
    catches, misses = [], []
    reversals, approach_steps, closing = 0, 0, []
    offset_by_ttc = {k: [] for k in (10, 8, 6, 4, 2, 0)}
    noop, total = 0, 0
    for rows in logs:
        prev_by, prev_dir = None, None
        for i, (t, a, bx, by, px, r) in enumerate(rows):
            total += 1
            noop += int(a in (0, 1))
            vy = None if (prev_by is None or by == 0 or prev_by == 0) else by - prev_by
            if vy is not None and vy > 0:
                ttc = (PADDLE_Y - by) / vy
                off = bx - px
                for k in offset_by_ttc:
                    if abs(ttc - k) <= 0.75:
                        offset_by_ttc[k].append(abs(off - 4))  # ~4 = centre of the catch window (measured)
                if ttc > 1.5:
                    d = 1 if a == RIGHT else (-1 if a == LEFT else 0)
                    if prev_dir is not None and d != 0 and prev_dir != 0 and d != prev_dir:
                        reversals += 1
                    approach_steps += 1
                    if i + 1 < len(rows):
                        nb, npx = rows[i + 1][2], rows[i + 1][4]
                        closing.append(abs(off - 4) - abs((nb - npx) - 4))
                    prev_dir = d if d != 0 else prev_dir
            # contact outcome: descending ball reaching the paddle row
            if vy is not None and vy > 0 and by >= PADDLE_Y - vy and by < PADDLE_Y + vy:
                # look ahead a few decisions: bounce (by decreases) = catch; ball lost (by==0) = miss
                fut = [rows[j][3] for j in range(i + 1, min(i + 6, len(rows)))]
                if fut:
                    if any(f == 0 for f in fut):
                        misses.append(bx - px)
                    elif fut[-1] < by:
                        catches.append(bx - px)
            prev_by = by
    n_c, n_m = len(catches), len(misses)
    return {
        "decisions": total, "episodes": len(logs),
        "catches": n_c, "misses": n_m, "catch_rate": n_c / max(n_c + n_m, 1),
        "catch_offset_mean": float(np.mean(catches)) if catches else None,
        "catch_offset_std": float(np.std(catches)) if catches else None,
        "miss_offset_mean_abs": float(np.mean(np.abs(np.array(misses) - 4))) if misses else None,
        "miss_offsets": [int(m) for m in misses],
        "reversal_rate_approach": reversals / max(approach_steps, 1),
        "closing_per_decision": float(np.mean(closing)) if closing else None,
        "noop_fraction": noop / max(total, 1),
        "abs_offset_by_ttc": {k: (float(np.mean(v)) if v else None) for k, v in offset_by_ttc.items()},
        "n_by_ttc": {k: len(v) for k, v in offset_by_ttc.items()},
    }


if __name__ == "__main__":
    device = torch.device("cuda")
    result = {}
    for name in sys.argv[1:]:
        result[name] = {}
        for s in range(3):
            d = f"runs/{name}/seed{s}"
            try:
                ckpt = load_checkpoint(f"{d}/checkpoint.pt", device)
            except FileNotFoundError:
                continue
            cfg, model = model_from_checkpoint(ckpt, device)
            model.eval()
            env = make_env(cfg.env)
            check_compatible(ckpt["env"], env.metadata())
            stacker = FrameStacker(cfg.env.history)
            result[name][s] = {}
            for ctl in ("q", "lookahead"):
                logs = play(model, cfg, env, stacker, device, ctl, 10_000)
                res = analyse(logs)
                res["K"] = cfg.replay.rollout_steps
                result[name][s][ctl] = res
                print(f"{name} seed{s} K={res['K']} {ctl:9s} dec={res['decisions']:5d} catches={res['catches']:3d} misses={res['misses']:3d} "
                      f"catch_rate={res['catch_rate']:.2f} catch_off={res['catch_offset_mean']} miss|off-4|={res['miss_offset_mean_abs']} "
                      f"reversal={res['reversal_rate_approach']:.3f} closing={res['closing_per_decision']} noop={res['noop_fraction']:.2f} "
                      f"|off| by ttc={ {k: (round(v,1) if v is not None else None) for k,v in res['abs_offset_by_ttc'].items()} } misses={res['miss_offsets']}", flush=True)
            env.close()
    print("JSON_BEGIN")
    print(json.dumps(result))
    print("JSON_END")
