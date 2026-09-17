"""Harness-fidelity diagnostic + one-step planners (true transition vs learned model).

modes:
  model_noclone : the learned-model one-step planner with NO cloning at all (must equal eval_lookahead_h1)
  model_clone   : clone + restore only, no branch steps, then learned-model planner
  model_branch  : clone, 4 real branch steps, restore, learned-model planner (order flag)
  oracle        : clone, 4 real branch steps, restore, act on r_a + g(1-term_a) max_b Q(f(x'_a))
order: 'asc' = 0..A-1 ; 'prev' = end on the previously executed real action
fidelity: fraction of decisions where the real next frame equals the branch frame of the executed action.
Read-only on the checkpoint; writes only to /tmp.
"""
import json
import sys
import time

import numpy as np
import torch

from atari_jepa.checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from atari_jepa.envs import FrameStacker, clip_reward, find_action, make_env
from atari_jepa.losses import expected_reward

EPISODES, SEED_BASE, EPS, MAX_DEC = 10, 10000, 0.01, 10000


@torch.no_grad()
def run(name, seed, device, mode, order):
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
    records = []
    disagree = n_dec = fid_hits = fid_n = 0
    t0 = time.perf_counter()
    for i in range(EPISODES):
        s = SEED_BASE + i
        frame, info = env.reset(seed=s)
        history = stacker.reset(frame)
        a_prev = fire if cfg.env.fire_on_reset else noop
        raw, decisions = 0.0, 0
        terminated = truncated = capped = False
        while True:
            z = model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))
            q_action = int(model.q_head(z).argmax(-1))
            succ = None
            if mode != "model_noclone":
                snap = env.clone_state()
                if mode in ("model_branch", "oracle"):
                    seq = list(range(A)) if order == "asc" else [a for a in range(A) if a != a_prev] + [a_prev]
                    succ, r_a, term_a = [None] * A, np.zeros(A), np.zeros(A)
                    for a in seq:
                        env.restore_state(snap)
                        nxt, r, term, trunc, _ = env.step(a)
                        succ[a] = nxt
                        r_a[a] = float(clip_reward(r))
                        term_a[a] = float(term)
                env.restore_state(snap)
            if mode == "oracle":
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
            if succ is not None:
                fid_hits += int(np.array_equal(frame, succ[action]))
                fid_n += 1
            a_prev = fire if sinfo.get("fire_frames", 0) else action
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
        print(f"{mode}/{order} {name} s{seed} ep{i} seed {s}: return {raw:+.0f} in {decisions} decisions "
              f"fid {fid_hits}/{fid_n} ({time.perf_counter() - t0:.0f}s)", flush=True)
    env.close()
    rets = np.array([r["raw_return"] for r in records])
    lens = np.array([r["decisions"] for r in records])
    return {"K": cfg.replay.rollout_steps, "seed": seed, "return_mean": float(rets.mean()),
            "return_std": float(rets.std()), "len_mean": float(lens.mean()),
            "disagreement_with_q": disagree / max(n_dec, 1), "fidelity": fid_hits / max(fid_n, 1),
            "wall_s": time.perf_counter() - t0, "records": records}


if __name__ == "__main__":
    mode, order, name = sys.argv[1], sys.argv[2], sys.argv[3]
    seeds = [int(x) for x in sys.argv[4:]] or [0, 1, 2]
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"arm": name, "mode": mode, "order": order, "seeds": {}}
    for s in seeds:
        out["seeds"][str(s)] = run(name, s, device, mode, order)
        json.dump(out, open(f"/tmp/o3_{mode}_{order}_{name}.json", "w"), indent=1)
    m = np.array([v["return_mean"] for v in out["seeds"].values()])
    L = np.array([v["len_mean"] for v in out["seeds"].values()])
    F = np.array([v["fidelity"] for v in out["seeds"].values()])
    print(f"RESULT {mode}/{order} {name} K={out['seeds'][str(seeds[0])]['K']} return {m.mean():+.2f} +- {m.std():.2f} "
          f"per-seed {' / '.join(f'{x:.1f}' for x in m)} | len {L.mean():.0f} | fidelity {F.mean():.3f} | "
          f"disagree {np.mean([v['disagreement_with_q'] for v in out['seeds'].values()]):.2f}", flush=True)
    print("DONE", flush=True)
