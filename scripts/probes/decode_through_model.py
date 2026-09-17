"""State decodability THROUGH the one-step model (read-only probe on frozen checkpoints).

Ridge probes (on PCA features) are fit from REAL online latents to RAM (ball_x, ball_y, ball vx, vy,
paddle_x) on the training half of held-out episodes. On the test half, from cloned counterfactual
branches, the same probe is applied to (a) the real successor latent f(x'_a) against the successor's
RAM, (b) the model successor g(z,a) against the SAME successor RAM, and (c) the root latent z against
the successor RAM (the "stale copy" floor). If the long-K model renders the paddle but not the ball's
advance, (b) ~ (a) for paddle_x while (b) falls toward (c) for ball_y / vy.
"""
import json
import sys

import numpy as np
import torch

from atari_jepa.checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from atari_jepa.envs import FrameStacker, make_env

EPS, TARGET, EVERY, EPISODES = 0.05, 900, 2, 12
NAMES = ["ball_x", "ball_y", "vx", "vy", "paddle_x"]


@torch.no_grad()
def collect(name, seed):
    device = torch.device("cuda")
    d = f"runs/{name}/seed{seed}"
    ckpt = load_checkpoint(f"{d}/checkpoint.pt", device)
    cfg, model = model_from_checkpoint(ckpt, device)
    model.eval()
    env = make_env(cfg.env)
    check_compatible(ckpt["env"], env.metadata())
    stacker = FrameStacker(cfg.env.history)
    A = env.num_actions
    acts = torch.arange(A, device=device)
    rng = np.random.default_rng(61_000 + seed)
    recs = []
    for ep in range(EPISODES):
        if len(recs) >= TARGET:
            break
        obs, _ = env.reset(seed=61_000 + seed * 10 + ep)
        history = stacker.reset(obs)
        prev = None
        for t in range(2500):
            ram = env.state_vector()
            bx, by, px = int(ram[99]), int(ram[101]), int(ram[72])
            ok = prev is not None and by > 0 and prev[1] > 0
            if t % EVERY == 0 and len(recs) < TARGET and ok:
                vx, vy = bx - prev[0], by - prev[1]
                snap = env.clone_state()
                succ, sram = [], []
                for a in range(A):
                    env.restore_state(snap)
                    nxt, _, _, _, _ = env.step(a)
                    succ.append(np.concatenate([history[1:], nxt[None]], 0))
                    r2 = env.state_vector()
                    sbx, sby, spx = int(r2[99]), int(r2[101]), int(r2[72])
                    sram.append([sbx, sby, sbx - bx, sby - by, spx])
                env.restore_state(snap)
                z = model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))
                z_true = model.encoder(torch.from_numpy(np.stack(succ)).to(device))
                z_pred = model.dynamics(z.expand(A, *z.shape[1:]), acts)
                recs.append({
                    "ep": ep, "root_ram": [bx, by, vx, vy, px], "succ_ram": sram,
                    "z": z.flatten(1).cpu().numpy().astype(np.float32),
                    "z_true": z_true.flatten(1).cpu().numpy().astype(np.float32),
                    "z_pred": z_pred.flatten(1).cpu().numpy().astype(np.float32),
                })
            action = int(model.q_head(model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))).argmax(-1))
            if rng.random() < EPS:
                action = int(rng.integers(A))
            obs, _, term, trunc, _ = env.step(action)
            history = stacker.push(obs)
            prev = (bx, by)
            if term or trunc:
                break
    env.close()
    return cfg.replay.rollout_steps, recs


def r2(y, yhat):
    ss = ((y - y.mean(0)) ** 2).sum(0)
    return 1.0 - ((y - yhat) ** 2).sum(0) / np.maximum(ss, 1e-9)


def analyse(recs, n_comp=256, lam=1.0):
    train = [r for r in recs if r["ep"] % 2 == 0]
    test = [r for r in recs if r["ep"] % 2 == 1]
    # training data: real latents (roots + real successors) with their own RAM
    X = np.concatenate([r["z"] for r in train] + [r["z_true"] for r in train], 0)
    Y = np.concatenate([np.array([r["root_ram"]], float) for r in train]
                       + [np.array(r["succ_ram"], float) for r in train], 0)
    mu, ymu = X.mean(0), Y.mean(0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:n_comp].T  # [D, n_comp]
    F = Xc @ P
    W = np.linalg.solve(F.T @ F + lam * np.eye(n_comp), F.T @ (Y - ymu))

    def predict(Z):
        return ((Z - mu) @ P) @ W + ymu

    # test sets: successor RAM as target for all three latent sources
    Ys = np.concatenate([np.array(r["succ_ram"], float) for r in test], 0)
    Zt = np.concatenate([r["z_true"] for r in test], 0)
    Zp = np.concatenate([r["z_pred"] for r in test], 0)
    Zr = np.concatenate([np.repeat(r["z"], len(r["succ_ram"]), 0) for r in test], 0)
    desc = np.concatenate([np.array([r["root_ram"][3] > 0] * len(r["succ_ram"])) for r in test], 0)
    imminent = np.concatenate([np.array([(r["root_ram"][3] > 0) and ((179 - r["root_ram"][1]) / max(r["root_ram"][3], 1) <= 2.5)]
                                        * len(r["succ_ram"])) for r in test], 0)
    out = {"n_train_states": len(train), "n_test_states": len(test)}
    for tag, mask in (("all", np.ones(len(Ys), bool)), ("descending", desc), ("imminent", imminent)):
        if mask.sum() < 20:
            continue
        out[tag] = {
            "n": int(mask.sum()),
            "real_succ": dict(zip(NAMES, r2(Ys[mask], predict(Zt[mask])).round(3).tolist())),
            "model_succ": dict(zip(NAMES, r2(Ys[mask], predict(Zp[mask])).round(3).tolist())),
            "stale_root": dict(zip(NAMES, r2(Ys[mask], predict(Zr[mask])).round(3).tolist())),
        }
        # sign of successor vy (bounce detection): accuracy of sign(predicted vy) vs true
        for src, Z in (("real_succ", Zt), ("model_succ", Zp), ("stale_root", Zr)):
            pv = predict(Z[mask])[:, 3]
            out[tag][f"vy_sign_acc_{src}"] = float(np.mean(np.sign(pv) == np.sign(Ys[mask][:, 3])))
    return out


if __name__ == "__main__":
    result = {}
    for name in sys.argv[1:]:
        result[name] = {}
        for s in range(3):
            try:
                K, recs = collect(name, s)
            except FileNotFoundError:
                continue
            res = analyse(recs)
            res["K"] = K
            result[name][s] = res
            for tag in ("all", "descending", "imminent"):
                if tag in res:
                    t = res[tag]
                    print(f"{name} seed{s} K={K} [{tag} n={t['n']}] real {t['real_succ']} | model {t['model_succ']} | stale {t['stale_root']} | vy-sign acc real {t['vy_sign_acc_real_succ']:.2f} model {t['vy_sign_acc_model_succ']:.2f} stale {t['vy_sign_acc_stale_root']:.2f}", flush=True)
    print("JSON_BEGIN")
    print(json.dumps(result))
    print("JSON_END")
