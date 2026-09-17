"""Read-only geometry probe on frozen checkpoints (prints to stdout, writes nothing).

Part A (counterfactual, cloned emulator states): decompose the one-step error g(z,a) - f(x'_a) into a
common-mode part (same for every action: the ball) and a differential part (varies with the action: the
paddle), measure the fidelity of the model's action displacement direction, and ask which part corrupts
the Q head's ranking of the branches by scoring two hybrid successors:
    hyb_ball_real   = mean_a f(x'_a) + (g(z,a) - mean_a g(z,a))   real ball, model paddle
    hyb_paddle_real = mean_a g(z,a) + (f(x'_a) - mean_a f(x'_a))  model ball, real paddle
Part B (replay roots): straightness of consecutive latent deltas along real trajectories, cross-state
consistency of each action's predicted displacement direction, and an ANOVA of V[s,a] = max_b Q(g(z_s,a))
into a per-action constant and a state x action interaction.
"""
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from atari_jepa.checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from atari_jepa.envs import FrameStacker, make_env
from atari_jepa.replay import SequenceReplay

PROBES, EVERY, EPS = 150, 6, 0.1
N_ROOTS = 512


def kendall(a, b):
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return s / (n * (n - 1) / 2)


def qmax(model, flat):
    return model.q_head(flat).max(-1).values


@torch.no_grad()
def part_a(model, cfg, ckpt, device, seed):
    env = make_env(cfg.env)
    check_compatible(ckpt["env"], env.metadata())
    stacker = FrameStacker(cfg.env.history)
    A = env.num_actions
    rng = np.random.default_rng(51_000 + seed)
    S = defaultdict(list)
    real_disp = defaultdict(list)  # per action: real displacement directions across states
    probes, ep = 0, 0
    while probes < PROBES and ep < 6:
        obs, _ = env.reset(seed=51_000 + seed * 10 + ep)
        history = stacker.reset(obs)
        for t in range(3000):
            if t % EVERY == 0 and probes < PROBES:
                snap = env.clone_state()
                succ = []
                for a in range(A):
                    env.restore_state(snap)
                    nxt, _, _, _, _ = env.step(a)
                    succ.append(np.concatenate([history[1:], nxt[None]], 0))
                env.restore_state(snap)
                z = model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))
                zf = z.flatten(1)
                zt = model.encoder(torch.from_numpy(np.stack(succ)).to(device)).flatten(1)  # [A,D]
                acts = torch.arange(A, device=device)
                zp = model.dynamics(z.expand(A, *z.shape[1:]), acts).flatten(1)  # [A,D]
                ft, gt = zt.mean(0, keepdim=True), zp.mean(0, keepdim=True)
                dt, dp = zt - ft, zp - gt  # real / model action displacement
                e = zp - zt
                e_common, e_diff = gt - ft, dp - dt
                znorm = float(zf.norm())
                S["z_norm"].append(znorm)
                S["step_real"].append(float((zt - zf).norm(dim=1).mean()))
                S["act_real"].append(float(dt.norm(dim=1).mean()))
                S["act_model"].append(float(dp.norm(dim=1).mean()))
                S["err_total"].append(float(e.norm(dim=1).mean()))
                S["err_common"].append(float(e_common.norm()))
                S["err_diff"].append(float(e_diff.norm(dim=1).mean()))
                S["fidelity_cos"].append(float(F.cosine_similarity(dp, dt, dim=1).mean()))
                # error projected onto the real action subspace U = span(dt)
                Qb, _ = torch.linalg.qr(dt.T)  # [D, A]
                pe = (e @ Qb) @ Qb.T
                S["err_in_action_subspace_frac"].append(float((pe.norm(dim=1) ** 2).sum() / (e.norm(dim=1) ** 2).sum()))
                S["err_in_action_subspace_over_act"].append(float(pe.norm(dim=1).mean() / dt.norm(dim=1).mean()))
                # how much of the ball's real step lies in the paddle subspace (sanity: should be small)
                sr = (zt - zf)
                S["step_in_action_subspace_frac"].append(float(((sr @ Qb) @ Qb.T).norm(dim=1).pow(2).sum() / sr.norm(dim=1).pow(2).sum()))
                for a in range(A):
                    real_disp[a].append(F.normalize(dt[a], dim=0).cpu())
                # value rankings under the arm's own Q head
                v_real = qmax(model, zt).cpu().numpy()
                v_model = qmax(model, zp).cpu().numpy()
                v_hb = qmax(model, ft + dp).cpu().numpy()  # real ball, model paddle
                v_hp = qmax(model, gt + dt).cpu().numpy()  # model ball, real paddle
                v_root = float(qmax(model, zf))
                S["tau_model_vs_real"].append(kendall(v_real, v_model))
                S["tau_hyb_ballreal_vs_real"].append(kendall(v_real, v_hb))
                S["tau_hyb_paddlereal_vs_real"].append(kendall(v_real, v_hp))
                S["agree_model"].append(float(v_real.argmax() == v_model.argmax()))
                S["agree_hyb_ballreal"].append(float(v_real.argmax() == v_hb.argmax()))
                S["agree_hyb_paddlereal"].append(float(v_real.argmax() == v_hp.argmax()))
                S["vrange_real"].append(float(v_real.max() - v_real.min()))
                S["vrange_model"].append(float(v_model.max() - v_model.min()))
                S["v_shift_common"].append(float(v_model.mean() - v_real.mean()))
                S["v_real_minus_root"].append(float(v_real.mean() - v_root))
                # Q-gradient at the real successors: error component along it, and the induced value error
                with torch.enable_grad():
                    zt_req = zt.detach().clone().requires_grad_(True)
                    g = torch.autograd.grad(model.q_head(zt_req).max(-1).values.sum(), zt_req)[0]
                gdir = F.normalize(g, dim=1)
                S["qgrad_norm"].append(float(g.norm(dim=1).mean()))
                S["err_along_qgrad_abs"].append(float((e * gdir).sum(1).abs().mean()))
                S["act_along_qgrad_abs"].append(float((dt * gdir).sum(1).abs().mean()))
                S["cos_err_qgrad"].append(float(F.cosine_similarity(e, g, dim=1).abs().mean()))
                S["cos_act_qgrad"].append(float(F.cosine_similarity(dt, g, dim=1).abs().mean()))
                probes += 1
            z = model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))
            action = int(model.q_head(z).argmax(-1))
            if rng.random() < EPS:
                action = int(rng.integers(A))
            obs, _, term, trunc, _ = env.step(action)
            history = stacker.push(obs)
            if term or trunc:
                break
        ep += 1
    env.close()
    out = {k: float(np.mean(v)) for k, v in S.items()}
    # cross-state consistency of the REAL action displacement direction, per action
    cons = []
    for a in range(A):
        M = torch.stack(real_disp[a])  # [n, D] unit
        G = M @ M.T
        n = G.shape[0]
        cons.append(float((G.sum() - G.diagonal().sum()) / (n * (n - 1))))
    out["real_action_dir_consistency"] = float(np.mean(cons))
    out["n"] = probes
    return out


@torch.no_grad()
def part_b(model, cfg, run_dir, device):
    replay = SequenceReplay.load(f"{run_dir}/replay.npz", (cfg.env.screen_size, cfg.env.screen_size))
    rng = np.random.default_rng(7)
    roots = replay.sample_roots(4 * N_ROOTS, rng)
    cap = replay.capacity
    ok = np.ones(len(roots), dtype=bool)
    for off in (1, 2):
        idx = roots + off
        slot = idx % cap
        ok &= (idx < replay.n_written) & (replay.abs_index[slot] == idx) & (replay.episode_id[slot] == replay.episode_id[roots % cap])
    ok &= replay.has_transition[(roots + 1) % cap]
    roots = roots[ok][:N_ROOTS]
    z0 = model.encoder(torch.from_numpy(replay.stacks_at(roots)).to(device)).flatten(1)
    z1 = model.encoder(torch.from_numpy(replay.stacks_at(roots + 1)).to(device)).flatten(1)
    z2 = model.encoder(torch.from_numpy(replay.stacks_at(roots + 2)).to(device)).flatten(1)
    d1, d2 = z1 - z0, z2 - z1
    out = {}
    out["n_roots"] = int(len(roots))
    out["delta_cos_consecutive"] = float(F.cosine_similarity(d1, d2, dim=1).mean())
    out["delta_straightness"] = float(((d1 + d2).norm(dim=1) / (d1.norm(dim=1) + d2.norm(dim=1))).mean())
    out["delta_norm_over_z"] = float((d1.norm(dim=1) / z0.norm(dim=1)).mean())
    # cross-state consistency of the delta direction, restricted to same taken action (a_t == a_{t+1})
    a0 = torch.from_numpy(replay.actions[roots % cap]).to(device)
    a1 = torch.from_numpy(replay.actions[(roots + 1) % cap]).to(device)
    u = F.normalize(d1, dim=1)
    G = u @ u.T
    n = G.shape[0]
    out["delta_dir_consistency_all"] = float((G.sum() - G.diagonal().sum()) / (n * (n - 1)))
    same = (a0[:, None] == a0[None, :]) & ~torch.eye(n, dtype=torch.bool, device=device)
    out["delta_dir_consistency_same_action"] = float(G[same].mean())
    # model action displacement direction consistency across states
    A = model.num_actions
    zg = model.encoder(torch.from_numpy(replay.stacks_at(roots)).to(device))
    zr = zg.repeat_interleave(A, 0)
    acts = torch.arange(A, device=device).repeat(len(roots))
    nz = model.dynamics(zr, acts).flatten(1).view(len(roots), A, -1)
    disp = nz - nz.mean(1, keepdim=True)
    cons = []
    for a in range(A):
        ua = F.normalize(disp[:, a], dim=1)
        Ga = ua @ ua.T
        cons.append(float((Ga.sum() - Ga.diagonal().sum()) / (n * (n - 1))))
    out["model_action_dir_consistency"] = float(np.mean(cons))
    # ANOVA of V[s,a]
    V = model.q_head(nz.flatten(0, 1)).max(-1).values.view(len(roots), A)
    mu = V.mean()
    alpha = V.mean(0) - mu
    beta = V.mean(1) - mu
    gamma = V - mu - alpha[None, :] - beta[:, None]
    out["V_var_action_main"] = float((alpha ** 2).mean())
    out["V_var_state_main"] = float((beta ** 2).mean())
    out["V_var_interaction"] = float((gamma ** 2).mean())
    out["V_action_main_over_interaction"] = float((alpha ** 2).mean() / (gamma ** 2).mean())
    out["planner_picks_main_effect_argmax_frac"] = float((V.argmax(1) == alpha.argmax()).float().mean())
    out["alpha"] = [round(float(x), 4) for x in alpha]
    q0 = model.q_head(zg)
    out["q_argmax_hist"] = [int((q0.argmax(1) == a).sum()) for a in range(A)]
    out["planner_argmax_hist"] = [int((V.argmax(1) == a).sum()) for a in range(A)]
    return out


if __name__ == "__main__":
    device = torch.device("cuda")
    for name in sys.argv[1:]:
        rows_a, rows_b = [], []
        for s in range(3):
            d = f"runs/{name}/seed{s}"
            try:
                ckpt = load_checkpoint(f"{d}/checkpoint.pt", device)
            except FileNotFoundError:
                continue
            cfg, model = model_from_checkpoint(ckpt, device)
            model.eval()
            b = part_b(model, cfg, d, device)
            a = part_a(model, cfg, ckpt, device, s)
            rows_a.append(a); rows_b.append(b)
            print(f"[{name} s{s}] A: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in a.items()), flush=True)
            print(f"[{name} s{s}] B: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in b.items()), flush=True)
        if rows_a:
            m = lambda rows, k: float(np.mean([r[k] for r in rows]))
            print(f"== {name} K={cfg.replay.rollout_steps} MEAN A: " + " ".join(f"{k}={m(rows_a,k):.4f}" for k in rows_a[0] if isinstance(rows_a[0][k], float)), flush=True)
            print(f"== {name} K={cfg.replay.rollout_steps} MEAN B: " + " ".join(f"{k}={m(rows_b,k):.4f}" for k in rows_b[0] if isinstance(rows_b[0][k], float)), flush=True)
    print("PROBE_DONE")
