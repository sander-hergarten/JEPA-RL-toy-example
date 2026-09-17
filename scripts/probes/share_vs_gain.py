"""Replay-only geometry vs planning gain across arms (prints only)."""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

from atari_jepa.checkpoint import load_checkpoint, model_from_checkpoint
from atari_jepa.replay import SequenceReplay

N = 512
ARMS = ["breakout_se_nstep_500k", "breakout_se_k10_500k", "breakout_se_k20_500k", "breakout_se_k30_500k",
        "breakout_se_k10_qd5_500k", "breakout_se_k20_qd5_500k", "breakout_se_lmu_500k", "breakout_se_lmu_k10_500k",
        "breakout_se_lmu_k20_500k", "breakout_se_lmu_k30_500k", "breakout_se_hjepa_detached_500k", "breakout_se_hjepa_500k",
        "breakout_se_nstep3_500k", "breakout_se_nstep10_500k", "breakout_se_base_500k", "breakout_se_sf_500k",
        "breakout_se_n5_rr2_bigbuf_500k", "breakout_wm_delta_motion_500k", "breakout_se_k30_depth_inverse_500k"]


def evalmean(d, fname):
    p = f"{d}/{fname}"
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    return float(j["return_mean"])


@torch.no_grad()
def geom(model, cfg, d, device):
    replay = SequenceReplay.load(f"{d}/replay.npz", (cfg.env.screen_size, cfg.env.screen_size))
    rng = np.random.default_rng(7)
    roots = replay.sample_roots(4 * N, rng)
    cap = replay.capacity
    ok = np.ones(len(roots), dtype=bool)
    for off in (1, 2):
        idx = roots + off
        slot = idx % cap
        ok &= (idx < replay.n_written) & (replay.abs_index[slot] == idx) & (replay.episode_id[slot] == replay.episode_id[roots % cap])
    ok &= replay.has_transition[(roots + 1) % cap]
    roots = roots[ok][:N]
    zg = model.encoder(torch.from_numpy(replay.stacks_at(roots)).to(device))
    z0 = zg.flatten(1)
    z1 = model.encoder(torch.from_numpy(replay.stacks_at(roots + 1)).to(device)).flatten(1)
    z2 = model.encoder(torch.from_numpy(replay.stacks_at(roots + 2)).to(device)).flatten(1)
    d1, d2 = z1 - z0, z2 - z1
    A = model.num_actions
    zr = zg.repeat_interleave(A, 0)
    acts = torch.arange(A, device=device).repeat(len(roots))
    nz = model.dynamics(zr, acts).flatten(1).view(len(roots), A, -1)
    V = model.q_head(nz.flatten(0, 1)).max(-1).values.view(len(roots), A)
    Vz = model.q_head(z0).max(-1).values
    Vbar = V.mean(1)
    corr = float(np.corrcoef(Vz.cpu().numpy(), Vbar.cpu().numpy())[0, 1])
    corr_min = float(np.corrcoef(Vz.cpu().numpy(), V.min(1).values.cpu().numpy())[0, 1])
    # novelty of the imagined successor relative to the root (fraction of energy not along z)
    zn = F.normalize(z0, dim=1)
    proj = (nz * zn[:, None, :]).sum(-1, keepdim=True) * zn[:, None, :]
    novelty = float(((nz - proj).norm(dim=-1) ** 2 / nz.norm(dim=-1) ** 2).mean())
    return {
        "cos_dd": float(F.cosine_similarity(d1, d2, dim=1).mean()),
        "share": float((d1.norm(dim=1) ** 2 / (2 * z0.norm(dim=1) ** 2)).mean()),
        "corr_V_root_vs_imagined": corr,
        "corr_V_root_vs_imagined_min": corr_min,
        "imagined_novelty": novelty,
        "V_range_imagined": float((V.max(1).values - V.min(1).values).mean()),
        "V_state_std_imagined": float(Vbar.std()),
        "V_state_std_root": float(Vz.std()),
    }


if __name__ == "__main__":
    device = torch.device("cuda")
    print("arm K seed | cos_dd share novelty corrV corrVmin Vrange Vstd_im Vstd_root | Q H1 H5 gain1")
    for name in ARMS:
        rows = []
        for s in range(3):
            d = f"runs/{name}/seed{s}"
            if not os.path.exists(f"{d}/checkpoint.pt") or not os.path.exists(f"{d}/replay.npz"):
                continue
            try:
                ckpt = load_checkpoint(f"{d}/checkpoint.pt", device)
                cfg, model = model_from_checkpoint(ckpt, device)
                model.eval()
                g = geom(model, cfg, d, device)
            except Exception as exc:
                print(f"[{name} s{s}] failed: {exc}", flush=True)
                continue
            q = evalmean(d, "eval_q_eps001.json"); h1 = evalmean(d, "eval_lookahead_h1_eps001.json"); h5 = evalmean(d, "eval_lookahead_h5_eps001.json")
            if h1 is None:
                h1 = evalmean(d, "eval_lookahead_eps001.json")
            g.update({"Q": q, "H1": h1, "H5": h5, "gain1": (h1 - q) if (h1 is not None and q is not None) else None})
            rows.append(g)
            fmt = lambda v: "None" if v is None else f"{v:.3f}"
            print(f"[{name} K={cfg.replay.rollout_steps} s{s}] " + " ".join(f"{k}={fmt(v)}" for k, v in g.items()), flush=True)
        if rows:
            m = lambda k: float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else float("nan")
            print(f"== {name} K={cfg.replay.rollout_steps} MEAN " + " ".join(f"{k}={m(k):.3f}" for k in rows[0]), flush=True)
    print("SHARE_DONE")
