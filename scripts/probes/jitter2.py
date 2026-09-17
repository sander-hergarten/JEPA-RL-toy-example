import numpy as np, torch
from atari_jepa.checkpoint import load_checkpoint, model_from_checkpoint
from atari_jepa.envs import FrameStacker, make_env
from atari_jepa.planning import one_step_scores

ck = load_checkpoint("runs/breakout_long_n5/seed0/checkpoint.pt", torch.device("cuda"))
cfg, model = model_from_checkpoint(ck, torch.device("cuda"))
meanings = ck["env"]["action_meanings"]

@torch.no_grad()
def rollout(kind, seed, n=2000):
    env = make_env(cfg.env); stack = FrameStacker(cfg.env.history)
    rng = np.random.default_rng(seed); obs, _ = env.reset(seed=seed); h = stack.reset(obs)
    acts, raw, score = [], [], 0.0
    for _ in range(n):
        z = model.encoder(torch.from_numpy(h).cuda().unsqueeze(0))
        s = (one_step_scores(model, z, cfg.loss.gamma) if kind == "lookahead" else model.q_head(z))[0]
        a = int(s.argmax())
        if rng.random() < 0.01: a = int(rng.integers(model.num_actions))
        acts.append(a)
        obs, r, term, trunc, _ = env.step(a); h = stack.push(obs)
        if r: raw.append(r); score += r
        if term or trunc: break
    env.close()
    acts = np.array(acts)
    lr = np.array([1 if "RIGHT" in meanings[a] else -1 if "LEFT" in meanings[a] else 0 for a in acts])
    mv = lr != 0
    pair = mv[1:] & mv[:-1]
    rev = float((lr[1:][pair] != lr[:-1][pair]).mean()) if pair.any() else float("nan")
    return dict(steps=len(acts), score=score, switch=float((acts[1:] != acts[:-1]).mean()), rev=rev,
                raw=raw, lr_frac=float(mv.mean()))

for kind in ("lookahead", "q"):
    rs = [rollout(kind, s) for s in (10000, 10002, 10004, 10005)]
    allraw = [v for r in rs for v in r["raw"]]
    vals, cnts = np.unique(allraw, return_counts=True) if allraw else ([], [])
    print(f"{kind:10s} switch {np.mean([r['switch'] for r in rs]):.2f}  direction-reversal "
          f"{np.nanmean([r['rev'] for r in rs]):.2f}  moving {np.mean([r['lr_frac'] for r in rs]):.2f}  "
          f"score/2000 steps {np.mean([r['score'] for r in rs]):5.1f}  episodes ended {sum(r['steps']<2000 for r in rs)}/4")
    print(f"{'':10s} brick values hit: " + ", ".join(f"{int(v)}pt x{c}" for v, c in zip(vals, cnts)) +
          f"   (clipped to +1 for training: {len(allraw)} identical rewards)")
