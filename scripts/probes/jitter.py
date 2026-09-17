import numpy as np, torch, json
from atari_jepa.checkpoint import load_checkpoint, model_from_checkpoint
from atari_jepa.envs import FrameStacker, make_env
from atari_jepa.planning import one_step_scores

ck = load_checkpoint("runs/breakout_long_n5/seed0/checkpoint.pt", torch.device("cuda"))
cfg, model = model_from_checkpoint(ck, torch.device("cuda"))
meanings = ck["env"]["action_meanings"]

@torch.no_grad()
def rollout(kind, n=1500, seed=10008):
    env = make_env(cfg.env); stack = FrameStacker(cfg.env.history)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed); h = stack.reset(obs)
    acts, margins, raw_rewards = [], [], []
    for _ in range(n):
        z = model.encoder(torch.from_numpy(h).cuda().unsqueeze(0))
        scores = (one_step_scores(model, z, cfg.loss.gamma) if kind == "lookahead" else model.q_head(z))[0]
        top2 = scores.topk(2).values
        margins.append(float(top2[0] - top2[1]) / max(float(scores.max() - scores.min()), 1e-9))
        a = int(scores.argmax())
        if rng.random() < 0.01: a = int(rng.integers(model.num_actions))
        acts.append(a)
        obs, r, term, trunc, _ = env.step(a); h = stack.push(obs)
        if r: raw_rewards.append(r)
        if term or trunc: break
    env.close()
    acts = np.array(acts)
    switch = float((acts[1:] != acts[:-1]).mean())
    lr = np.array([1 if "RIGHT" in meanings[a] else -1 if "LEFT" in meanings[a] else 0 for a in acts])
    moving = lr != 0
    rev = float((lr[1:][moving[1:] & moving[:-1]] != lr[:-1][moving[1:] & moving[:-1]]).mean())
    counts = {meanings[a]: int((acts == a).sum()) for a in range(model.num_actions)}
    return dict(kind=kind, steps=len(acts), switch_rate=switch, direction_reversal_rate=rev,
                median_relative_margin=float(np.median(margins)), action_counts=counts,
                raw_reward_values=dict(zip(*[x.tolist() for x in np.unique(raw_rewards, return_counts=True)])) if raw_rewards else {})

for kind in ("lookahead", "q"):
    r = rollout(kind)
    print(f"{r['kind']:10s} steps {r['steps']:5d}  action-switch rate {r['switch_rate']:.2f}  "
          f"direction reversals {r['direction_reversal_rate']:.2f}  median margin (relative) {r['median_relative_margin']:.3f}")
    print(f"{'':10s} actions {r['action_counts']}   raw rewards collected {r['raw_reward_values']}")
