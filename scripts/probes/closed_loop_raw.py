"""Raw closed-loop logs (read-only, frozen checkpoints): per decision, action, ball (x,y), paddle x,
lives, reward, for the Q policy and the H=1 planner. Analysis is done offline."""
import json
import sys

import numpy as np
import torch

from atari_jepa.checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from atari_jepa.envs import FrameStacker, make_env
from atari_jepa.planning import one_step_scores

EPISODES, EPS, MAXDEC = 3, 0.01, 3000


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
            lives = int(env._ale.lives())
            z = model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))
            qa = int(model.q_head(z).argmax(-1))
            if controller == "q":
                action = qa
            else:
                action = int(one_step_scores(model, z, g).argmax(-1))
            if rng.random() < EPS:
                action = int(rng.integers(A))
            obs, r, term, trunc, _ = env.step(action)
            history = stacker.push(obs)
            rows.append([t, action, qa, bx, by, px, lives, int(np.sign(r))])
            if term or trunc:
                break
        logs.append(rows)
    return logs


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
            result[name][s] = {"K": cfg.replay.rollout_steps}
            for ctl in ("q", "lookahead"):
                logs = play(model, cfg, env, stacker, device, ctl, 10_000)
                result[name][s][ctl] = logs
                print(f"{name} seed{s} {ctl}: episodes {[len(r) for r in logs]} returns {[sum(x[7] for x in r) for r in logs]}", flush=True)
            env.close()
    print("JSON_BEGIN")
    print(json.dumps(result))
    print("JSON_END")
