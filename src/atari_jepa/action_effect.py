"""How action-sensitive *should* a latent be? The counterfactual reference this project never had.

Every action-sensitivity number reported here -- the shuffled-action penalty, the mean pairwise
next-latent distance -- is uncalibrated. A model scoring 0.066 might be doing well or catastrophically
badly, because nobody measured how much the choice of action actually changes the next observation.

This measures it. From one identical emulator state (``cloneSystemState``, so branches share the
sticky-action draw and differ only in the action) it takes every action in turn, encodes each real
successor with the frozen encoder, and reports the same statistic the diagnostics compute on predicted
latents. Three numbers make it readable:

* ``true_action_distance``   -- mean pairwise cosine distance between real successors under different
                                actions: the ceiling a perfect one-step model would reproduce
* ``model_action_distance``  -- the same statistic on the model's predicted successors, same states
* ``total_change``           -- distance between the root latent and its successor, whatever the action

The ratio of the first to the third says what fraction of a transition the agent controls at all; with
sticky actions (0.25 here) the chosen action is ignored a quarter of the time, so this is the
*realized* effect, which is what the model is trained to predict.

    python -m atari_jepa.action_effect --checkpoint RUN/checkpoint.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from .envs import FrameStacker, make_env
from .losses import cosine_distance
from .utils import configure_threads, select_device, write_json


def pairwise_mean_distance(z: torch.Tensor) -> float:
    """Mean cosine distance over all unordered pairs of ``[A, ...]`` latents."""
    A = z.shape[0]
    pairs = [(i, j) for i in range(A) for j in range(i + 1, A)]
    if not pairs:
        return float("nan")
    u = torch.stack([z[i] for i, _ in pairs])
    v = torch.stack([z[j] for _, j in pairs])
    return float(cosine_distance(u, v).mean())


@torch.no_grad()
def measure(model, cfg, env_meta, device, episodes: int, max_decisions: int, seed_base: int,
            epsilon: float, probe_every: int, max_probes: int) -> dict[str, Any]:
    env = make_env(cfg.env)
    check_compatible(env_meta, env.metadata())
    if not hasattr(env, "clone_state"):
        raise SystemExit(f"{cfg.env.id} cannot clone emulator state; the counterfactual needs an ALE env")
    stacker = FrameStacker(cfg.env.history)
    A = env.num_actions
    rng = np.random.default_rng(seed_base)
    true_d, model_d, total_d, controlled = [], [], [], []
    probes = 0
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed_base + ep)
            history = stacker.reset(obs)
            for t in range(max_decisions):
                if probes < max_probes and t % probe_every == 0:
                    snapshot = env.clone_state()
                    successors = []
                    for a in range(A):
                        env.restore_state(snapshot)
                        nxt, _, _, _, _ = env.step(a)
                        successors.append(np.concatenate([history[1:], nxt[None]], axis=0))
                    env.restore_state(snapshot)
                    z_root = model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))
                    z_true = model.encoder(torch.from_numpy(np.stack(successors)).to(device))
                    acts = torch.arange(A, device=device)
                    z_pred = model.dynamics(z_root.expand(A, *z_root.shape[1:]), acts)
                    true_d.append(pairwise_mean_distance(z_true))
                    model_d.append(pairwise_mean_distance(z_pred))
                    total_d.append(float(cosine_distance(z_root.expand_as(z_true), z_true).mean()))
                    probes += 1

                z = model.encoder(torch.from_numpy(history).to(device).unsqueeze(0))
                action = int(model.q_head(z).argmax(dim=-1))
                if epsilon > 0 and rng.random() < epsilon:
                    action = int(rng.integers(A))
                obs, _, terminated, truncated, _ = env.step(action)
                history = stacker.push(obs)
                if terminated or truncated:
                    break
    finally:
        env.close()

    t_mean, m_mean, c_mean = float(np.mean(true_d)), float(np.mean(model_d)), float(np.mean(total_d))
    return {
        "probes": probes,
        "num_actions": A,
        "sticky_action_prob": cfg.env.sticky_action_prob,
        "true_action_distance": t_mean,
        "true_action_distance_std": float(np.std(true_d)),
        "model_action_distance": m_mean,
        "model_action_distance_std": float(np.std(model_d)),
        "total_change": c_mean,
        "controlled_fraction": t_mean / c_mean if c_mean > 0 else float("nan"),
        "model_over_true": m_mean / t_mean if t_mean > 0 else float("nan"),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--max-decisions", type=int, default=3000)
    p.add_argument("--seed-base", type=int, default=30_000)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--probe-every", type=int, default=5)
    p.add_argument("--max-probes", type=int, default=2000)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    configure_threads(args.threads)
    device = select_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    cfg, model = model_from_checkpoint(ckpt, device)
    model.eval()
    r = measure(model, cfg, ckpt["env"], device, args.episodes, args.max_decisions, args.seed_base,
                args.epsilon, args.probe_every, args.max_probes)
    r["checkpoint"] = str(args.checkpoint)
    r["variant"] = cfg.loss.variant_name()
    r["rollout_steps"] = cfg.replay.rollout_steps
    print(f"{args.checkpoint}: K={r['rollout_steps']}, {r['probes']} probed states")
    print(f"  real successors differ by   {r['true_action_distance']:.4f} across actions")
    print(f"  model predicts              {r['model_action_distance']:.4f}   "
          f"({r['model_over_true']:.2f}x the real effect)")
    print(f"  a transition changes        {r['total_change']:.4f} in total, so the action controls "
          f"{r['controlled_fraction']:.1%} of it")
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "action_effect.json"
    write_json(out, r)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
