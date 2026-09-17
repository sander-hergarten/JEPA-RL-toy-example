"""Where does rollout credit go, and how much of it is about the action?

Two measurements on a frozen checkpoint, both aimed at the same suspicion. ``gradient_reach`` (in
``diagnostics``) showed that credit does not decay backwards through the rollout -- it *saturates*,
with the per-step gain falling to ~1.007 near the root. A gradient propagated through a repeated
Jacobian converges to that Jacobian's dominant direction, after which further steps only rescale it.
If that is what happens here, the K loss terms of a long rollout are not K independent constraints but
one direction counted K times, and the action -- a small, nearly orthogonal component -- is what gets
filtered out.

* ``depth_gradient_alignment``: backpropagate each depth's latent term *separately* and compare the
  gradients they deliver to the root, pairwise. Near-parallel deep gradients mean redundancy, and an
  effective rank well below K says how much.

* ``action_pathway_share``: the gradient reaching the action embedding at step k from the depth-d term,
  against the gradient reaching the latent at the same step. An action taken at step k barely moves
  z_{k+d}, so this should fall with the gap -- which would make "deep terms carry little action
  information, and long rollouts are mostly deep terms" a measured statement rather than a story.

    python -m atari_jepa.gradient_anatomy --checkpoint RUN/checkpoint.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint, model_from_checkpoint
from .losses import dynamics_step, jepa_distances
from .replay import SequenceReplay
from .utils import configure_threads, select_device, write_json


def effective_rank(gram: torch.Tensor, eps: float = 1e-12) -> float:
    """exp(entropy) of the normalized eigenvalue spectrum: 1 = every vector parallel, n = orthogonal."""
    eig = torch.linalg.eigvalsh(gram.double()).clamp(min=0)
    p = eig / eig.sum().clamp(min=eps)
    p = p[p > eps]
    return float(torch.exp(-(p * p.log()).sum()))


def rollout_with_taps(model, obs: torch.Tensor, actions: torch.Tensor, steps: int):
    """Roll out, retaining gradients on every latent and on every action embedding.

    The action embedding is captured with a forward hook rather than re-implemented, so the tapped
    quantity is exactly the tensor the dynamics used.
    """
    embeddings: list[torch.Tensor] = []

    def tap(_module, _inputs, output):
        output.retain_grad()
        embeddings.append(output)

    handle = model.dynamics.action_embed.register_forward_hook(tap)
    try:
        z_hat, state = [model.encoder(obs[:, 0])], None
        z_hat[0].retain_grad()
        for k in range(steps):
            z_next, state = dynamics_step(model.dynamics, z_hat[k], actions[:, k], state)
            z_next.retain_grad()
            z_hat.append(z_next)
    finally:
        handle.remove()
    return z_hat, embeddings


def anatomy(model, cfg, replay: SequenceReplay, device, batch_size: int, seed: int) -> dict[str, Any]:
    K = cfg.replay.rollout_steps
    batch = replay.sample(batch_size, K, np.random.default_rng(seed)).to_torch(device)
    obs, actions = batch.observations, batch.actions

    with torch.no_grad():
        flat = obs[:, 1:].flatten(0, 1)
        target_z = model.target_encoder(flat).unflatten(0, (batch_size, K))
        target_z0 = model.target_encoder(obs[:, 0])

    root_grads, share = [], {}
    for d in range(1, K + 1):
        z_hat, embeddings = rollout_with_taps(model, obs, actions, K)
        dist = jepa_distances(z_hat, target_z, target_z0, cfg.loss.jepa_target, cfg.loss.cosine_eps)
        model.zero_grad(set_to_none=True)
        dist[:, d - 1].mean().backward()
        root_grads.append(z_hat[0].grad.detach().flatten().clone())
        # gradient reaching the action embedding at step k, against the latent at that step
        ratios = []
        for k in range(min(d, len(embeddings))):
            e_grad = embeddings[k].grad
            z_grad = z_hat[k].grad
            if e_grad is None or z_grad is None:
                ratios.append(float("nan"))
                continue
            zn = float(z_grad.norm())
            ratios.append(float(e_grad.norm()) / zn if zn > 0 else float("nan"))
        share[d] = ratios
    model.zero_grad(set_to_none=True)

    G = F.normalize(torch.stack(root_grads), dim=1)
    gram = (G @ G.T).cpu()
    deep = [d for d in range(1, K + 1) if d > max(K // 3, 1)]
    deep_pairs = [float(gram[i - 1, j - 1]) for i in deep for j in deep if i < j]
    all_pairs = [float(gram[i, j]) for i in range(K) for j in range(i + 1, K)]
    conflicting = [c for c in all_pairs if c < 0]

    # action share as a function of the gap d - k, averaged over the (d, k) pairs with that gap
    by_gap: dict[int, list[float]] = {}
    for d, ratios in share.items():
        for k, r in enumerate(ratios):
            if np.isfinite(r):
                by_gap.setdefault(d - k, []).append(r)

    return {
        "rollout_steps": K,
        "batch_size": batch_size,
        "depth_gradient_alignment": {
            "note": "cosine similarity between the root gradients delivered by different depth terms",
            "matrix": gram.tolist(),
            "effective_rank": effective_rank(gram),
            "mean_cosine_adjacent": float(np.mean([float(gram[i, i + 1]) for i in range(K - 1)])) if K > 1 else 1.0,
            "mean_cosine_deep_pairs": float(np.mean(deep_pairs)) if deep_pairs else float("nan"),
            "cosine_shallow_vs_deepest": float(gram[0, K - 1]),
            # Redundancy would show as pairs near +1; conflict shows as pairs below 0, where two depth
            # terms pull the shared encoder in opposing directions and partly cancel.
            "mean_cosine_all_pairs": float(np.mean(all_pairs)) if all_pairs else float("nan"),
            "fraction_conflicting_pairs": len(conflicting) / len(all_pairs) if all_pairs else float("nan"),
            "mean_cosine_conflicting": float(np.mean(conflicting)) if conflicting else 0.0,
        },
        "action_pathway_share": {
            "note": "||dL_d/d embed(a_k)|| / ||dL_d/d z_k||, by the gap d - k",
            "by_gap": {str(g): float(np.mean(v)) for g, v in sorted(by_gap.items())},
            "by_depth": {str(d): [float(r) for r in ratios] for d, ratios in share.items()},
        },
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--replay", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    configure_threads(args.threads)
    device = select_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    cfg, model = model_from_checkpoint(ckpt, device)
    model.eval()
    path = Path(args.replay) if args.replay else Path(args.checkpoint).parent / "replay.npz"
    if not path.exists():
        raise SystemExit(f"no replay at {path}; the run needs replay.save_with_checkpoint")
    replay = SequenceReplay.load(path, (cfg.env.screen_size, cfg.env.screen_size))

    result = anatomy(model, cfg, replay, device, args.batch_size, args.seed)
    result["checkpoint"] = str(args.checkpoint)
    result["variant"] = cfg.loss.variant_name()
    a = result["depth_gradient_alignment"]
    print(f"{args.checkpoint}: K={result['rollout_steps']}")
    print(f"  depth-gradient effective rank {a['effective_rank']:.2f} of {result['rollout_steps']}"
          f"   mean pair cos {a['mean_cosine_all_pairs']:+.3f}"
          f"   conflicting pairs {a['fraction_conflicting_pairs']:.1%}"
          f" (mean {a['mean_cosine_conflicting']:+.3f})")
    gaps = result["action_pathway_share"]["by_gap"]
    shown = "  ".join(f"gap {g}:{v:.4f}" for g, v in list(gaps.items())[:6])
    print(f"  action/latent gradient share  {shown}")
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "gradient_anatomy.json"
    write_json(out, result)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
