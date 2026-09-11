"""Held-out diagnostics for a frozen checkpoint.

    python -m atari_jepa.diagnostics --checkpoint RUN/checkpoint.pt

Collects fresh trajectories in a separately seeded environment (never training replay) with an
epsilon-greedy Q-policy, then evaluates every valid root of those trajectories with the checkpoint's
online and target networks (no parameter updates, so targets cannot move):

* latent prediction distance by depth vs. a persistence baseline and vs. shuffled / random actions;
  depths beyond the training rollout K are extrapolative and labelled so,
* root-latent statistics (per-dimension std, fraction below the variance floor, pairwise cosine),
* reward and continuation prediction by depth: loss, class counts, event precision/recall, and the
  loss of a constant class-prior predictor,
* Q TD error on real roots and on each imagined depth,
* action sensitivity of predictions at a fixed state,
* controller latency and the collection episodes' returns/lengths/interactions.

``--fit-fixed-batch N`` additionally checks that N optimizer steps on one held-out batch, with fixed
targets, reduce the prediction objective (an optimization-path check, not a gameplay claim).
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .agent import trained_modules
from .checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from .config import Config
from .envs import FrameStacker, make_env
from .losses import (
    compute_losses,
    cosine_distance,
    double_dqn_value,
    expected_reward,
    latent_std,
    reward_to_class,
    td_targets,
    unroll,
)
from .planning import QController, one_step_scores
from .replay import SequenceReplay
from .utils import configure_threads, select_device, write_json


def collect_heldout(model, cfg: Config, env_meta: dict[str, Any], device, episodes: int, max_decisions: int,
                    seed_base: int, epsilon: float) -> tuple[SequenceReplay, list[dict[str, Any]]]:
    env = make_env(cfg.env)
    check_compatible(env_meta, env.metadata())
    size = cfg.env.screen_size
    replay = SequenceReplay(max_decisions + episodes + 1, (size, size), cfg.env.history)
    controller = QController(model, device, epsilon, seed=seed_base)
    stacker = FrameStacker(cfg.env.history)
    stats = []
    total = 0
    try:
        for i in range(episodes):
            if total >= max_decisions:
                break
            frames_before = env.total_frames
            frame, _ = env.reset(seed=seed_base + i)
            history = stacker.reset(frame)
            replay.start_episode(frame)
            ret, n, term, trunc = 0.0, 0, False, False
            while total < max_decisions:
                action, _ = controller.act(history)
                frame, reward, term, trunc, _ = env.step(action)
                replay.add(action, reward, term, trunc, frame)
                history = stacker.push(frame)
                ret += reward
                n += 1
                total += 1
                if term or trunc:
                    break
            stats.append({"reset_seed": seed_base + i, "raw_return": ret, "decisions": n,
                          "emulator_frames": env.total_frames - frames_before,
                          "terminated": term, "truncated": trunc, "complete": term or trunc})
    finally:
        env.close()
    return replay, stats


def _prf(pred: np.ndarray, true: np.ndarray) -> dict[str, float | int]:
    tp = int((pred & true).sum())
    return {
        "events": int(true.sum()),
        "predicted": int(pred.sum()),
        "precision": tp / pred.sum() if pred.sum() else float("nan"),
        "recall": tp / true.sum() if true.sum() else float("nan"),
    }


class _Acc:
    """Accumulates masked per-depth sums."""

    def __init__(self):
        self.sums: dict[str, np.ndarray] = {}
        self.arrays: dict[str, list[np.ndarray]] = {}

    def add(self, key: str, values: torch.Tensor, mask: torch.Tensor) -> None:
        v = torch.where(mask, values, torch.zeros_like(values)).sum(0).double().cpu().numpy()
        c = mask.sum(0).double().cpu().numpy()
        if key not in self.sums:
            self.sums[key] = np.zeros_like(v)
            self.sums[key + "#n"] = np.zeros_like(c)
        self.sums[key] += v
        self.sums[key + "#n"] += c

    def mean(self, key: str) -> list[float | None]:
        s, n = self.sums[key], self.sums[key + "#n"]
        return [float(a / b) if b > 0 else None for a, b in zip(s, n)]

    def count(self, key: str) -> list[int]:
        return [int(x) for x in self.sums[key + "#n"]]

    def keep(self, key: str, values: np.ndarray) -> None:
        self.arrays.setdefault(key, []).append(values)

    def cat(self, key: str) -> np.ndarray:
        return np.concatenate(self.arrays[key], axis=0)


@torch.no_grad()
def evaluate_model(model, cfg: Config, replay: SequenceReplay, device, depths: list[int], batch_size: int,
                   seed: int = 0) -> dict[str, Any]:
    H = max(depths)
    A = model.num_actions
    gamma = cfg.loss.gamma
    rng = np.random.default_rng(seed)
    torch_rng = torch.Generator(device="cpu").manual_seed(seed)
    roots = replay.valid_roots()
    if len(roots) == 0:
        raise ValueError("no valid held-out roots; collect more decisions")
    acc = _Acc()
    # Mean target latent over held-out roots: Pong latents share a large static (background) component,
    # so distances are also reported after subtracting it ("centered").
    mu = torch.zeros(1, device=device)
    for start in range(0, len(roots), batch_size):
        stacks = torch.from_numpy(replay.gather(roots[start : start + batch_size], 1).observations[:, 0]).to(device)
        mu = mu + model.target_encoder(stacks).sum(0)
    mu = (mu / len(roots)).unsqueeze(0)
    root_latents, target_root_latents, obs_std = [], [], []
    sens_latent, sens_reward, sens_cont = [], [], []
    for start in range(0, len(roots), batch_size):
        batch = replay.gather(roots[start : start + batch_size], H).to_torch(device)
        obs, actions = batch.observations, batch.actions
        B = actions.shape[0]
        valid, term = batch.valid.bool(), batch.terminated.bool()
        lmask = valid & ~term
        flat = lambda enc, x: enc(x.flatten(0, 1)).unflatten(0, x.shape[:2])  # noqa: E731
        z0 = model.encoder(obs[:, 0])
        target_z = flat(model.target_encoder, obs[:, 1:])  # [B, H, ...]
        root_latents.append(z0.flatten(1).cpu())
        target_root_latents.append(model.target_encoder(obs[:, 0]).flatten(1).cpu())
        obs_std.append(obs[:, 0].float().flatten(1).cpu())

        # latent prediction with true / shuffled / random actions, and persistence
        perm = torch.randperm(B, generator=torch_rng).to(device)
        rand_actions = torch.from_numpy(rng.integers(0, A, size=actions.shape)).to(device)
        z_true = unroll(model.dynamics, z0, actions, H)
        z_shuf = unroll(model.dynamics, z0, actions[perm], H)
        z_rand = unroll(model.dynamics, z0, rand_actions, H)
        stack_d = lambda zs: torch.stack(  # noqa: E731
            [cosine_distance(zs[k + 1], target_z[:, k]) for k in range(H)], dim=1)
        stack_c = lambda zs: torch.stack(  # noqa: E731
            [cosine_distance(zs[k + 1] - mu, target_z[:, k] - mu) for k in range(H)], dim=1)
        z_persist = [z0] * (H + 1)
        for key, zs in (("true", z_true), ("shuffled", z_shuf), ("random", z_rand), ("persistence", z_persist)):
            acc.add(f"latent_{key}", stack_d(zs), lmask)
            acc.add(f"latent_centered_{key}", stack_c(zs), lmask)

        # reward / continuation / Q along the true-action rollout (depth k uses z_hat[k])
        rewards = batch.rewards
        classes = reward_to_class(rewards, valid)
        online_next_q = flat(lambda x: model.q_head(model.encoder(x)), obs[:, 1:])
        target_next_q = model.target_q_head(target_z.flatten(0, 1)).unflatten(0, (B, H))
        y = td_targets(rewards, term, double_dqn_value(online_next_q, target_next_q), gamma)
        r_logits = torch.stack([model.reward_head(z_true[k], actions[:, k]) for k in range(H)], dim=1)
        c_logits = torch.stack([model.continuation_head(z_true[k], actions[:, k]) for k in range(H)], dim=1)
        q_pred = torch.stack([model.q_head(z_true[k]).gather(1, actions[:, k : k + 1]).squeeze(1)
                              for k in range(H)], dim=1)
        ce = F.cross_entropy(r_logits.flatten(0, 1), classes.flatten(), reduction="none").view(B, H)
        acc.add("reward_ce", ce, valid)
        cont_target = (~term).float()
        p_cont = torch.sigmoid(c_logits)
        acc.add("cont_bce", F.binary_cross_entropy_with_logits(c_logits, cont_target, reduction="none"), valid)
        acc.add("cont_brier", (p_cont - cont_target) ** 2, valid)
        acc.add("q_td_huber", F.smooth_l1_loss(q_pred, y, reduction="none", beta=cfg.loss.huber_delta), valid)
        acc.add("q_td_abs", (q_pred - y).abs(), valid)
        v = valid.cpu().numpy()
        acc.keep("valid", v)
        acc.keep("reward_class", classes.cpu().numpy())
        acc.keep("reward_pred", r_logits.argmax(-1).cpu().numpy())
        acc.keep("terminal", term.cpu().numpy())
        acc.keep("terminal_pred", (p_cont < 0.5).cpu().numpy())

        # action sensitivity at the real root
        zr = z0.repeat_interleave(A, dim=0)
        acts = torch.arange(A, device=device).repeat(B)
        nz = model.dynamics(zr, acts).unflatten(0, (B, A)).flatten(2)
        nz = F.normalize(nz, dim=-1)
        pair = 1 - nz @ nz.transpose(1, 2)  # [B, A, A] cosine distances
        sens_latent.append((pair.sum((1, 2)) / (A * (A - 1))).cpu())
        er = expected_reward(model.reward_head(zr, acts)).view(B, A)
        pc = torch.sigmoid(model.continuation_head(zr, acts)).view(B, A)
        sens_reward.append((er.max(1).values - er.min(1).values).cpu())
        sens_cont.append((pc.max(1).values - pc.min(1).values).cpu())

    K = cfg.replay.rollout_steps
    out: dict[str, Any] = {"num_roots": int(len(roots)), "depths_reported": depths, "training_rollout_steps": K}

    latent = {}
    for d in depths:
        i = d - 1
        latent[f"depth_{d}"] = {
            "extrapolative": d > K,
            "count": acc.count("latent_true")[i],
            "predicted": acc.mean("latent_true")[i],
            "persistence": acc.mean("latent_persistence")[i],
            "shuffled_actions": acc.mean("latent_shuffled")[i],
            "random_actions": acc.mean("latent_random")[i],
            "centered_predicted": acc.mean("latent_centered_true")[i],
            "centered_persistence": acc.mean("latent_centered_persistence")[i],
            "centered_shuffled_actions": acc.mean("latent_centered_shuffled")[i],
            "centered_random_actions": acc.mean("latent_centered_random")[i],
        }
    out["latent_prediction_cosine_distance"] = latent

    z = torch.cat(root_latents)
    zt = torch.cat(target_root_latents)
    floor = cfg.loss.variance_floor
    out["root_latents"] = {"online": _latent_stats(z, floor, cfg.loss.variance_eps),
                           "target": _latent_stats(zt, floor, cfg.loss.variance_eps)}
    pix = torch.cat(obs_std)
    out["input_variation"] = {"pixel_std_mean": float(pix.std(0).mean()),
                              "note": "if inputs barely vary, low latent variance is not evidence of collapse"}

    valid_all = acc.cat("valid")
    rc, rp = acc.cat("reward_class"), acc.cat("reward_pred")
    tt, tp = acc.cat("terminal"), acc.cat("terminal_pred")
    # constant class-prior predictor fitted on the real-root (depth 0) held-out transitions
    root_classes = rc[:, 0][valid_all[:, 0]]
    prior = np.bincount(root_classes, minlength=3) / max(len(root_classes), 1)
    prior_ce = float(-(prior * np.log(np.clip(prior, 1e-12, 1))).sum())
    p_term = tt[:, 0][valid_all[:, 0]].mean() if valid_all[:, 0].any() else 0.0
    pc = 1 - p_term
    prior_bce = float(-(pc * np.log(max(pc, 1e-12)) + p_term * np.log(max(p_term, 1e-12))))
    reward, cont, q = {}, {}, {}
    ce, bce, brier = acc.mean("reward_ce"), acc.mean("cont_bce"), acc.mean("cont_brier")
    huber, tdabs = acc.mean("q_td_huber"), acc.mean("q_td_abs")
    for k in range(H):
        m = valid_all[:, k]
        ck, pk = rc[:, k][m], rp[:, k][m]
        tag = f"depth_{k}" + ("" if k < K else "_extrapolative")
        reward[tag] = {
            "count": int(m.sum()),
            "class_counts": {"-1": int((ck == 0).sum()), "0": int((ck == 1).sum()), "+1": int((ck == 2).sum())},
            "cross_entropy": ce[k],
            "prior_cross_entropy": prior_ce,
            "event_+1": _prf(pk == 2, ck == 2),
            "event_-1": _prf(pk == 0, ck == 0),
            "accuracy_note": "accuracy omitted: dominated by zero-reward steps",
        }
        cont[tag] = {
            "count": int(m.sum()),
            "terminal_events": int(tt[:, k][m].sum()),
            "bce": bce[k],
            "brier": brier[k],
            "prior_bce": prior_bce,
            "terminal": _prf(tp[:, k][m], tt[:, k][m]),
        }
        q[tag] = {"count": int(m.sum()), "huber": huber[k], "abs_td": tdabs[k],
                  "state": "real root" if k == 0 else "imagined"}
    out["reward_prediction"] = reward
    out["continuation_prediction"] = cont
    out["q_td_error"] = q
    out["action_sensitivity_at_root"] = {
        "mean_pairwise_cosine_distance_next_latent": float(torch.cat(sens_latent).mean()),
        "mean_expected_reward_range": float(torch.cat(sens_reward).mean()),
        "mean_continuation_prob_range": float(torch.cat(sens_cont).mean()),
        "note": "not every action must change every state immediately; compare with latent_prediction distances",
    }
    return out


def _latent_stats(z: torch.Tensor, floor: float, eps: float) -> dict[str, Any]:
    std = latent_std(z, eps)
    raw_std = z.std(0, unbiased=False)
    n = min(len(z), 1024)
    zn = F.normalize(z[:n], dim=1)
    sim = zn @ zn.T
    pairwise = float((sim.sum() - sim.diagonal().sum()) / max(n * (n - 1), 1))
    q = torch.quantile(raw_std, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return {
        "samples": int(len(z)),
        "dims": int(z.shape[1]),
        "std_quantiles_5_25_50_75_95": [float(x) for x in q],
        "std_mean": float(raw_std.mean()),
        "frac_below_variance_floor": float((std < floor).float().mean()),
        "frac_dims_std_below_1e-3": float((raw_std < 1e-3).float().mean()),
        "mean_pairwise_cosine_similarity": pairwise,
    }


def controller_latency(model, cfg: Config, replay: SequenceReplay, device, n: int = 200) -> dict[str, float]:
    roots = replay.valid_roots()[:n]
    stacks = replay.gather(roots, 1).observations[:, 0]
    out = {}
    with torch.no_grad():
        for name in ("q", "lookahead"):
            times = []
            for s in stacks:
                t = time.perf_counter()
                z = model.encoder(torch.from_numpy(s).to(device).unsqueeze(0))
                if name == "q":
                    int(model.q_head(z).argmax())
                else:
                    int(one_step_scores(model, z, cfg.loss.gamma).argmax())
                times.append(time.perf_counter() - t)
            out[f"{name}_ms_mean"] = float(np.mean(times) * 1e3)
    return out


FIT_COMPONENTS = ("q_imagined", "jepa", "reward", "continuation", "variance")


def fit_fixed_batch(model, cfg: Config, replay: SequenceReplay, device, steps: int, batch_size: int,
                    seed: int = 0, components: list[str] | None = None) -> dict[str, Any]:
    """Optimize a copy of the model on one fixed batch with *fixed* target networks (no EMA).

    ``components`` restricts the objective (root Q is always included); default: the checkpoint's losses.
    """
    if components is not None:
        unknown = set(components) - set(FIT_COMPONENTS)
        if unknown:
            raise ValueError(f"unknown fit components {sorted(unknown)}; choose from {FIT_COMPONENTS}")
        cfg = copy.deepcopy(cfg)
        for name in FIT_COMPONENTS:
            setattr(cfg.loss, name, name in components)
    model = copy.deepcopy(model).train()
    batch = replay.sample(batch_size, cfg.replay.rollout_steps, np.random.default_rng(seed)).to_torch(device)
    opt = torch.optim.Adam(model.online_parameters(), lr=cfg.optim.lr, eps=cfg.optim.adam_eps)
    history = []
    for i in range(steps):
        loss, metrics = compute_losses(model, batch, cfg.loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.online_parameters(), cfg.optim.grad_clip_norm)
        opt.step()
        if i == 0 or i == steps - 1 or (i + 1) % max(steps // 10, 1) == 0:
            history.append({"step": i, **{k: metrics[k] for k in metrics if k.startswith("loss_")}})
    return {"steps": steps, "batch_size": batch_size, "components": components or "checkpoint losses",
            "trajectory": history,
            "note": "fixed targets; checks the optimization path only, not gameplay learning"}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--max-decisions", type=int, default=None)
    p.add_argument("--seed-base", type=int, default=None)
    p.add_argument("--policy-epsilon", type=float, default=None)
    p.add_argument("--depths", type=int, nargs="+", default=None)
    p.add_argument("--fit-fixed-batch", type=int, default=0, metavar="STEPS")
    p.add_argument("--fit-components", nargs="+", default=None, choices=FIT_COMPONENTS,
                   help="objective for --fit-fixed-batch (root Q always on); default: checkpoint losses")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    configure_threads(args.threads)
    device = select_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    cfg, model = model_from_checkpoint(ckpt, device)
    d = cfg.diagnostics
    episodes = args.episodes or d.episodes
    max_decisions = args.max_decisions or d.max_decisions
    seed_base = args.seed_base if args.seed_base is not None else d.seed_base
    eps = args.policy_epsilon if args.policy_epsilon is not None else d.policy_epsilon
    depths = sorted(args.depths or d.depths)

    t0 = time.perf_counter()
    replay, episodes_stats = collect_heldout(model, cfg, ckpt["env"], device, episodes, max_decisions, seed_base, eps)
    collect_s = time.perf_counter() - t0
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "variant": cfg.loss.variant_name(),
        "trained_modules": trained_modules(cfg),
        "train_counters": ckpt["counters"],
        "heldout": {"seed_base": seed_base, "policy": f"epsilon-greedy Q, epsilon={eps}",
                    "episodes": episodes_stats, "decisions": int(sum(e["decisions"] for e in episodes_stats)),
                    "transitions": replay.num_transitions(), "collect_wall_time_s": collect_s},
    }
    result.update(evaluate_model(model, cfg, replay, device, depths, d.batch_size, seed=seed_base))
    result["latency"] = controller_latency(model, cfg, replay, device)
    if args.fit_fixed_batch:
        result["fixed_batch_fit"] = fit_fixed_batch(model, cfg, replay, device, args.fit_fixed_batch,
                                                    cfg.optim.batch_size, components=args.fit_components)
    result["wall_time_s"] = time.perf_counter() - t0
    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "diagnostics.json"
    write_json(out, result)
    print_summary(result)
    print(f"-> {out}")


def _f(x) -> str:
    return "  n/a " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:6.3f}"


def print_summary(r: dict[str, Any]) -> None:
    print(f"variant {r['variant']}; trained modules: {', '.join(r['trained_modules'])}")
    h = r["heldout"]
    rets = [e["raw_return"] for e in h["episodes"]]
    print(f"held-out: {len(rets)} episodes, {h['decisions']} decisions, returns {rets}")
    print("latent cosine distance     predicted  persist  shuffled  random  | centered: pred  persist  shuffled  random")
    for k, v in r["latent_prediction_cosine_distance"].items():
        tag = " (extrapolative)" if v["extrapolative"] else ""
        print(f"  {k:<10}{tag:<16} {_f(v['predicted'])}   {_f(v['persistence'])}   "
              f"{_f(v['shuffled_actions'])}   {_f(v['random_actions'])}  |         {_f(v['centered_predicted'])} "
              f"{_f(v['centered_persistence'])}   {_f(v['centered_shuffled_actions'])}   "
              f"{_f(v['centered_random_actions'])}")
    s = r["root_latents"]["online"]
    print(f"root latents: std median {s['std_quantiles_5_25_50_75_95'][2]:.4f}, frac below floor "
          f"{s['frac_below_variance_floor']:.3f}, pairwise cos {s['mean_pairwise_cosine_similarity']:.3f}; "
          f"input pixel std {r['input_variation']['pixel_std_mean']:.2f}")
    print("depth  reward CE (prior)   +1 P/R (n)          -1 P/R (n)          cont BCE  term P/R (n)   Q huber")
    for (k, rw), cn, q in zip(r["reward_prediction"].items(), r["continuation_prediction"].values(),
                              r["q_td_error"].values()):
        ep, en, tm = rw["event_+1"], rw["event_-1"], cn["terminal"]
        print(f"  {k:<22} {_f(rw['cross_entropy'])} ({rw['prior_cross_entropy']:.3f})  "
              f"{_f(ep['precision'])}/{_f(ep['recall'])} ({ep['events']:>3})  "
              f"{_f(en['precision'])}/{_f(en['recall'])} ({en['events']:>3})  {_f(cn['bce'])}  "
              f"{_f(tm['precision'])}/{_f(tm['recall'])} ({tm['events']})  {_f(q['huber'])}")
    a = r["action_sensitivity_at_root"]
    print(f"action sensitivity: next-latent pairwise dist {a['mean_pairwise_cosine_distance_next_latent']:.4f}, "
          f"E[r] range {a['mean_expected_reward_range']:.4f}, P(cont) range {a['mean_continuation_prob_range']:.4f}")
    print(f"latency: {r['latency']}")
    if "fixed_batch_fit" in r:
        tr = r["fixed_batch_fit"]["trajectory"]
        print(f"fixed-batch fit: {tr[0]} -> {tr[-1]}")


if __name__ == "__main__":
    main()
