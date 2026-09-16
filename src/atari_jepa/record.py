"""Record gameplay videos from checkpoints, one panel per controller.

    # one checkpoint, one controller
    python -m atari_jepa.record --checkpoint RUN/checkpoint.pt --controller q --out q.mp4

    # two checkpoints side by side, same reset seed, labelled
    python -m atari_jepa.record --out compare.mp4 \
        --panel A_RUN/checkpoint.pt q "A: Q baseline" \
        --panel C_RUN/checkpoint.pt lookahead "C+delta+motion: lookahead"

Every emulator frame is captured (not only decision boundaries), so playback at 60 fps matches the
game's real speed. Panels share the reset seed and the evaluation epsilon; a shorter episode freezes on
its last frame so the longer one plays out. The overlay shows the live score and decision count.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from .envs import FrameStacker, make_env
from .planning import make_controller
from .utils import configure_threads, select_device


def rollout(checkpoint: str, controller_name: str, seed: int, epsilon: float, max_decisions: int,
            device: torch.device, horizon: int | None = None) -> dict[str, Any]:
    """Play one episode, capturing every emulator frame and the score at each frame."""
    ckpt = load_checkpoint(checkpoint, device)
    cfg, model = model_from_checkpoint(ckpt, device)
    env = make_env(cfg.env)
    check_compatible(ckpt["env"], env.metadata())
    frames: list[np.ndarray] = []
    if not hasattr(env, "frame_sink"):
        raise SystemExit(f"{cfg.env.id} does not expose RGB frames for recording")
    env.frame_sink = lambda rgb: frames.append(np.asarray(rgb).copy())
    controller = make_controller(controller_name, model, cfg, device, epsilon, seed=seed, horizon=horizon)
    stacker = FrameStacker(cfg.env.history)
    score_at_frame: list[float] = []
    score, decisions, terminated, truncated = 0.0, 0, False, False
    try:
        obs, _ = env.reset(seed=seed)
        history = stacker.reset(obs)
        score_at_frame = [0.0] * len(frames)
        while decisions < max_decisions:
            action, _ = controller.act(history)
            obs, reward, terminated, truncated, _ = env.step(action)
            history = stacker.push(obs)
            score += reward
            decisions += 1
            score_at_frame.extend([score] * (len(frames) - len(score_at_frame)))
            if terminated or truncated:
                break
    finally:
        env.frame_sink = None
        env.close()
    return {
        "frames": frames,
        "score_at_frame": score_at_frame + [score] * (len(frames) - len(score_at_frame)),
        "decisions_at_frame": np.linspace(0, decisions, num=len(frames)).astype(int),
        "score": score,
        "decisions": decisions,
        "terminated": terminated,
        "truncated": truncated,
        "variant": cfg.loss.variant_name(),
        "controller": controller_name,
    }


def compose(panels: list[dict[str, Any]], labels: list[str], scale: int = 3, header: int = 46) -> list[np.ndarray]:
    """Stack panels horizontally, upscaled, with a label/score header. Shorter panels freeze."""
    import cv2

    length = max(len(p["frames"]) for p in panels)
    h, w = panels[0]["frames"][0].shape[:2]
    out = []
    for i in range(length):
        tiles = []
        for panel, label in zip(panels, labels):
            idx = min(i, len(panel["frames"]) - 1)
            img = cv2.resize(panel["frames"][idx], (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bar = np.zeros((header, w * scale, 3), dtype=np.uint8)
            done = "" if idx < len(panel["frames"]) - 1 else "  (episode over)"
            cv2.putText(bar, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(bar, f"score {panel['score_at_frame'][idx]:.0f}   decision "
                             f"{panel['decisions_at_frame'][idx]}{done}",
                        (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1, cv2.LINE_AA)
            tiles.append(np.vstack([bar, img]))
        gap = np.zeros((tiles[0].shape[0], 6, 3), dtype=np.uint8)
        row = tiles[0]
        for tile in tiles[1:]:
            row = np.hstack([row, gap, tile])
        out.append(row)
    return out


def write_video(frames: list[np.ndarray], path: str | Path, fps: int) -> None:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"could not open a video writer for {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panel", action="append", nargs=3, metavar=("CHECKPOINT", "CONTROLLER", "LABEL"),
                   default=[], help="repeatable: one video panel")
    p.add_argument("--checkpoint", help="single-panel shorthand")
    p.add_argument("--controller", default="q", choices=["q", "sf", "lookahead", "hierarchical"])
    p.add_argument("--label", default=None)
    p.add_argument("--seed", type=int, default=10_000, help="reset seed, shared by all panels")
    p.add_argument("--epsilon", type=float, default=0.01)
    p.add_argument("--max-decisions", type=int, default=3000)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    panels_spec = list(args.panel)
    if args.checkpoint:
        panels_spec.append([args.checkpoint, args.controller, args.label or args.controller])
    if not panels_spec:
        p.error("give --checkpoint or at least one --panel")

    configure_threads(args.threads)
    device = select_device(args.device)
    panels, labels = [], []
    for checkpoint, controller, label in panels_spec:
        r = rollout(checkpoint, controller, args.seed, args.epsilon, args.max_decisions, device, args.horizon)
        print(f"{label}: score {r['score']:+.0f} in {r['decisions']} decisions "
              f"({len(r['frames'])} frames, terminated={r['terminated']}, truncated={r['truncated']})")
        panels.append(r)
        labels.append(label)
    frames = compose(panels, labels, scale=args.scale)
    write_video(frames, args.out, args.fps)
    print(f"-> {args.out} ({len(frames)} frames, {len(frames) / args.fps:.1f}s at {args.fps} fps, "
          f"seed {args.seed}, epsilon {args.epsilon})")


if __name__ == "__main__":
    main()
