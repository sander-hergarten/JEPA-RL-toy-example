"""Video composition and writing (no ROM or checkpoint needed)."""

import cv2
import numpy as np

from atari_jepa.record import compose, write_video


def fake_panel(n_frames, score):
    return {
        "frames": [np.full((210, 160, 3), i % 255, dtype=np.uint8) for i in range(n_frames)],
        "score_at_frame": list(np.linspace(0, score, n_frames)),
        "decisions_at_frame": np.linspace(0, n_frames // 4, n_frames).astype(int),
        "score": score,
    }


def test_compose_pads_the_shorter_panel_and_stacks_horizontally():
    short, long = fake_panel(10, 3), fake_panel(25, 7)
    frames = compose([short, long], ["A", "B"], scale=2, header=40)
    assert len(frames) == 25  # the longer episode plays out
    h, w = frames[0].shape[:2]
    assert h == 210 * 2 + 40 and w == 2 * 160 * 2 + 6  # two panels, header, separator
    # after the short panel ends it freezes: its half stops changing
    left_last = frames[-1][40:, : 160 * 2]
    left_at_end_of_short = frames[10][40:, : 160 * 2]
    assert np.array_equal(left_last, left_at_end_of_short)


def test_write_video_produces_a_readable_file(tmp_path):
    frames = compose([fake_panel(12, 2)], ["solo"], scale=1)
    path = tmp_path / "out.mp4"
    write_video(frames, path, fps=30)
    assert path.stat().st_size > 0
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == frames[0].shape[:2]
