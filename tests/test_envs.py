"""Environment contract, reset helpers, config strictness, and optional real-ALE checks."""

import numpy as np
import pytest

from atari_jepa.config import EnvConfig, config_from_dict
from atari_jepa.envs import CatchEnv, FrameStacker, clip_reward, find_action, make_env


def test_find_action_uses_meanings_and_refuses_to_guess():
    assert find_action(["NOOP", "FIRE", "RIGHT"], "FIRE") == 1
    assert find_action(["FIRE", "NOOP"], "NOOP") == 1
    with pytest.raises(ValueError):
        find_action(["NOOP", "RIGHT", "LEFT"], "FIRE")


def test_frame_stacker_pads_with_first_observation():
    s = FrameStacker(4)
    a, b = np.full((2, 2), 1, np.uint8), np.full((2, 2), 2, np.uint8)
    assert s.reset(a)[:, 0, 0].tolist() == [1, 1, 1, 1]
    assert s.push(b)[:, 0, 0].tolist() == [1, 1, 1, 2]


def test_reward_clipping_keeps_raw_values_separate():
    assert clip_reward(np.array([-3.0, 0.0, 2.5])).tolist() == [-1.0, 0.0, 1.0]


def test_config_requires_explicit_sticky_actions_and_rejects_unknown_keys():
    with pytest.raises(ValueError):
        EnvConfig()
    with pytest.raises(KeyError):
        config_from_dict({"name": "x"})
    with pytest.raises(KeyError):
        config_from_dict({"env": {"sticky_action_prob": 0.0}, "loss": {"jepaa": True}})
    with pytest.raises(ValueError):
        config_from_dict({"env": {"sticky_action_prob": 0.0, "frame_skip": 4}})
    with pytest.raises(ValueError):
        config_from_dict({"env": {"sticky_action_prob": 0.0, "reward_transform": "none"}})


def test_synthetic_env_contract_and_truncation():
    env = CatchEnv(EnvConfig(id="synthetic:catch", sticky_action_prob=0.0, max_episode_decisions=20))
    f, info = env.reset(seed=0)
    assert f.shape == (84, 84) and f.dtype == np.uint8
    for _ in range(20):
        f, r, term, trunc, _ = env.step(0)
        if term or trunc:
            break
    assert trunc and not term


def ale_available() -> bool:
    try:
        import ale_py  # noqa: F401
        import gymnasium  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not ale_available(), reason="ale-py/gymnasium not installed")
def test_real_pong_contract():
    cfg = EnvConfig(id="ALE/Pong-v5", sticky_action_prob=0.0, max_episode_decisions=5)
    env = make_env(cfg)
    assert env.action_meanings[0] == "NOOP" and "FIRE" in env.action_meanings
    f0, info = env.reset(seed=123)
    assert f0.shape == (84, 84) and f0.dtype == np.uint8
    assert 1 <= info["reset_noops"] <= 30 and info["reset_frames"] == info["reset_noops"]
    frames = []
    for _ in range(5):
        f, r, term, trunc, step_info = env.step(0)
        frames.append(f)
        assert step_info["frames"] == 4
    assert trunc and not term  # external decision limit
    # seeded resets are reproducible
    env2 = make_env(cfg)
    g0, info2 = env2.reset(seed=123)
    assert info2["reset_noops"] == info["reset_noops"] and np.array_equal(f0, g0)
    meta = env.metadata()
    assert meta["preprocessing"]["action_repeat"] == 4 and meta["preprocessing"]["base_frame_skip"] == 1
    env.close()
    env2.close()


@pytest.mark.skipif(not ale_available(), reason="ale-py/gymnasium not installed")
def test_fire_reset_is_counted_in_frames():
    env = make_env(EnvConfig(id="ALE/Pong-v5", sticky_action_prob=0.0, fire_on_reset=True))
    _, info = env.reset(seed=1)
    assert info["reset_fire"] == 1 and info["reset_frames"] == info["reset_noops"] + 1
    env.close()
