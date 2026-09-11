"""Checkpoint round trip, compatibility checks, resume, and the synthetic end-to-end path."""

import json

import numpy as np
import pytest
import torch

from atari_jepa.checkpoint import check_compatible, load_checkpoint, model_from_checkpoint
from atari_jepa.config import Config, EnvConfig, LossConfig, OptimConfig, ReplayConfig, TrainConfig
from atari_jepa.evaluate import main as evaluate_main
from atari_jepa.train import Trainer, epsilon_at


def tiny_config(run_dir, total=240) -> Config:
    return Config(
        name="ckpt_test",
        seed=3,
        device="cpu",
        run_dir=str(run_dir),
        env=EnvConfig(id="synthetic:catch", sticky_action_prob=0.0, max_episode_decisions=60),
        loss=LossConfig(q_imagined=True, jepa=True, reward=True, continuation=True, variance=True),
        replay=ReplayConfig(capacity=2000, rollout_steps=3, save_with_checkpoint=True),
        optim=OptimConfig(batch_size=4),
        train=TrainConfig(
            total_decisions=total, warmup_decisions=100, update_every=4, eps_decay_decisions=200,
            log_every_updates=10, checkpoint_every_decisions=120, eval_every_decisions=10_000,
            eval_episodes=1, eval_controllers=["q", "lookahead"], eval_max_episode_decisions=30,
        ),
    )


def test_checkpoint_round_trip(tmp_path):
    cfg = tiny_config(tmp_path / "run")
    trainer = Trainer(cfg, tmp_path / "run", torch.device("cpu"))
    trainer.run(["test"])
    ckpt = load_checkpoint(tmp_path / "run" / "checkpoint.pt")

    # a fresh trainer restored from the checkpoint has identical state
    restored = Trainer(cfg, tmp_path / "run", torch.device("cpu"), resume=ckpt)
    obs = torch.randint(0, 256, (5, 4, 84, 84), dtype=torch.uint8)
    acts = torch.randint(0, 4, (5,))
    with torch.no_grad():
        for m1, m2 in ((trainer.model, restored.model),):
            z1, z2 = m1.encoder(obs), m2.encoder(obs)
            torch.testing.assert_close(z1, z2, rtol=0, atol=0)
            torch.testing.assert_close(m1.dynamics(z1, acts), m2.dynamics(z2, acts), rtol=0, atol=0)
            torch.testing.assert_close(m1.reward_head(z1, acts), m2.reward_head(z2, acts), rtol=0, atol=0)
            torch.testing.assert_close(m1.q_head(z1), m2.q_head(z2), rtol=0, atol=0)
    for p1, p2 in zip(trainer.model.target_parameters(), restored.model.target_parameters()):
        assert torch.equal(p1, p2)
    s1, s2 = trainer.learner.optimizer.state_dict(), restored.learner.optimizer.state_dict()
    assert s1["param_groups"] == s2["param_groups"]
    for k in s1["state"]:
        for name, v in s1["state"][k].items():
            assert torch.equal(v, s2["state"][k][name])
    assert ckpt["config"] == cfg.to_dict()
    for key in ("decisions", "updates", "episodes", "sampled_valid_transitions", "warmup_until"):
        assert restored.counters[key] == trainer.counters[key]
    assert restored.counters["resume_count"] == 1
    assert ckpt["schedule"]["epsilon_now"] == epsilon_at(cfg.train.total_decisions, cfg.train)
    assert trainer.explore_rng.random() == restored.explore_rng.random()
    assert restored.replay.n_written == trainer.replay.n_written
    np.testing.assert_array_equal(restored.replay.valid_roots(), trainer.replay.valid_roots())
    assert ckpt["env"]["action_meanings"] == ["NOOP", "FIRE", "RIGHT", "LEFT"]

    # the loaded model for evaluation matches as well
    _, loaded = model_from_checkpoint(ckpt, torch.device("cpu"))
    with torch.no_grad():
        torch.testing.assert_close(loaded.q_values(obs), trainer.model.q_values(obs), rtol=0, atol=0)


def test_training_resume_continues_counters_and_logs(tmp_path):
    run = tmp_path / "run"
    Trainer(tiny_config(run, total=240), run, torch.device("cpu")).run(["test"])
    ckpt = load_checkpoint(run / "checkpoint.pt")
    cfg = tiny_config(run, total=360)
    t = Trainer(cfg, run, torch.device("cpu"), resume=ckpt)
    assert t.counters["warmup_until"] == 100  # replay persisted -> no new warmup
    t.run(["test", "--resume"])
    assert t.counters["decisions"] == 360
    assert t.counters["updates"] == len(range(104, 361, 4))
    assert (run / "resumes.jsonl").exists()
    lines = [json.loads(line) for line in open(run / "updates.jsonl")]
    assert all(np.isfinite(v) for rec in lines for v in rec.values() if isinstance(v, float))


def test_resume_trims_logs_written_after_the_checkpoint(tmp_path):
    run = tmp_path / "run"
    Trainer(tiny_config(run), run, torch.device("cpu")).run(["test"])
    with open(run / "train_episodes.jsonl", "a") as fh:  # simulate records written after a crash point
        fh.write(json.dumps({"episode": 999, "decisions": 10_000}) + "\n")
    Trainer(tiny_config(run, total=300), run, torch.device("cpu"), resume=load_checkpoint(run / "checkpoint.pt"))
    records = [json.loads(line) for line in open(run / "train_episodes.jsonl")]
    assert records and all(r["decisions"] <= 240 for r in records)


def test_resume_without_replay_reenters_warmup(tmp_path):
    run = tmp_path / "run"
    Trainer(tiny_config(run), run, torch.device("cpu")).run(["test"])
    (run / "replay.npz").unlink()
    t = Trainer(tiny_config(run, total=400), run, torch.device("cpu"), resume=load_checkpoint(run / "checkpoint.pt"))
    assert len(t.replay) == 0 and t.counters["warmup_until"] == 240 + 100


def test_compatibility_checks():
    meta = {"id": "ALE/Pong-v5", "action_meanings": ["NOOP", "FIRE"], "observation_shape": [4, 84, 84],
            "preprocessing": {"reward_transform": "sign", "action_repeat": 4}}
    check_compatible(meta, json.loads(json.dumps(meta)))
    for key, value in (("id", "ALE/Breakout-v5"), ("action_meanings", ["NOOP"]), ("observation_shape", [4, 64, 64])):
        with pytest.raises(ValueError):
            check_compatible(meta, {**meta, key: value})
    with pytest.raises(ValueError):
        check_compatible(meta, {**meta, "preprocessing": {"reward_transform": "none", "action_repeat": 4}})


def test_evaluate_cli_on_synthetic_checkpoint(tmp_path, capsys):
    run = tmp_path / "run"
    Trainer(tiny_config(run), run, torch.device("cpu")).run(["test"])
    before = load_checkpoint(run / "checkpoint.pt")["model"]
    evaluate_main(["--checkpoint", str(run / "checkpoint.pt"), "--controller", "both", "--episodes", "2",
                   "--device", "cpu", "--max-episode-decisions", "40"])
    q = json.load(open(run / "eval_q.json"))
    la = json.load(open(run / "eval_lookahead.json"))
    assert [r["reset_seed"] for r in q["records"]] == [r["reset_seed"] for r in la["records"]]
    assert 0 <= la["controller_stats"]["disagreement_with_q"] <= 1
    after = load_checkpoint(run / "checkpoint.pt")["model"]
    assert all(torch.equal(before[k], after[k]) for k in before)  # evaluation never modifies it
