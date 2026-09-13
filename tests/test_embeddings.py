"""Latent-space analysis: spectrum, temporal structure, probes, and the evaluation-only state capture."""

import json

import numpy as np
import torch
from test_checkpoint import tiny_config

from atari_jepa.config import EnvConfig
from atari_jepa.embeddings import main as embeddings_main
from atari_jepa.embeddings import spectrum, temporal_structure
from atari_jepa.envs import CatchEnv
from atari_jepa.train import Trainer


def test_spectrum_detects_low_rank_and_full_rank_latents():
    g = torch.Generator().manual_seed(0)
    full = torch.randn(400, 64, generator=g)
    s_full = spectrum(full)
    assert s_full["effective_rank_entropy"] > 40 and s_full["dims_for_90pct"] > 30
    # a latent living in 3 directions: the variance floor per dimension can still be satisfied
    low = torch.randn(400, 3, generator=g) @ torch.randn(3, 64, generator=g)
    s_low = spectrum(low)
    assert s_low["effective_rank_entropy"] < 5 and s_low["dims_for_99pct"] <= 3
    assert s_low["top10_variance_frac"] > 0.99
    constant = torch.ones(50, 8)
    assert spectrum(constant).get("degenerate") is True


def test_temporal_structure_grows_with_gap_for_a_drifting_latent():
    steps = np.tile(np.arange(100), 3)
    episodes = np.repeat(np.arange(3), 100)
    drift = torch.zeros(300, 16)
    drift[:, 0] = torch.from_numpy(steps * 0.1).float()
    drift[:, 1:] = torch.randn(300, 15, generator=torch.Generator().manual_seed(0)) * 0.01
    r = temporal_structure(drift, episodes, steps, gaps=(1, 5, 20))
    d = [r[f"gap_{g}"]["mean_cosine_distance"] for g in (1, 5, 20)]
    assert d[0] < d[1] < d[2], d
    assert r["different_episode"]["pairs"] > 100


def test_state_vector_is_recorded_and_probes_run(tmp_path):
    env = CatchEnv(EnvConfig(id="synthetic:catch", sticky_action_prob=0.0))
    env.reset(seed=0)
    assert env.state_vector().shape == (3,)  # ball row/col and paddle

    run_dir = tmp_path / "run"
    Trainer(tiny_config(run_dir), run_dir, torch.device("cpu")).run(["test"])
    embeddings_main(["--checkpoint", str(run_dir / "checkpoint.pt"), "--episodes", "4",
                     "--max-decisions", "400", "--device", "cpu"])
    r = json.load(open(run_dir / "embeddings.json"))
    assert r["spectrum_online"]["dims"] == 3136
    assert r["state_decodability"]["varying_targets"] == 3
    assert r["state_decodability"]["mean_r2_over_varying"] <= 1.0
    assert r["heldout"]["roots"] > 50
    assert "gap_1" in r["temporal_structure"] and "different_episode" in r["temporal_structure"]
