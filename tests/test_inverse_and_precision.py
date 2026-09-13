"""Inverse-dynamics loss, bfloat16 conv trunks, the movement probe, and old-config compatibility."""

import copy
import json

import numpy as np
import pytest
import torch
from conftest import random_batch, world_model_config
from test_checkpoint import tiny_config
from test_gradients import no_grad, nonzero_grad
from test_masks import run

from atari_jepa.agent import WorldModel, trained_modules
from atari_jepa.config import config_from_dict
from atari_jepa.diagnostics import main as diagnostics_main
from atari_jepa.diagnostics import movement_classes, movement_probe
from atari_jepa.losses import compute_losses
from atari_jepa.train import Trainer


def make(**loss):
    cfg = world_model_config(**loss)
    torch.manual_seed(0)
    return cfg, WorldModel(cfg, num_actions=4)


def test_inverse_real_trains_encoder_not_dynamics():
    cfg, model = make(q_imagined=False, jepa=False, reward=False, continuation=False, variance=False, inverse="real")
    loss, m = compute_losses(model, random_batch(), cfg.loss)
    loss.backward()
    assert m["loss_inverse"] > 0 and 0 <= m["inverse_acc"] <= 1
    assert nonzero_grad(model.encoder) and nonzero_grad(model.inverse_head)
    assert no_grad(model.dynamics)
    assert all(p.grad is None for p in model.target_parameters())
    assert "inverse_head" in trained_modules(cfg)


def test_inverse_predicted_trains_dynamics():
    cfg, model = make(q_imagined=False, jepa=False, reward=False, continuation=False, variance=False,
                      inverse="predicted")
    loss, _ = compute_losses(model, random_batch(), cfg.loss)
    loss.backward()
    assert nonzero_grad(model.dynamics) and nonzero_grad(model.encoder) and nonzero_grad(model.inverse_head)


@pytest.mark.parametrize("inverse", ["real", "predicted"])
def test_padding_invariance_with_inverse(inverse):
    cfg, model = make(inverse=inverse)
    batch = random_batch()
    loss_a, m_a, g_a = run(model, batch, cfg)
    other = copy.deepcopy(batch)
    obs_valid = torch.cat([torch.ones_like(other.valid[:, :1]), other.valid], dim=1)
    noise = torch.randint(0, 256, other.observations.shape, dtype=torch.uint8)
    other.observations = torch.where(obs_valid[..., None, None, None], other.observations, noise)
    other.actions = torch.where(other.valid, other.actions, torch.full_like(other.actions, 3))
    loss_b, m_b, g_b = run(model, other, cfg)
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=1e-6)
    assert abs(m_a["loss_inverse"] - m_b["loss_inverse"]) < 1e-6
    for name in g_a:
        torch.testing.assert_close(g_a[name], g_b[name], rtol=1e-5, atol=1e-7, msg=name)


def test_old_configs_without_new_fields_still_load():
    cfg, model = make()
    data = cfg.to_dict()
    for key in ("inverse", "lambda_inverse"):
        data["loss"].pop(key)
    for key in ("inverse_hidden", "conv_dtype"):
        data["network"].pop(key)
    old = config_from_dict(data)
    assert old.loss.inverse == "none" and old.network.conv_dtype == "float32"
    reloaded = WorldModel(old, num_actions=4)
    reloaded.load_state_dict(model.state_dict())  # identical layout: no inverse head is built
    assert reloaded.inverse_head is None


def bf16_training_devices() -> list[str]:
    from atari_jepa.agent import check_precision_supported

    cfg = world_model_config()
    cfg.network.conv_dtype = "bfloat16"
    devices = []
    for name in ("cpu", "cuda"):
        if name == "cuda" and not torch.cuda.is_available():
            continue
        try:
            check_precision_supported(cfg, torch.device(name))
            devices.append(name)
        except RuntimeError:
            pass
    return devices


def test_bfloat16_forward_matches_float32_and_keeps_float32_latents():
    cfg32, m32 = make(inverse="real")
    cfg16 = copy.deepcopy(cfg32)
    cfg16.network.conv_dtype = "bfloat16"
    m16 = WorldModel(cfg16, num_actions=4)
    m16.load_state_dict(m32.state_dict())
    batch = random_batch()
    with torch.no_grad():
        z32, z16 = m32.encoder(batch.observations[:, 0]), m16.encoder(batch.observations[:, 0])
        assert z16.dtype == torch.float32
        torch.testing.assert_close(z16, z32, rtol=0, atol=0.1)
        n32, n16 = m32.dynamics(z32, batch.actions[:, 0]), m16.dynamics(z32, batch.actions[:, 0])
        assert n16.dtype == torch.float32
        torch.testing.assert_close(n16, n32, rtol=0, atol=0.1)
        _, metrics = compute_losses(m16, batch, cfg16.loss)
    assert all(np.isfinite(v) for v in metrics.values())


@pytest.mark.parametrize("device", bf16_training_devices() or [pytest.param("none", marks=pytest.mark.skip(
    reason="no device here supports bfloat16 conv backward (CPU oneDNN on this platform does not)"))])
def test_bfloat16_training_step(device):
    cfg32, m32 = make(inverse="real")
    cfg16 = copy.deepcopy(cfg32)
    cfg16.network.conv_dtype = "bfloat16"
    m16 = WorldModel(cfg16, num_actions=4).to(device)
    m16.load_state_dict(m32.state_dict())
    b = random_batch()
    batch = type(b)(*(getattr(b, f).to(device) for f in
                      ("observations", "actions", "rewards", "terminated", "truncated", "valid")))
    loss, metrics = compute_losses(m16, batch, cfg16.loss)
    loss.backward()
    assert torch.isfinite(loss) and all(np.isfinite(v) for v in metrics.values())
    assert all(p.grad is None or p.grad.dtype == torch.float32 for p in m16.parameters())
    assert nonzero_grad(m16.encoder) and nonzero_grad(m16.dynamics)


def test_movement_classes_and_probe_on_separable_data():
    assert movement_classes(["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]).tolist() == [0, 0, 1, 2, 1, 2]
    g = torch.Generator().manual_seed(0)
    n, d = 600, 20
    labels = np.random.default_rng(0).integers(0, 3, n)
    z = torch.randn(n, d, generator=g)
    z_next = z.clone()
    z_next[:, 0] += torch.tensor([0.0, 1.0, -1.0])[labels]  # the "paddle" moves with the action
    groups = np.repeat(np.arange(4), n // 4)
    r = movement_probe(z, z_next, labels, groups)
    assert r["test_acc"] > 0.95 and r["test_samples"] == n // 2
    shuffled = movement_probe(z, z_next, np.random.default_rng(1).permutation(labels), groups)
    assert shuffled["test_balanced_acc"] < 0.5


def test_diagnostics_report_inverse_head_and_probe(tmp_path):
    run_dir = tmp_path / "run"
    cfg = tiny_config(run_dir)
    cfg.loss.inverse = "real"
    Trainer(cfg, run_dir, torch.device("cpu")).run(["test"])
    diagnostics_main(["--checkpoint", str(run_dir / "checkpoint.pt"), "--episodes", "3", "--max-decisions", "150",
                      "--depths", "1", "3", "--device", "cpu"])
    r = json.load(open(run_dir / "diagnostics.json"))
    assert r["variant"] == "C_world_model+inverse_real"
    assert 0 <= r["inverse_head"]["action_acc_real_pairs"] <= 1
    assert "test_acc" in r["movement_probe"]


def test_learner_single_transfer_still_rejects_bad_rewards():
    from atari_jepa.agent import Learner

    cfg, model = make()
    learner = Learner(model, cfg)
    metrics = learner.update(random_batch())
    assert isinstance(metrics["loss_total"], float) and "_reward_out_of_range" not in metrics
    bad = random_batch()
    bad.rewards[0, 0] = 2.0  # valid step with an unclipped reward
    with pytest.raises(ValueError, match="outside"):
        learner.update(bad)
