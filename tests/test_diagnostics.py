"""Fixed-batch optimization check and a diagnostics pass on a synthetic checkpoint."""

import json
import math

import pytest
import torch
from test_checkpoint import tiny_config

from atari_jepa.checkpoint import load_checkpoint, model_from_checkpoint
from atari_jepa.diagnostics import collect_heldout, fit_fixed_batch
from atari_jepa.diagnostics import main as diagnostics_main
from atari_jepa.train import Trainer


def trained_run(tmp_path):
    run = tmp_path / "run"
    Trainer(tiny_config(run), run, torch.device("cpu")).run(["test"])
    return run


def test_fixed_batch_fit_reduces_prediction_losses_with_fixed_targets(tmp_path):
    run = trained_run(tmp_path)
    ckpt = load_checkpoint(run / "checkpoint.pt")
    cfg, model = model_from_checkpoint(ckpt, torch.device("cpu"))
    replay, _ = collect_heldout(model, cfg, ckpt["env"], torch.device("cpu"), 2, 150, 777, 0.5)
    targets_before = [p.clone() for p in model.target_parameters()]
    cfg.optim.lr = 1e-3
    for comps, key in ((["jepa"], "loss_jepa"), (["reward"], "loss_reward")):
        traj = fit_fixed_batch(model, cfg, replay, torch.device("cpu"), 150, 16, components=comps)["trajectory"]
        assert traj[-1][key] < 0.5 * traj[0][key], (comps, traj[0], traj[-1])
    # fitting works on a copy: the checkpoint model and its targets are untouched
    assert all(torch.equal(a, b) for a, b in zip(targets_before, model.target_parameters()))


def test_diagnostics_cli_reports_finite_metrics(tmp_path):
    run = trained_run(tmp_path)
    diagnostics_main(["--checkpoint", str(run / "checkpoint.pt"), "--episodes", "3", "--max-decisions", "150",
                      "--depths", "1", "3", "5", "--device", "cpu"])
    r = json.load(open(run / "diagnostics.json"))
    assert r["num_roots"] > 100
    lat = r["latent_prediction_cosine_distance"]
    assert lat["depth_5"]["extrapolative"] and not lat["depth_3"]["extrapolative"]  # K=3 in tiny_config
    for v in lat.values():
        for key in ("predicted", "persistence", "shuffled_actions", "centered_predicted"):
            assert v[key] is None or math.isfinite(v[key])
    online = r["root_latents"]["online"]
    assert online["std_mean"] > 0 and online["frac_dims_std_below_1e-3"] < 1.0  # latents are not constant
    assert r["reward_prediction"]["depth_0"]["count"] == r["q_td_error"]["depth_0"]["count"]
    assert math.isfinite(r["q_td_error"]["depth_0"]["huber"])


def test_gradient_reach_measures_credit_travelling_back_through_the_rollout(tmp_path):
    """A residual chain (z' = LayerNorm(z + delta)) must not attenuate credit: the whole point of
    reporting this is that "long rollouts lose gradient" is checkable rather than assumed."""
    import numpy as np

    from atari_jepa.diagnostics import gradient_reach

    run = trained_run(tmp_path)
    ckpt = load_checkpoint(run / "checkpoint.pt")
    cfg, model = model_from_checkpoint(ckpt, torch.device("cpu"))
    device = torch.device("cpu")
    replay, _ = collect_heldout(model, cfg, ckpt["env"], device, 2, 150, 777, 0.5)

    out = gradient_reach(model, cfg, replay, device, batch_size=8, seed=0)
    norms = out["grad_norm_by_step"]
    assert len(norms) == cfg.replay.rollout_steps + 1
    assert all(np.isfinite(n) and n > 0 for n in norms)
    assert out["relative_to_deepest"][cfg.replay.rollout_steps] == pytest.approx(1.0)
    assert out["reach_root_over_deepest"] == pytest.approx(norms[0] / norms[-1])
    # the residual path carries credit all the way back rather than decaying it away
    assert out["reach_root_over_deepest"] > 0.1

    # it leaves no gradient state behind on the model it was given
    assert all(p.grad is None for p in model.parameters())
