"""Fixed-batch optimization check and a diagnostics pass on a synthetic checkpoint."""

import json
import math

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
