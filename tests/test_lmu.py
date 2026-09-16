"""Legendre Memory Unit dynamics: the fixed basis, the threaded state, and what it changes."""

import numpy as np
import pytest
import torch
from conftest import random_batch, world_model_config

from atari_jepa.agent import WorldModel
from atari_jepa.losses import compute_losses, unroll
from atari_jepa.networks import LMUDynamics, legendre_delay_matrices


def lmu_config(**net_overrides):
    cfg = world_model_config()
    cfg.network.dynamics_kind = "lmu"
    cfg.network.lmu_order = 6
    cfg.network.lmu_theta = 5.0
    cfg.network.lmu_signals = 8
    cfg.network.lmu_context = 8
    for k, v in net_overrides.items():
        setattr(cfg.network, k, v)
    return cfg


@pytest.mark.parametrize("theta", [2.0, 5.0, 20.0])
def test_the_legendre_basis_represents_a_constant_as_its_zeroth_coefficient(theta):
    """A constant signal has exactly one non-zero Legendre coefficient, so the LDN's steady state
    under u = 1 must be [1, 0, 0, ...]. This pins down A and B, not just their shapes."""
    order = 6
    A, B = legendre_delay_matrices(order, theta)
    steady = torch.linalg.solve(torch.eye(order) - A, B)
    torch.testing.assert_close(steady, torch.eye(order)[0], atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("theta", [2.0, 5.0, 20.0])
def test_the_memory_is_stable_and_its_window_grows_with_theta(theta):
    A, _ = legendre_delay_matrices(6, theta)
    radius = float(torch.linalg.eigvals(A).abs().max())
    assert radius < 1.0  # a memory that does not blow up over a rollout
    slower, _ = legendre_delay_matrices(6, theta * 4)
    assert float(torch.linalg.eigvals(slower).abs().max()) > radius  # longer window forgets slower


def test_an_impulse_decays_more_slowly_with_a_longer_window():
    def energy_after(steps, theta):
        A, B = legendre_delay_matrices(6, theta)
        m = B.clone()  # one impulse at t=0
        for _ in range(steps):
            m = A @ m
        return float(m.norm())

    assert energy_after(10, 20.0) > energy_after(10, 2.0)


def test_the_basis_is_fixed_and_never_trained():
    cfg = lmu_config()
    dyn = LMUDynamics((8, 7, 7), 4, cfg.network)
    names = {n for n, _ in dyn.named_parameters()}
    assert "lmu_A" not in names and "lmu_B" not in names  # buffers: no gradient, never in the optimizer
    assert {"lmu_A", "lmu_B"} <= {n for n, _ in dyn.named_buffers()}
    z, a = torch.randn(2, 8, 7, 7), torch.zeros(2, dtype=torch.long)
    out, memory = dyn.step(z, a)
    out.sum().backward()
    assert dyn.lmu_A.grad is None and dyn.lmu_B.grad is None
    assert dyn.signal_in.weight.grad.abs().sum() > 0 and dyn.memory_out.weight.grad.abs().sum() > 0
    assert memory.shape == (2, cfg.network.lmu_signals, cfg.network.lmu_order)


def test_the_same_state_and_action_give_a_different_result_after_a_different_history():
    """The point of the unit: step k depends on steps < k, which a memoryless map cannot express."""
    torch.manual_seed(0)
    dyn = LMUDynamics((8, 7, 7), 4, lmu_config().network)
    z = torch.randn(1, 8, 7, 7)
    a0, a1 = torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.long)
    with torch.no_grad():
        _, after_a0 = dyn.step(z, a0)
        _, after_a1 = dyn.step(z, a1)
        out_a0 = dyn.step(z, a0, after_a0)[0]
        out_a1 = dyn.step(z, a0, after_a1)[0]
        fresh = dyn.step(z, a0, None)[0]
    assert (out_a0 - out_a1).abs().max() > 1e-5  # same (z, a), different history
    assert (out_a0 - fresh).abs().max() > 1e-5   # and different from starting with an empty memory


def test_the_rollout_threads_the_memory_instead_of_restarting_each_step():
    torch.manual_seed(0)
    cfg = lmu_config()
    model = WorldModel(cfg, num_actions=4)
    batch = random_batch(A=4)
    with torch.no_grad():
        z0 = model.encoder(batch.observations[:, 0])
        threaded = unroll(model.dynamics, z0, batch.actions, 3)
        stateless = [z0]
        for k in range(3):
            stateless.append(model.dynamics(stateless[k], batch.actions[:, k]))
    torch.testing.assert_close(threaded[1], stateless[1])  # step 1 has no history either way
    assert (threaded[3] - stateless[3]).abs().max() > 1e-5  # deeper steps diverge: the memory is used


def test_a_conv_core_keeps_a_none_state_and_is_unchanged():
    cfg = world_model_config()
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4)
    z = model.encoder(torch.randint(0, 256, (2, 4, 84, 84), dtype=torch.uint8))
    a = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        out, state = model.dynamics.step(z, a, None)
        torch.testing.assert_close(out, model.dynamics(z, a))
    assert state is None


def test_lmu_trains_end_to_end_and_beam_search_carries_the_memory():
    from atari_jepa.planning import beam_search

    cfg = lmu_config()
    torch.manual_seed(0)
    model = WorldModel(cfg, num_actions=4)
    loss, metrics = compute_losses(model, random_batch(A=4), cfg.loss)
    loss.backward()
    assert torch.isfinite(loss) and metrics["loss_jepa"] > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.dynamics.parameters())
    assert model.dynamics.lmu_A.grad is None

    z = model.encoder(torch.randint(0, 256, (1, 4, 84, 84), dtype=torch.uint8))
    action, score = beam_search(model, z, horizon=3, beam_width=4, gamma=0.99)
    assert 0 <= action < 4 and np.isfinite(score)


def test_dynamics_kind_is_validated():
    from atari_jepa.config import NetworkConfig

    assert NetworkConfig(dynamics_kind="lmu").dynamics_kind == "lmu"
    with pytest.raises(ValueError, match="dynamics_kind"):
        NetworkConfig(dynamics_kind="legendre")
    with pytest.raises(ValueError, match="lmu_theta"):
        NetworkConfig(dynamics_kind="lmu", lmu_theta=0.0)


def test_a_conv_pretrain_transfers_its_encoder_into_an_lmu_run_and_skips_the_dynamics(tmp_path, capsys):
    """The offline arms pretrain a conv model; an LMU run must still inherit the *encoder* rather than
    refusing to start, and must say that its dynamics began fresh."""
    from test_checkpoint import tiny_config

    from atari_jepa.train import Trainer

    pre_cfg = tiny_config(tmp_path / "pre")
    torch.manual_seed(0)
    source = Trainer(pre_cfg, tmp_path / "pre", torch.device("cpu"))
    # Mark every source tensor so a transfer is unmistakable: both models seed their submodules the
    # same way, so an untouched tensor can otherwise equal the source by coincidence.
    with torch.no_grad():
        for prm in source.model.parameters():
            prm.add_(7.0)
    source.save()
    pretrained = source.model

    run = tmp_path / "lmu"
    cfg = tiny_config(run)
    cfg.network.dynamics_kind = "lmu"
    cfg.train.init_from = str(tmp_path / "pre" / "checkpoint.pt")
    trainer = Trainer(cfg, run, torch.device("cpu"))
    out = capsys.readouterr().out
    assert "skipped" in out and "dynamics" in out

    assert "encoder" in out and "leaving dynamics" in out

    # The whole dynamics module stays fresh, not just the mismatched tensor: a partly-inherited module
    # would differ from the live arm in a way no config records.
    loaded = dict(trainer.model.dynamics.named_parameters())
    shared = [n for n, pre in pretrained.dynamics.named_parameters()
              if n in loaded and loaded[n].shape == pre.shape]
    assert shared, "no shared-shape dynamics tensor to check; the test would be vacuous"
    for name, pre in pretrained.dynamics.named_parameters():
        if name in loaded and loaded[name].shape == pre.shape:
            assert not torch.equal(loaded[name], pre), f"{name} came from a mismatched dynamics"

    assert isinstance(trainer.model.dynamics, LMUDynamics)
    assert trainer.model.dynamics.conv_in.weight.shape != pretrained.dynamics.conv_in.weight.shape
