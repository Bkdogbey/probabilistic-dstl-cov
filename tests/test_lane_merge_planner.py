"""Integration tests for the lane-change planner."""

import torch
import pytest

from dynamics import DoubleIntegrator
from planning.environment import Environment
from planning.planner import ProbabilisticSTLPlanner


def _make_planner(steerer="open_loop", max_iters=50, T=20):
    dyn = DoubleIntegrator(dt=0.1, u_max=2.0, sigma_w=0.1)
    env = Environment()
    env.set_goal([9.0, 11.0], [3.0, 5.5])
    env.set_bounds([-1.0, 11.0], [0.0, 7.0])
    env.add_circle_obstacle([5.0, 1.5], 0.8)
    env.add_circle_obstacle([7.5, 2.5], 0.8)

    mu0    = torch.tensor([0.0, 1.5, 3.0, 0.0])
    Sigma0 = torch.diag(torch.tensor([0.1, 0.1, 0.05, 0.05]))

    planner = ProbabilisticSTLPlanner(dyn, env, T, steerer=steerer,
                                      config={"max_iters": max_iters})
    return planner, mu0, Sigma0, T


def test_open_loop_returns_valid_result():
    planner, mu0, Sigma0, T = _make_planner("open_loop")
    mean_trace, cov_trace, best_u, best_K, best_p, history = planner.solve(mu0, Sigma0, verbose=False)

    assert mean_trace.shape  == (1, T + 1, 4)
    assert cov_trace.shape   == (1, T + 1, 4, 4)
    assert torch.isfinite(mean_trace).all()
    assert torch.isfinite(cov_trace).all()
    assert 0.0 <= best_p <= 1.0


def test_closed_loop_returns_valid_result():
    planner, mu0, Sigma0, T = _make_planner("closed_loop")
    mean_trace, cov_trace, best_u, best_K, best_p, history = planner.solve(mu0, Sigma0, verbose=False)

    assert mean_trace.shape == (1, T + 1, 4)
    assert torch.isfinite(mean_trace).all()
    assert 0.0 <= best_p <= 1.0


def test_open_loop_K_is_none():
    """Open-loop planner returns K=None."""
    planner, mu0, Sigma0, _ = _make_planner("open_loop")
    _, _, _, best_K, _, _ = planner.solve(mu0, Sigma0, verbose=False)
    assert best_K is None


def test_closed_loop_K_receives_gradient():
    """K must receive non-zero gradient in the planner's STL loss.

    We check gradient at initialization (small V, no saturation) to verify
    the covariance steering gradient path is live end-to-end.
    """
    import torch.nn as nn
    from pdstl.base import GaussianBelief, BeliefTrajectory

    dyn = DoubleIntegrator(dt=0.1, u_max=2.0, sigma_w=0.1)
    env = Environment()
    env.set_goal([9.0, 11.0], [3.0, 5.5])
    env.set_bounds([-1.0, 11.0], [0.0, 7.0])
    env.add_circle_obstacle([5.0, 1.5], 0.8)
    env.add_circle_obstacle([7.5, 2.5], 0.8)

    T, mu0 = 20, torch.tensor([0.0, 1.5, 1.0, 0.0])
    Sigma0 = torch.diag(torch.tensor([0.1, 0.1, 0.05, 0.05]))

    V = torch.randn(T, dyn.nu) * 0.1   # small V → sech²(V) ≈ 1, K_eff ≠ 0
    K = nn.Parameter(torch.zeros(T, dyn.nu, dyn.nx))

    spec = env.get_specification(T)
    mean_trace, cov_trace = dyn(V, mu0, Sigma0, K=K)
    beliefs = [GaussianBelief(mean_trace[:, t, :], cov_trace[:, t, :, :])
               for t in range(T + 1)]
    stl_trace = spec(BeliefTrajectory(beliefs))

    loss = -torch.log(stl_trace[0, 0, 1] + 1e-4)  # upper bound for gradient signal
    loss.backward()

    assert K.grad is not None, "K.grad is None"
    assert K.grad.abs().sum() > 0, f"K.grad is all zeros"


def test_covariance_psd_throughout_trajectory():
    """All Σ_t in the returned trajectory must be positive semi-definite."""
    planner, mu0, Sigma0, T = _make_planner("closed_loop")
    _, cov_trace, _, _, _, _ = planner.solve(mu0, Sigma0, verbose=False)
    for t in range(T + 1):
        eigs = torch.linalg.eigvalsh(cov_trace[0, t])
        assert (eigs >= -1e-5).all(), f"Non-PSD covariance at t={t}"


def test_history_nonempty():
    """Loss history must be recorded each iteration."""
    planner, mu0, Sigma0, _ = _make_planner("open_loop")
    _, _, _, _, _, history = planner.solve(mu0, Sigma0, verbose=False)
    assert len(history) > 0
