"""Tests for visualization.animation: animate_trajectory and the live callback."""

import torch
import matplotlib

matplotlib.use("Agg")  # headless — no display needed

from visualization.animation import animate_trajectory
from planning.environment import Environment
from planning.planner import ProbabilisticSTLPlanner
from dynamics import DoubleIntegrator


T_TEST = 5
NX = 4


def _make_traces(T=T_TEST):
    """Return (mean_trace [1, T+1, 4], cov_trace [1, T+1, 4, 4]) with simple linear ramp."""
    mu = torch.zeros(1, T + 1, NX)
    for t in range(T + 1):
        mu[0, t, 0] = float(t)  # x increases linearly
    cov = torch.eye(NX).unsqueeze(0).unsqueeze(0).expand(1, T + 1, -1, -1) * 0.01
    return mu, cov


def _simple_env():
    env = Environment()
    env.set_goal([4.0, 6.0], [0.0, 1.0])
    env.set_bounds([0.0, 7.0], [-1.0, 2.0])
    return env


# ---------------------------------------------------------------------------
# animate_trajectory
# ---------------------------------------------------------------------------


def test_animate_trajectory_creates_gif(tmp_path):
    """`animate_trajectory` must write a non-empty GIF file."""
    mu, cov = _make_traces()
    env = _simple_env()
    out = tmp_path / "out.gif"
    animate_trajectory(mu, cov, env, filename=str(out), fps=5)
    assert out.exists(), "GIF file was not created"
    assert out.stat().st_size > 0, "GIF file is empty"


def test_animate_trajectory_accepts_squeezed_inputs(tmp_path):
    """[T+1, nx] and [T+1, nx, nx] inputs (no batch dim) must not raise."""
    mu, cov = _make_traces()
    env = _simple_env()
    out = tmp_path / "out2.gif"
    animate_trajectory(mu.squeeze(0), cov.squeeze(0), env, filename=str(out), fps=5)
    assert out.exists()


def test_animate_trajectory_with_moving_obstacle(tmp_path):
    """animate_trajectory correctly handles a moving obstacle in env."""
    import numpy as np

    mu, cov = _make_traces()
    env = _simple_env()
    times = np.arange(T_TEST + 1) * 0.1
    env.add_moving_obstacle(
        x_traj=1.0 + 0.3 * times,
        y_traj=np.ones(T_TEST + 1) * 4.0,
        width=1.0,
        height=0.8,
    )
    out = tmp_path / "moving.gif"
    animate_trajectory(mu, cov, env, filename=str(out), fps=5)
    assert out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# make_live_callback — verify it fires without error in headless mode
# ---------------------------------------------------------------------------


def test_callback_fires_during_solve():
    """solve(callback=cb) must invoke cb at least once."""
    dyn = DoubleIntegrator(dt=0.1, u_max=2.0, sigma_w=0.1)
    env = Environment()
    env.set_goal([0.5, 2.0], [0.5, 2.0])
    env.set_bounds([0.0, 3.0], [0.0, 3.0])

    planner = ProbabilisticSTLPlanner(
        dyn,
        env,
        T=5,
        steerer="open_loop",
        config={"max_iters": 30, "warmup_iters": 0},
    )
    mu0 = torch.zeros(4)
    S0 = torch.eye(4) * 0.01

    call_log = []

    def cb(k, mean_trace, cov_trace, loss, p_lower):
        call_log.append((k, float(p_lower)))

    planner.solve(mu0, S0, verbose=False, callback=cb, callback_every=5)
    assert len(call_log) > 0, "callback was never called"
    assert all(0.0 <= p <= 1.0 for _, p in call_log), "p_lower out of [0,1]"


def test_callback_every_respected():
    """callback_every=10 means iterations 0, 10, 20, ... are reported."""
    dyn = DoubleIntegrator(dt=0.1, u_max=2.0, sigma_w=0.1)
    env = Environment()
    env.set_goal([0.5, 2.0], [0.5, 2.0])
    env.set_bounds([0.0, 3.0], [0.0, 3.0])

    planner = ProbabilisticSTLPlanner(
        dyn,
        env,
        T=5,
        steerer="open_loop",
        config={"max_iters": 55, "warmup_iters": 0},
    )
    mu0 = torch.zeros(4)
    S0 = torch.eye(4) * 0.01

    call_log = []
    planner.solve(
        mu0,
        S0,
        verbose=False,
        callback=lambda k, *_: call_log.append(k),
        callback_every=10,
    )
    assert all(k % 10 == 0 for k in call_log), f"unexpected iterations: {call_log}"
