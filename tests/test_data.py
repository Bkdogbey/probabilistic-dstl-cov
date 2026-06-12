"""Tests for MovingRectangularObstaclePredicate and lane-merge environment setup."""

import numpy as np
import torch

from pdstl.base import GaussianBelief, BeliefTrajectory
from planning.environment import MovingRectangularObstaclePredicate, Environment

T = 10


def _obs_def(T):
    times = np.arange(T + 1) * 0.1
    return {
        "x_traj": 3.0 + 1.5 * times,
        "y_traj": np.ones(T + 1) * 1.5,
        "width": 2.0,
        "height": 1.5,
    }


def _make_traj(mu_const, var_const, T):
    """Belief trajectory with constant mean [B=1, D=4] and variance."""
    mu = torch.tensor(mu_const, dtype=torch.float32).unsqueeze(0)  # [1, 4]
    var = torch.tensor(var_const, dtype=torch.float32).unsqueeze(0)  # [1, 4]
    return BeliefTrajectory([GaussianBelief(mu, var) for _ in range(T + 1)])


def test_moving_predicate_output_shape():
    """Output must be [B, T+1, 2]."""
    pred = MovingRectangularObstaclePredicate(_obs_def(T))
    traj = _make_traj([0.0, 10.0, 0.0, 0.0], [0.1, 0.1, 0.05, 0.05], T)
    out = pred(traj)
    assert out.shape == (1, T + 1, 2), f"Expected (1,{T + 1},2), got {out.shape}"


def test_moving_predicate_bounds_in_unit_interval():
    """All probability bounds must lie in [0, 1]."""
    pred = MovingRectangularObstaclePredicate(_obs_def(T))
    traj = _make_traj([2.0, 1.5, 0.0, 0.0], [0.1, 0.1, 0.05, 0.05], T)
    out = pred(traj)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_moving_predicate_safe_when_far():
    """Ego far above obstacle → p_safe near 1 at every timestep."""
    pred = MovingRectangularObstaclePredicate(_obs_def(T))
    traj = _make_traj([0.0, 10.0, 0.0, 0.0], [0.01, 0.01, 0.0, 0.0], T)
    out = pred(traj)
    assert torch.all(out[..., 0] > 0.99), f"Expected high p_safe, got min={out[..., 0].min():.4f}"


def test_moving_predicate_unsafe_when_overlapping():
    """Ego mean tracking obstacle centre → p_safe near 0 at every timestep."""
    obs = _obs_def(T)
    pred = MovingRectangularObstaclePredicate(obs)

    beliefs = []
    for t in range(T + 1):
        mu = torch.tensor([[float(obs["x_traj"][t]), float(obs["y_traj"][t]), 0.0, 0.0]])
        var = torch.full((1, 4), 1e-4)
        beliefs.append(GaussianBelief(mu, var))
    traj = BeliefTrajectory(beliefs)

    out = pred(traj)
    assert torch.all(out[..., 0] < 0.1), f"Expected low p_safe, got max={out[..., 0].max():.4f}"


_ROAD = {"y_min": -2.0, "y_max": 6.0, "lane_divider": 2.0}
_OBSTACLE = {"x0": 3.0, "speed": 0.3, "y": 4.0, "width": 2.5, "height": 1.5}


def test_configure_lane_merge_builds_moving_obstacle():
    """configure_lane_change stores T+1 trajectory points and 3 lane markings."""
    env = Environment()
    env.configure_lane_change(
        road=_ROAD,
        obstacle=_OBSTACLE,
        horizon=20,
        dt=0.1,
    )
    assert len(env.moving_obstacles) == 1
    assert len(env.lane_markings) == 3
    traj_len = len(env.moving_obstacles[0]["x_traj"])
    assert traj_len == 21, f"Expected 21 points (T+1=21), got {traj_len}"


def test_configure_lane_merge_obstacle_speed():
    """Obstacle x-position at the last step equals x0 + speed * T * dt."""
    env = Environment()
    env.configure_lane_change(
        road=_ROAD,
        obstacle=_OBSTACLE,
        horizon=20,
        dt=0.1,
    )
    x_final = env.moving_obstacles[0]["x_traj"][-1]
    expected = 3.0 + 0.3 * 20 * 0.1  # 3.0 + 0.6 = 3.6
    assert abs(float(x_final) - expected) < 1e-5, f"Expected {expected}, got {x_final}"


def test_moving_obstacle_in_specification():
    """get_specification() must succeed when a moving obstacle is present."""
    env = Environment()
    env.set_goal([0.0, 200.0], [2.0, 6.0])
    env.set_bounds([-1.0, 15.0], [-2.0, 6.0])
    env.configure_lane_change(
        road=_ROAD,
        obstacle=_OBSTACLE,
        horizon=20,
        dt=0.1,
    )
    spec = env.get_specification(T=20)
    assert spec is not None
