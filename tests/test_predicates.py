"""Tests for probabilistic predicate soundness."""

import torch

from planning.environment import RectangularGoalPredicate
from pdstl.base import GaussianBelief, BeliefTrajectory


def test_goal_leaf_within_unit_interval():
    """Goal predicate bounds must lie in [0,1] with lower <= upper."""
    pred = RectangularGoalPredicate({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    mu = torch.tensor([[0.5, 0.5]])
    var = torch.tensor([[0.04, 0.04]])
    traj = BeliefTrajectory([GaussianBelief(mu, var)])
    out = pred(traj)
    lo, hi = out[..., 0], out[..., 1]
    assert torch.all(lo >= 0) and torch.all(hi <= 1), f"Bounds out of [0,1]: lo={lo}, hi={hi}"
    assert torch.all(lo <= hi), f"Lower bound exceeds upper: lo={lo}, hi={hi}"


def test_goal_leaf_high_confidence():
    """Mean centered in region with tiny variance should give near-1 lower bound."""
    pred = RectangularGoalPredicate({"x": [0.0, 2.0], "y": [0.0, 2.0]})
    mu = torch.tensor([[1.0, 1.0]])
    var = torch.tensor([[1e-4, 1e-4]])
    traj = BeliefTrajectory([GaussianBelief(mu, var)])
    out = pred(traj)
    lo = out[..., 0]
    assert lo.item() > 0.99, f"Expected near-1 lower bound, got {lo.item()}"


def test_goal_leaf_zero_confidence():
    """Mean far outside region should give near-0 lower bound."""
    pred = RectangularGoalPredicate({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    mu = torch.tensor([[10.0, 10.0]])
    var = torch.tensor([[0.01, 0.01]])
    traj = BeliefTrajectory([GaussianBelief(mu, var)])
    out = pred(traj)
    lo = out[..., 0]
    assert lo.item() < 1e-6, f"Expected near-0 lower bound, got {lo.item()}"
