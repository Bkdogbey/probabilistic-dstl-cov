"""Shared pytest fixtures."""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dynamics import DoubleIntegrator
from planning.environment import Environment


@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture
def di():
    return DoubleIntegrator(dt=0.1, u_max=2.0, sigma_w=0.1)


@pytest.fixture
def env():
    e = Environment()
    e.set_goal([4.0, 5.0], [0.0, 1.0])
    e.set_bounds([0.0, 6.0], [-1.0, 2.0])
    e.add_circle_obstacle([2.5, 0.5], 0.4)
    return e
