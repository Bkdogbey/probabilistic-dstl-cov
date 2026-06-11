"""Belief representations for probabilistic STL evaluation."""

from abc import ABC, abstractmethod


class Belief(ABC):
    @abstractmethod
    def value(self): ...

    @abstractmethod
    def probability_of(self, residual): ...


class GaussianBelief(Belief):
    """Gaussian belief wrapping (μ, Σ) tensors.

    Attributes:
        mean_full: [B, D]
        var_full:  [B, D] (diagonal) or [B, D, D] (full)
    """

    def __init__(self, mean_full, var_full):
        self.mean_full = mean_full
        self.var_full = var_full

    def value(self):
        return self.mean_full

    def probability_of(self, residual):
        raise NotImplementedError(
            "GaussianBelief is used with predicates that read mean/var directly."
        )


class BeliefTrajectory:
    """Sequence of beliefs indexed by time."""

    def __init__(self, beliefs):
        self.beliefs = beliefs

    def __getitem__(self, t):
        return self.beliefs[t]

    def __len__(self):
        return len(self.beliefs)

    def suffix(self, t):
        return BeliefTrajectory(self.beliefs[t:])
