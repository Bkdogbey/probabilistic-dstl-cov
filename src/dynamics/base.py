"""Abstract base class for discrete-time stochastic dynamics models."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


class BaseDynamics(nn.Module, ABC):
    """
    Discrete-time linear-Gaussian dynamics model.

    Subclasses implement `_build_matrices` to register system matrices
    `_A`, `_B`, `_DDT` as buffers; this class provides the common
    belief rollout (`forward`) and control saturation (`bound_control`).

    State evolution:
        Open-loop:   mu_{t+1}    = A mu_t + B u_t
                     Sigma_{t+1} = A Sigma_t A^T + DDT
        Closed-loop: Sigma_{t+1} = (A + B K_t) Sigma_t (A + B K_t)^T + DDT
    """

    def __init__(
        self,
        dt: float = 0.1,
        u_max: float = 2.0,
        sigma_w: float = 0.1,
        nx: int = 4,
        nu: int = 2,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.dt = dt
        self.u_max = u_max
        self.nx = nx
        self.nu = nu
        self.device = device
        self._build_matrices(sigma_w)

    @abstractmethod
    def _build_matrices(self, sigma_w: float) -> None:
        """Register system matrices _A [nx,nx], _B [nx,nu], _DDT [nx,nx] as buffers."""

    def saturate(self, v: torch.Tensor) -> torch.Tensor:
        """Smooth saturation u = u_max * tanh(v / u_max). Slope 1 at origin."""
        if self.u_max is None:
            return v
        return self.u_max * torch.tanh(v / self.u_max)

    def saturation_jacobian(self, v: torch.Tensor) -> torch.Tensor:
        """Diagonal S_k = diag(1 - tanh²(v / u_max)) = diag(sech²(v / u_max)).
        Uses 1 - tanh² (numerically stable) rather than 1/cosh² (overflows for large v)."""
        if self.u_max is None:
            return torch.eye(self.nu, dtype=v.dtype, device=v.device)
        z = v / self.u_max
        return torch.diag(1.0 - torch.tanh(z).pow(2))

    def bound_control(self, v: torch.Tensor) -> torch.Tensor:
        """Alias for saturate (backwards compatibility)."""
        return self.saturate(v)

    def forward(
        self,
        v_sequence: torch.Tensor,
        x0_mean: torch.Tensor,
        x0_cov: torch.Tensor,
        K: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Roll out belief trajectory from t=0 to T.

        Args:
            v_sequence: [T, nu]      unconstrained feedforward parameters
            x0_mean:    [nx]         initial state mean
            x0_cov:     [nx, nx]     initial covariance
            K:          [T, nu, nx]  feedback gains (None = open-loop)

        Returns:
            mean_trace: [1, T+1, nx]
            cov_trace:  [1, T+1, nx, nx]
        """
        T = v_sequence.shape[0]
        mu, Sigma = x0_mean, x0_cov
        means = [mu]
        covs = [Sigma]

        for t in range(T):
            v_t = v_sequence[t]
            u = self.saturate(v_t)
            mu = self._A @ mu + self._B @ u

            if K is not None:
                S_t = self.saturation_jacobian(v_t)
                A_cl = self._A + self._B @ S_t @ K[t]
                Sigma = A_cl @ Sigma @ A_cl.T + self._DDT
            else:
                Sigma = self._A @ Sigma @ self._A.T + self._DDT

            means.append(mu)
            covs.append(Sigma)

        return torch.stack(means).unsqueeze(0), torch.stack(covs).unsqueeze(0)
