"""Single integrator: 2D position model with velocity control."""

import torch
from dynamics.base import BaseDynamics


class SingleIntegrator(BaseDynamics):
    """x_{t+1} = x_t + dt * u_t + D w_t,  D = sigma_w * I.

    State:   [x, y]   (nx=2)
    Control: [vx, vy] (nu=2), saturated via tanh to [-u_max, u_max]
    Noise:   sigma_w on all channels
    """

    def __init__(self, dt: float = 0.1, u_max: float = 1.0, sigma_w: float = 0.05, device: str = "cpu"):
        super().__init__(dt=dt, u_max=u_max, sigma_w=sigma_w, nx=2, nu=2, device=device)

    def _build_matrices(self, sigma_w: float) -> None:
        self.register_buffer("_A", torch.eye(2, device=self.device))
        self.register_buffer("_B", self.dt * torch.eye(2, device=self.device))
        self.register_buffer("_DDT", (sigma_w ** 2) * torch.eye(2, device=self.device))
