"""Double integrator: 2D position-velocity model with acceleration control."""

import torch
from dynamics.base import BaseDynamics


class DoubleIntegrator(BaseDynamics):
    """x_{t+1} = A x_t + B u_t + D w_t,  D = sigma_w * diag([0,0,1,1]).

    State:   [x, y, vx, vy]  (nx=4)
    Control: [ax, ay]         (nu=2), saturated via tanh to [-u_max, u_max]
    Noise:   sigma_w on velocity channels only
    """

    def __init__(
        self, dt: float = 0.1, u_max: float = 2.0, sigma_w: float = 0.1, device: str = "cpu"
    ) -> None:
        super().__init__(dt=dt, u_max=u_max, sigma_w=sigma_w, nx=4, nu=2, device=device)

    def _build_matrices(self, sigma_w: float) -> None:
        dt = self.dt
        self.register_buffer(
            "_A",
            torch.tensor(
                [
                    [1.0, 0.0, dt, 0.0],
                    [0.0, 1.0, 0.0, dt],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device=self.device,
            ),
        )

        self.register_buffer(
            "_B",
            torch.tensor(
                [
                    [0.5 * dt**2, 0.0],
                    [0.0, 0.5 * dt**2],
                    [dt, 0.0],
                    [0.0, dt],
                ],
                device=self.device,
            ),
        )

        d = torch.tensor([0.0, 0.0, sigma_w, sigma_w], device=self.device)
        self.register_buffer("_DDT", torch.diag(d**2))
