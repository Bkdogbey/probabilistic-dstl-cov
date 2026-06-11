import torch
import torch.nn as nn


class DoubleIntegrator(nn.Module):
    """
    Physics-based model with acceleration control and covariance steering.

    State:   [px, py, vx, vy]
    Control: [ax, ay]

    Open-loop:
        mu_{t+1}    = A mu_t + B u_t
        Sigma_{t+1} = A Sigma_t A^T + DDT

    Closed-loop (covariance steering, K != None):
        Sigma_{t+1} = (A + B K_t) Sigma_t (A + B K_t)^T + DDT

    K is optimized directly as the closed-loop gain; it is decoupled from the
    feedforward saturation so that covariance steering works even when V is large.
    """

    def __init__(self, dt=0.1, u_max=2.0, sigma_w=0.1, device="cpu"):
        super().__init__()
        self.dt = dt
        self.u_max = u_max
        self.device = device

        self.A = torch.tensor([
            [1., 0., dt,  0.],
            [0., 1.,  0., dt],
            [0., 0.,  1.,  0.],
            [0., 0.,  0.,  1.],
        ], device=device)

        self.B = torch.tensor([
            [0.5 * dt**2, 0.],
            [0., 0.5 * dt**2],
            [dt, 0.],
            [0., dt],
        ], device=device)

        # Process noise on velocity channels: D = sigma_w * diag([0,0,1,1])
        d = torch.tensor([0., 0., sigma_w, sigma_w], device=device)
        self.DDT = torch.diag(d ** 2)

        self.nx = 4
        self.nu = 2

    def bound_control(self, v):
        """Smooth saturation: v in R -> u in [-u_max, u_max]."""
        return self.u_max * torch.tanh(v)

    def forward(self, v_sequence, x0_mean, x0_cov, K=None):
        """
        Roll out belief trajectory from t=0 to T.

        Args:
            v_sequence: [T, 2]      unconstrained feedforward parameters
            x0_mean:    [4]         initial state mean
            x0_cov:     [4, 4]      initial covariance
            K:          [T, 2, 4]   feedback gains (None = open-loop)

        Returns:
            mean_trace: [1, T+1, 4]
            cov_trace:  [1, T+1, 4, 4]
        """
        T = v_sequence.shape[0]
        mu, Sigma = x0_mean, x0_cov
        means = [mu]
        covs  = [Sigma]

        for t in range(T):
            u = self.bound_control(v_sequence[t])
            mu = self.A @ mu + self.B @ u

            if K is not None:
                A_cl  = self.A + self.B @ K[t]              # [4, 4]
                Sigma = A_cl @ Sigma @ A_cl.T + self.DDT
            else:
                Sigma = self.A @ Sigma @ self.A.T + self.DDT

            means.append(mu)
            covs.append(Sigma)

        return torch.stack(means).unsqueeze(0), torch.stack(covs).unsqueeze(0)
