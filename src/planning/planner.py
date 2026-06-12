"""Gradient-based probabilistic STL planner with open- and closed-loop covariance steering."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from pdstl.base import GaussianBelief, BeliefTrajectory


class ProbabilisticSTLPlanner:
    """
    Gradient-based motion planner with covariance steering.

    Open-loop  (steerer='open_loop'):
        Optimizes V [T, nu] (feedforward controls).
        Covariance evolves as Sigma_{t+1} = A Sigma_t A^T + DDT.

    Closed-loop (steerer='closed_loop'):
        Optimizes V [T, nu] and K [T, nu, nx] jointly.
        Covariance is steered via (A + B K_eff_t) Sigma (A + B K_eff_t)^T + DDT.
    """

    def __init__(
        self,
        dynamics,
        environment,
        T: int,
        steerer: str = "open_loop",
        config: Optional[dict] = None,
    ) -> None:
        self.dyn    = dynamics
        self.env    = environment
        self.T      = T
        self.device = dynamics.device
        self.steerer = steerer

        self.cfg = {
            "w_phi":      10.0,  # STL satisfaction weight
            "w_du":        0.001, # control smoothness
            "w_dist":      0.1,   # goal distance heuristic
            "w_obs":       5.0,   # obstacle repulsion heuristic
            "w_K":         0.005, # feedback gain regularization (closed-loop only)
            "w_cov":       0.1,   # terminal covariance penalty (steers K to reduce Σ_T)
            "lr_v":        0.05,  # learning rate for V
            "lr_k":        0.01,  # learning rate for K (closed-loop only)
            "max_iters":   500,
            "warmup_iters": 100,  # iters to run V-only before K optimization starts
            "alpha":       0.95,  # satisfaction threshold for early stop
        }
        if config:
            self.cfg.update(config)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rollout(
        self,
        V: nn.Parameter,
        x0_mean: torch.Tensor,
        x0_cov: torch.Tensor,
        K: Optional[nn.Parameter],
        warmup_active: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Roll out the belief trajectory for one optimization step.

        Returns:
            mean_trace: [1, T+1, nx]
            cov_trace:  [1, T+1, nx, nx]
            u_seq:      [T, nu]  bounded controls
        """
        K_rollout = K.detach() if (K is not None and warmup_active) else K
        mean_trace, cov_trace = self.dyn(V, x0_mean, x0_cov, K=K_rollout)
        u_seq = self.dyn.bound_control(V)
        return mean_trace, cov_trace, u_seq

    def _build_belief_trajectory(
        self,
        mean_trace: torch.Tensor,
        cov_trace: torch.Tensor,
    ) -> BeliefTrajectory:
        """Wrap mean/covariance traces into a BeliefTrajectory for STL evaluation."""
        beliefs = [
            GaussianBelief(mean_trace[:, t, :], cov_trace[:, t, :, :])
            for t in range(self.T + 1)
        ]
        return BeliefTrajectory(beliefs)

    def _compute_loss(
        self,
        u_seq: torch.Tensor,
        mean_trace: torch.Tensor,
        cov_trace: torch.Tensor,
        stl_trace: torch.Tensor,
        K: Optional[nn.Parameter],
    ) -> Tuple[torch.Tensor, float, float]:
        """Compute the weighted multi-term loss.

        Returns:
            J:       scalar loss tensor (differentiable)
            p_upper: upper-bound satisfaction probability (for tracking)
            p_sat:   lower-bound satisfaction probability (conservative)
        """
        # Use upper bound for gradient signal: Frechet lower bound clips to 0
        # when individual sub-formula probs are low, zeroing K's gradient.
        p_upper = stl_trace[0, 0, 1]
        p_sat   = stl_trace[0, 0, 0]

        loss_phi = -torch.log(p_upper + 1e-4)
        loss_du  = torch.sum((u_seq[1:] - u_seq[:-1]) ** 2) + torch.sum(u_seq[0] ** 2)

        loss_dist = torch.tensor(0.0, device=self.device)
        if self.env.goal is not None:
            gx = sum(self.env.goal["x"]) / 2.0
            gy = sum(self.env.goal["y"]) / 2.0
            goal_center = torch.tensor([[gx, gy]], device=self.device)
            loss_dist = torch.sum((mean_trace[:, -1, :2] - goal_center) ** 2)

        loss_obs = torch.tensor(0.0, device=self.device)
        for obs in self.env.obstacles:
            cx = (obs["x"][0] + obs["x"][1]) / 2.0
            cy = (obs["y"][0] + obs["y"][1]) / 2.0
            r  = max(obs["x"][1] - obs["x"][0], obs["y"][1] - obs["y"][0]) / 2.0 + 0.5
            center_t = torch.tensor([[cx, cy]], device=self.device)
            dists = torch.norm(mean_trace[:, :, :2] - center_t, dim=2)
            loss_obs += torch.sum(torch.relu(r - dists) ** 2)
        for obs in self.env.circle_obstacles:
            center = torch.tensor([obs["center"]], device=self.device)
            r = obs["radius"] + 0.5
            dists = torch.norm(mean_trace[:, :, :2] - center, dim=2)
            loss_obs += torch.sum(torch.relu(r - dists) ** 2)

        loss_K = (
            torch.sum(K ** 2) if K is not None else torch.tensor(0.0, device=self.device)
        )
        # Directly rewards covariance reduction, preventing K from growing in
        # directions that blow up the terminal covariance.
        loss_cov = torch.trace(cov_trace[0, -1])

        J = (
            self.cfg["w_phi"]  * loss_phi
            + self.cfg["w_du"] * loss_du
            + self.cfg["w_dist"] * loss_dist
            + self.cfg["w_obs"] * loss_obs
            + self.cfg["w_K"]  * loss_K
            + self.cfg["w_cov"] * loss_cov
        )
        return J, p_upper, p_sat

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(
        self, x0_mean: torch.Tensor, x0_cov: torch.Tensor, verbose: bool = True, spec=None
    ) -> Tuple:
        """
        Run optimization to find V (and K if closed-loop).

        Args:
            x0_mean: [nx]      initial state mean
            x0_cov:  [nx, nx]  initial covariance
            verbose: print progress every 50 iterations
            spec:    optional pre-built STL formula; built from env if None

        Returns:
            mean_trace:  [1, T+1, nx]   best trajectory mean
            cov_trace:   [1, T+1, nx, nx] best covariance trace
            best_u:      [T, nu]         best bounded controls
            best_K:      [T, nu, nx] or None
            best_p:      float           best P(phi) achieved
            history:     list[float]     loss per iteration
        """
        if spec is None:
            spec = self.env.get_specification(self.T)

        V = nn.Parameter(torch.randn(self.T, self.dyn.nu, device=self.device) * 0.1)

        if self.steerer == "closed_loop":
            K = nn.Parameter(
                torch.zeros(self.T, self.dyn.nu, self.dyn.nx, device=self.device)
            )
            optimizer = optim.Adam([
                {"params": V, "lr": self.cfg["lr_v"]},
                {"params": K, "lr": self.cfg["lr_k"]},
            ])
        else:
            K = None
            optimizer = optim.Adam([V], lr=self.cfg["lr_v"])

        best_p    = -1.0
        best_u    = None
        best_K    = None
        best_mean = None
        best_cov  = None
        history   = []
        converged_iters = 0
        warmup = self.cfg["warmup_iters"]

        if verbose:
            print(
                f"Starting optimization ({self.steerer}, max_iters={self.cfg['max_iters']})..."
            )

        for k in range(self.cfg["max_iters"]):
            optimizer.zero_grad()

            mean_trace, cov_trace, u_seq = self._rollout(
                V, x0_mean, x0_cov, K, warmup_active=(k < warmup)
            )
            traj      = self._build_belief_trajectory(mean_trace, cov_trace)
            stl_trace = spec(traj)
            J, p_upper, _ = self._compute_loss(
                u_seq, mean_trace, cov_trace, stl_trace, K
            )

            J.backward()
            if K is not None and k < warmup:
                K.grad = None  # keep K frozen during warmup
            optimizer.step()

            current_p = p_upper.item()
            history.append(J.item())

            if current_p > best_p:
                best_p    = current_p
                best_u    = u_seq.detach().clone()
                best_K    = K.detach().clone() if K is not None else None
                best_mean = mean_trace.detach().clone()
                best_cov  = cov_trace.detach().clone()

            if current_p >= self.cfg["alpha"]:
                converged_iters += 1
                if converged_iters >= 20:
                    if verbose:
                        print(f"Converged at iter {k}. P(phi)={best_p:.4f}")
                    break
            else:
                converged_iters = 0

            if verbose and k % 50 == 0:
                k_norm = K.data.norm().item() if K is not None else 0.0
                print(
                    f"  iter {k:4d} | loss={J.item():.4f} "
                    f"| P(phi)={current_p:.4f} | ||K||={k_norm:.3f}"
                )

        return best_mean, best_cov, best_u, best_K, best_p, history
