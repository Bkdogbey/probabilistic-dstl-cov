"""Workspace environment: probabilistic predicates and STL specification builder."""

import math

import numpy as np
import torch

from pdstl.operators import STLFormula, Always, Eventually, And


# =============================================================================
# BELIEF TRAJECTORY UTILITIES
# =============================================================================


def extract_trajectory_stats(belief_trajectory, diagonal_only=True):
    """Stack mean and covariance tensors from a BeliefTrajectory.

    Parameters
    ----------
    belief_trajectory : BeliefTrajectory
    diagonal_only : bool
        If True, extract only the diagonal of full covariance matrices,
        returning var of shape [Batch, Time, Dim].
        If False, stack full covariance matrices as-is.

    Returns
    -------
    mu  : Tensor [Batch, Time, Dim]
    var : Tensor [Batch, Time, Dim]      (diagonal_only=True)
          or [Batch, Time, Dim, Dim]     (diagonal_only=False)
    """
    means, vars_ = [], []
    for belief in belief_trajectory:
        means.append(belief.mean_full)
        if diagonal_only and belief.var_full.ndim > 2:
            vars_.append(torch.diagonal(belief.var_full, dim1=-2, dim2=-1))
        else:
            vars_.append(belief.var_full)
    return torch.stack(means, dim=1), torch.stack(vars_, dim=1)


# =============================================================================
# PROBABILISTIC PREDICATES
# =============================================================================


def normal_cdf(value, mean, var):
    """Compute P(X <= value) for X ~ N(mean, var)."""
    std = torch.sqrt(var + 1e-6)
    z = (value - mean) / std
    return 0.5 * (1 + torch.erf(z / math.sqrt(2.0)))


class RectangularGoalPredicate(STLFormula):
    """
    Per-axis interval probability combined with the product rule (independence):
        p_x = P(x_min <= x <= x_max) = Phi(x_max) - Phi(x_min)
        p_y = P(y_min <= y <= y_max) = Phi(y_max) - Phi(y_min)
        lower = p_x * p_y   (product bound — matches And operator)
        upper = min(p_x, p_y)

    (Fix 4: product lower bound is tighter than Fréchet for joint intervals.)
    """

    def __init__(self, region):
        super().__init__()
        self.x_min, self.x_max = region["x"]
        self.y_min, self.y_max = region["y"]

    def robustness_trace(self, belief_trajectory, **kwargs):
        mu, var = extract_trajectory_stats(belief_trajectory)

        mu_x, mu_y = mu[..., 0], mu[..., 1]
        var_x, var_y = var[..., 0], var[..., 1]

        p_x = normal_cdf(self.x_max, mu_x, var_x) - normal_cdf(self.x_min, mu_x, var_x)
        p_y = normal_cdf(self.y_max, mu_y, var_y) - normal_cdf(self.y_min, mu_y, var_y)
        p_x = p_x.clamp(0.0, 1.0)
        p_y = p_y.clamp(0.0, 1.0)

        lower = p_x * p_y
        upper = torch.minimum(p_x, p_y)
        return torch.stack([lower, upper], dim=-1)

    def __str__(self):
        return f"goal([{self.x_min},{self.x_max}]×[{self.y_min},{self.y_max}])"


class RectangularObstaclePredicate(STLFormula):
    """
    P_safe(t) = max( P(x<=x_min), P(x>=x_max), P(y<=y_min), P(y>=y_max) )
    Safe if the robot is to the left, right, below, or above the obstacle.
    """

    def __init__(self, region):
        super().__init__()
        self.x_min, self.x_max = region["x"]
        self.y_min, self.y_max = region["y"]

    def robustness_trace(self, belief_trajectory, **kwargs):
        mu, var = extract_trajectory_stats(belief_trajectory)

        mu_x, mu_y = mu[..., 0], mu[..., 1]
        var_x, var_y = var[..., 0], var[..., 1]

        p_left = normal_cdf(self.x_min, mu_x, var_x)
        p_right = 1.0 - normal_cdf(self.x_max, mu_x, var_x)
        p_below = normal_cdf(self.y_min, mu_y, var_y)
        p_above = 1.0 - normal_cdf(self.y_max, mu_y, var_y)

        p_safe, _ = torch.stack([p_left, p_right, p_below, p_above], dim=0).max(dim=0)
        return torch.stack([p_safe, p_safe], dim=-1)

    def __str__(self):
        return f"avoid([{self.x_min},{self.x_max}]×[{self.y_min},{self.y_max}])"


class MovingRectangularObstaclePredicate(STLFormula):
    """
    P_safe(t) = max( P(x<=x_min(t)), P(x>=x_max(t)), P(y<=y_min(t)), P(y>=y_max(t)) )
    Obstacle bounds are time-varying; x_traj/y_traj have shape [T+1].
    """

    def __init__(self, obs_def, device="cpu"):
        super().__init__()
        self.x_traj = torch.as_tensor(obs_def["x_traj"], device=device, dtype=torch.float32)
        self.y_traj = torch.as_tensor(obs_def["y_traj"], device=device, dtype=torch.float32)
        self.width = obs_def["width"]
        self.height = obs_def["height"]

    def robustness_trace(self, belief_trajectory, **kwargs):
        mu, var = extract_trajectory_stats(belief_trajectory)

        mu_x, mu_y = mu[..., 0], mu[..., 1]
        var_x, var_y = var[..., 0], var[..., 1]

        x_min = self.x_traj - self.width / 2.0  # [T+1] broadcasts against [B, T+1]
        x_max = self.x_traj + self.width / 2.0
        y_min = self.y_traj - self.height / 2.0
        y_max = self.y_traj + self.height / 2.0

        p_left = normal_cdf(x_min, mu_x, var_x)
        p_right = 1.0 - normal_cdf(x_max, mu_x, var_x)
        p_below = normal_cdf(y_min, mu_y, var_y)
        p_above = 1.0 - normal_cdf(y_max, mu_y, var_y)

        p_safe, _ = torch.stack([p_left, p_right, p_below, p_above], dim=0).max(dim=0)
        return torch.stack([p_safe, p_safe], dim=-1)

    def __str__(self):
        return f"avoid_moving(w={self.width}, h={self.height})"


class CircularObstaclePredicate(STLFormula):
    """
    P_safe(t) = P( ||x(t) - center|| > radius )
    Approximated via projected variance along the radial direction.
    Valid when the mean clears the obstacle by more than one covariance length.
    """

    def __init__(self, circle_def, device="cpu"):
        super().__init__()
        self.center = torch.tensor(circle_def["center"], dtype=torch.float32, device=device)
        self.radius = circle_def["radius"]

    def robustness_trace(self, belief_trajectory, **kwargs):
        mu, sigma = extract_trajectory_stats(belief_trajectory, diagonal_only=False)

        diff = mu[..., :2] - self.center  # [B, T, 2]
        dist = torch.norm(diff, dim=-1)  # [B, T]
        dir_vec = diff / (dist.unsqueeze(-1) + 1e-6)

        if sigma.ndim == 3:  # diagonal: [B, T, D]
            sigma_proj = torch.sum(dir_vec**2 * sigma[..., :2], dim=-1)
        else:  # full: [B, T, D, D]
            sigma_proj = torch.einsum("bti,btij,btj->bt", dir_vec, sigma[..., :2, :2], dir_vec)

        p_safe = 1.0 - normal_cdf(self.radius, dist, sigma_proj)
        return torch.stack([p_safe, p_safe], dim=-1)

    def __str__(self):
        return f"avoid_circle(c={self.center.tolist()}, r={self.radius})"


# =============================================================================
# ENVIRONMENT
# =============================================================================


class Environment:
    """
    Defines the workspace, obstacles, and goal region.
    Generates the probabilistic STL specification for the optimizer.
    """

    def __init__(self, device="cpu"):
        self.obstacles = []
        self.circle_obstacles = []
        self.moving_obstacles = []
        self.lane_markings = []
        self.visit_regions = []
        self.goal = None
        self.bounds = None
        self.device = device
        self.road = None
        self.lane_change = None  # populated by configure_lane_change()
        self.success = None  # optional success criterion dict
        self.label = ""
        self.plot_xlim = None
        self.robot_dims = None

    # ------------------------------------------------------------------
    # Workspace setup
    # ------------------------------------------------------------------

    def add_obstacle(self, x_range: list, y_range: list) -> None:
        """Axis-aligned rectangular obstacle."""
        self.obstacles.append({"x": x_range, "y": y_range})

    def add_circle_obstacle(self, center: list, radius: float) -> None:
        """Circular obstacle defined by center [x, y] and radius r."""
        self.circle_obstacles.append({"center": center, "radius": radius})

    def add_moving_obstacle(self, x_traj, y_traj, width: float, height: float) -> None:
        """Constant-speed rectangular obstacle; x_traj/y_traj are [T+1] arrays."""
        self.moving_obstacles.append(
            {"x_traj": x_traj, "y_traj": y_traj, "width": width, "height": height}
        )

    def add_lane_marking(
        self, x_range: list, y_pos: float, style: str = "dashed", color: str = "white"
    ) -> None:
        """Visual lane marking at y=y_pos."""
        self.lane_markings.append({"x": x_range, "y": y_pos, "style": style, "color": color})

    def add_visit_region(self, x_range: list, y_range: list) -> None:
        """Region that must be visited at some point (liveness)."""
        self.visit_regions.append({"x": x_range, "y": y_range})

    def set_goal(self, x_range: list, y_range: list) -> None:
        """Goal region G = [x_min, x_max] × [y_min, y_max]."""
        self.goal = {"x": x_range, "y": y_range}

    def set_bounds(self, x_range: list, y_range: list) -> None:
        """Hard workspace boundaries (always-satisfy constraint)."""
        self.bounds = {"x": x_range, "y": y_range}

    def draw_on_ax(self, ax, **kwargs) -> None:
        """Draw the environment onto a matplotlib Axes (deferred import)."""
        from visualization import _draw_env  # noqa: PLC0415

        _draw_env(ax, self, **kwargs)

    # ------------------------------------------------------------------
    # Lane-change configuration
    # ------------------------------------------------------------------

    def configure_lane_change(
        self,
        *,
        road: dict,
        obstacle: dict,
        horizon: int,
        dt: float,
        label: str = "",
        plot_xlim=None,
        robot_dims=None,
    ) -> None:
        """Configure a lane-change problem from scenario parameters.

        Args:
            road:       dict with keys y_min, y_max, lane_divider,
                        and optionally marking_x_range
            obstacle:   dict with keys x0, speed, y, width, height
            horizon:    planning horizon T (number of steps)
            dt:         timestep [s]
            label:      scenario label for visualization
            plot_xlim:  optional [x_lo, x_hi] for plot axis limits
            robot_dims: optional [length, width] for ego-vehicle rectangle
        """
        self.road = dict(road)
        self.label = label
        self.plot_xlim = plot_xlim
        self.robot_dims = tuple(robot_dims) if robot_dims is not None else None

        marking_x = road.get("marking_x_range", [-2.0, 20.0])
        self.add_lane_marking(x_range=marking_x, y_pos=road["lane_divider"], style="dashed")
        self.add_lane_marking(x_range=marking_x, y_pos=road["y_min"], style="solid")
        self.add_lane_marking(x_range=marking_x, y_pos=road["y_max"], style="solid")

        times = np.arange(horizon + 1) * dt
        obs_x = obstacle["x0"] + obstacle["speed"] * times  # [T+1]
        obs_y = np.ones_like(times) * obstacle["y"]

        self.lane_change = {
            "obstacle": dict(obstacle),
            "horizon": horizon,
            "dt": dt,
            "obs_x_global": obs_x,
            "obs_y_global": obs_y,
        }
        self.add_moving_obstacle(obs_x, obs_y, width=obstacle["width"], height=obstacle["height"])

    def moving_obstacle_position(self, step: int):
        """Return the first moving obstacle center [x, y] at global step index.

        Returns None if no lane_change has been configured.
        """
        if self.lane_change is None:
            return None
        obs_x = self.lane_change["obs_x_global"]
        obs_y = self.lane_change["obs_y_global"]
        idx = min(step, len(obs_x) - 1)
        return np.array([obs_x[idx], obs_y[idx]])

    def clip_moving_obstacles(self, num_points: int) -> None:
        """Trim all moving obstacle trajectories to num_points entries."""
        for obs in self.moving_obstacles:
            obs["x_traj"] = obs["x_traj"][:num_points]
            obs["y_traj"] = obs["y_traj"][:num_points]

    def make_local_lane_change_window(self, step: int, curr_mean, cfg: dict) -> "Environment":
        """Build a local planning Environment for one MPC step.

        Not used by the single-shot planner but part of the canonical API.
        """
        if self.road is None or self.lane_change is None:
            raise ValueError("Lane-change local windows require configure_lane_change().")

        horizon = self.lane_change["horizon"]
        obstacle = self.lane_change["obstacle"]
        obs_x = self.lane_change["obs_x_global"]
        obs_y = self.lane_change["obs_y_global"]
        road = self.road

        curr_x = float(curr_mean.detach().cpu()[0])
        goal_lookahead = cfg["mpc_goal_lookahead"]
        goal_width = cfg["mpc_goal_window_width"]
        goal_y_inset = cfg["goal_y_inset"]
        lane_margin = cfg["lane_boundary_margin"]

        env_local = Environment(device=self.device)
        env_local.set_goal(
            x_range=[curr_x + goal_lookahead, curr_x + goal_lookahead + goal_width],
            y_range=[
                self.goal["y"][0] + goal_y_inset,
                self.goal["y"][1] - goal_y_inset,
            ],
        )
        y_min_bound = road["y_min"] + lane_margin
        if curr_mean[1] > road["lane_divider"] - lane_margin:
            y_min_bound = road["lane_divider"]
        env_local.set_bounds(
            x_range=cfg["mpc_local_x_range"],
            y_range=[y_min_bound, road["y_max"]],
        )

        idx_end = step + horizon + 1
        if idx_end <= len(obs_x):
            sl_x = obs_x[step:idx_end]
            sl_y = obs_y[step:idx_end]
        else:
            pad = idx_end - len(obs_x)
            sl_x = np.concatenate([obs_x[step:], np.full(pad, obs_x[-1])])
            sl_y = np.concatenate([obs_y[step:], np.full(pad, obs_y[-1])])

        env_local.add_moving_obstacle(sl_x, sl_y, obstacle["width"], obstacle["height"])
        return env_local

    # ------------------------------------------------------------------
    # STL specification builder
    # ------------------------------------------------------------------

    def get_predicates(self) -> dict:
        """Build and return all probabilistic predicates as a categorised dict."""
        preds: dict = {"goal": None, "visit": [], "obstacles": []}

        if self.goal:
            preds["goal"] = RectangularGoalPredicate(self.goal)

        for region in self.visit_regions:
            preds["visit"].append(RectangularGoalPredicate(region))

        for obs in self.obstacles:
            preds["obstacles"].append(RectangularObstaclePredicate(obs))
        for obs in self.circle_obstacles:
            preds["obstacles"].append(CircularObstaclePredicate(obs, device=self.device))
        for obs in self.moving_obstacles:
            preds["obstacles"].append(MovingRectangularObstaclePredicate(obs, device=self.device))

        return preds

    def get_specification(self, T: int, t_goal_start: int = 0, t_constraints_start: int = 1):
        """Build the combined STL formula: φ = (Eventually Goal) ∧ (Always Safe).

        Args:
            T:                  planning horizon in steps
            t_goal_start:       earliest timestep the goal can first be reached
            t_constraints_start: first timestep where safety constraints apply
                                 (default 1 — skips t=0 initial condition)

        Returns:
            STLFormula producing [B, T+1, 2] traces
        """
        preds = self.get_predicates()
        specs = []

        # 1. Goal (liveness)
        if preds["goal"]:
            specs.append(Eventually(preds["goal"], interval=[t_goal_start, T]))

        # 2. Visit regions (liveness)
        for vp in preds["visit"]:
            specs.append(Eventually(vp, interval=[0, T]))

        # 3. Obstacle safety (static rectangular, circular, and moving rectangular)
        if preds["obstacles"]:
            safe = preds["obstacles"][0]
            for p in preds["obstacles"][1:]:
                safe = And(safe, p)
            specs.append(Always(safe, interval=[t_constraints_start, T]))

        # 4. Workspace bounds (always stay inside)
        if self.bounds:
            specs.append(
                Always(
                    RectangularGoalPredicate(self.bounds),
                    interval=[t_constraints_start, T],
                )
            )

        if not specs:
            raise ValueError("No constraints defined in environment.")

        combined = specs[0]
        for s in specs[1:]:
            combined = And(combined, s)
        return combined
