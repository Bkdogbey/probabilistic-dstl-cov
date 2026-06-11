import math
import torch
from pdstl.operators import STL_Formula, Always, Eventually, And


# =============================================================================
# PROBABILISTIC PREDICATES
# =============================================================================

def normal_cdf(value, mean, var):
    """
    Computes P(X <= value) for X ~ N(mean, var).
    Standard Normal CDF Phi(z) where z = (value - mean) / sigma
    """
    std = torch.sqrt(var + 1e-6)
    z = (value - mean) / std
    return 0.5 * (1 + torch.erf(z / math.sqrt(2.0)))


class RectangularGoalPredicate(STL_Formula):
    """
    P_goal(t) = min( P(x >= x_min), P(x <= x_max), P(y >= y_min), P(y <= y_max) )

    Returns conservative probability intervals [lower, upper] where lower = upper.
    """

    def __init__(self, region):
        super().__init__()
        self.x_min, self.x_max = region["x"]
        self.y_min, self.y_max = region["y"]

    def robustness_trace(self, belief_trajectory, **kwargs):
        means, vars_diag = [], []
        for belief in belief_trajectory:
            means.append(belief.mean_full)
            if belief.var_full.ndim > 2:
                vars_diag.append(torch.diagonal(belief.var_full, dim1=-2, dim2=-1))
            else:
                vars_diag.append(belief.var_full)

        mu  = torch.stack(means, dim=1)     # [B, T, D]
        var = torch.stack(vars_diag, dim=1) # [B, T, D]

        mu_x, mu_y   = mu[..., 0], mu[..., 1]
        var_x, var_y = var[..., 0], var[..., 1]

        p_xmin = 1.0 - normal_cdf(self.x_min, mu_x, var_x)   # P(x >= x_min)
        p_xmax = normal_cdf(self.x_max, mu_x, var_x)           # P(x <= x_max)
        p_ymin = 1.0 - normal_cdf(self.y_min, mu_y, var_y)    # P(y >= y_min)
        p_ymax = normal_cdf(self.y_max, mu_y, var_y)           # P(y <= y_max)

        p_goal, _ = torch.stack([p_xmin, p_xmax, p_ymin, p_ymax], dim=0).min(dim=0)
        return torch.stack([p_goal, p_goal], dim=-1)

    def __str__(self):
        return f"goal([{self.x_min},{self.x_max}]×[{self.y_min},{self.y_max}])"


class RectangularObstaclePredicate(STL_Formula):
    """
    P_safe(t) = max( P(x <= x_min), P(x >= x_max), P(y <= y_min), P(y >= y_max) )
    Safe if the robot is to the left, right, below, or above the obstacle.
    """

    def __init__(self, region):
        super().__init__()
        self.x_min, self.x_max = region["x"]
        self.y_min, self.y_max = region["y"]

    def robustness_trace(self, belief_trajectory, **kwargs):
        means, vars_diag = [], []
        for belief in belief_trajectory:
            means.append(belief.mean_full)
            if belief.var_full.ndim > 2:
                vars_diag.append(torch.diagonal(belief.var_full, dim1=-2, dim2=-1))
            else:
                vars_diag.append(belief.var_full)

        mu  = torch.stack(means, dim=1)
        var = torch.stack(vars_diag, dim=1)

        mu_x, mu_y   = mu[..., 0], mu[..., 1]
        var_x, var_y = var[..., 0], var[..., 1]

        p_left  = normal_cdf(self.x_min, mu_x, var_x)          # P(x <= x_min)
        p_right = 1.0 - normal_cdf(self.x_max, mu_x, var_x)    # P(x >= x_max)
        p_below = normal_cdf(self.y_min, mu_y, var_y)           # P(y <= y_min)
        p_above = 1.0 - normal_cdf(self.y_max, mu_y, var_y)    # P(y >= y_max)

        p_safe, _ = torch.stack([p_left, p_right, p_below, p_above], dim=0).max(dim=0)
        return torch.stack([p_safe, p_safe], dim=-1)

    def __str__(self):
        return f"avoid([{self.x_min},{self.x_max}]×[{self.y_min},{self.y_max}])"


class CircularObstaclePredicate(STL_Formula):
    """
    P_safe(t) = P( ||x(t) - center|| > radius )
    Approximated via projected variance along the radial direction.
    """

    def __init__(self, circle_def, device="cpu"):
        super().__init__()
        self.center = torch.tensor(circle_def["center"], dtype=torch.float32, device=device)
        self.radius = circle_def["radius"]

    def robustness_trace(self, belief_trajectory, **kwargs):
        means, covs = [], []
        for belief in belief_trajectory:
            means.append(belief.mean_full)
            covs.append(belief.var_full)

        mu    = torch.stack(means, dim=1)   # [B, T, D]
        sigma = torch.stack(covs,  dim=1)   # [B, T, D] or [B, T, D, D]

        diff    = mu[..., :2] - self.center
        dist    = torch.norm(diff, dim=-1)
        dir_vec = diff / (dist.unsqueeze(-1) + 1e-6)

        if sigma.ndim == 3:
            sigma_proj = torch.sum(dir_vec ** 2 * sigma[..., :2], dim=-1)
        else:
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
        self.obstacles       = []
        self.circle_obstacles = []
        self.visit_regions   = []
        self.goal   = None
        self.bounds = None
        self.device = device

    def add_obstacle(self, x_range, y_range):
        """Axis-aligned rectangular obstacle."""
        self.obstacles.append({"x": x_range, "y": y_range})

    def add_circle_obstacle(self, center, radius):
        """Circular obstacle defined by center [x, y] and radius r."""
        self.circle_obstacles.append({"center": center, "radius": radius})

    def add_visit_region(self, x_range, y_range):
        """Region that must be visited at some point (liveness)."""
        self.visit_regions.append({"x": x_range, "y": y_range})

    def set_goal(self, x_range, y_range):
        """Goal region G = [x_min, x_max] × [y_min, y_max]."""
        self.goal = {"x": x_range, "y": y_range}

    def set_bounds(self, x_range, y_range):
        """Workspace bounds (treated as a always-satisfy region)."""
        self.bounds = {"x": x_range, "y": y_range}

    def get_specification(self, T, t_goal_start=0):
        """
        Build the combined STL formula: phi = (Eventually Goal) ∧ (Always Safe)

        Args:
            T: planning horizon in steps
            t_goal_start: earliest timestep the goal can be reached

        Returns:
            STL_Formula producing [B, T+1, 2] traces
        """
        specs = []

        # 1. Goal (liveness)
        if self.goal:
            specs.append(Eventually(RectangularGoalPredicate(self.goal),
                                    interval=[t_goal_start, T]))

        # 2. Visit regions (liveness)
        for region in self.visit_regions:
            specs.append(Eventually(RectangularGoalPredicate(region), interval=[0, T]))

        # 3. Obstacle safety
        obs_preds = (
            [RectangularObstaclePredicate(obs) for obs in self.obstacles]
            + [CircularObstaclePredicate(obs, device=self.device)
               for obs in self.circle_obstacles]
        )
        if obs_preds:
            safe = obs_preds[0]
            for p in obs_preds[1:]:
                safe = And(safe, p)
            specs.append(Always(safe, interval=[0, T]))

        # 4. Workspace bounds (safety)
        if self.bounds:
            specs.append(Always(RectangularGoalPredicate(self.bounds), interval=[0, T]))

        if not specs:
            raise ValueError("No constraints defined in environment.")

        combined = specs[0]
        for s in specs[1:]:
            combined = And(combined, s)
        return combined
