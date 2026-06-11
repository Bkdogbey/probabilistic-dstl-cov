import sys
import os
import yaml
import torch

sys.path.insert(0, os.path.dirname(__file__))

from utils import skip_run
from dynamics import DoubleIntegrator
from planning.environment import Environment
from planning.planner import ProbabilisticSTLPlanner
from visualization import plot_trajectory, plot_comparison


def _build_env(cfg, device="cpu"):
    env = Environment(device=device)
    if "goal" in cfg:
        env.set_goal(cfg["goal"]["x"], cfg["goal"]["y"])
    if "bounds" in cfg:
        env.set_bounds(cfg["bounds"]["x"], cfg["bounds"]["y"])
    for obs in cfg.get("obstacles", []):
        env.add_circle_obstacle(obs["center"], obs["radius"])
    return env


# =============================================================================
# LANE CHANGE — OPEN LOOP
# =============================================================================
with skip_run("run", "Lane Change - Open Loop") as check, check():
    cfg = yaml.safe_load(open("configs/scenarios/lane_change.yaml"))
    T = cfg["horizon"]

    dyn = DoubleIntegrator(dt=0.1, u_max=2.0, sigma_w=0.1)
    env = _build_env(cfg)

    ic = cfg["initial_state"]
    mu0    = torch.tensor(ic["mean"],     dtype=torch.float32)
    Sigma0 = torch.diag(torch.tensor(ic["cov_diag"], dtype=torch.float32))

    planner_cfg = {**cfg.get("optimizer", {}), **cfg.get("weights", {})}
    planner = ProbabilisticSTLPlanner(dyn, env, T, steerer="open_loop", config=planner_cfg)
    mean_trace, cov_trace, best_u, best_K, best_p, history = planner.solve(mu0, Sigma0)

    print(f"\n  P(phi)       = {best_p:.4f}")
    print(f"  cov_trace(T) = {(cov_trace[0,-1,0,0] + cov_trace[0,-1,1,1]).item():.4f}")

    os.makedirs("results", exist_ok=True)
    plot_trajectory(mean_trace, cov_trace, env,
                    title=cfg.get("label", "Open Loop"),
                    save_path="results/open_loop.png")


# =============================================================================
# LANE CHANGE — CLOSED LOOP
# =============================================================================
with skip_run("run", "Lane Change - Closed Loop") as check, check():
    cfg = yaml.safe_load(open("configs/scenarios/lane_change_closed.yaml"))
    T = cfg["horizon"]

    dyn = DoubleIntegrator(dt=0.1, u_max=2.0, sigma_w=0.1)
    env = _build_env(cfg)

    ic = cfg["initial_state"]
    mu0    = torch.tensor(ic["mean"],     dtype=torch.float32)
    Sigma0 = torch.diag(torch.tensor(ic["cov_diag"], dtype=torch.float32))

    planner_cfg = {**cfg.get("optimizer", {}), **cfg.get("weights", {})}
    planner = ProbabilisticSTLPlanner(dyn, env, T, steerer="closed_loop", config=planner_cfg)
    mean_trace, cov_trace, best_u, best_K, best_p, history = planner.solve(mu0, Sigma0)

    print(f"\n  P(phi)       = {best_p:.4f}")
    print(f"  ||K||        = {best_K.norm().item():.4f}")
    print(f"  cov_trace(T) = {(cov_trace[0,-1,0,0] + cov_trace[0,-1,1,1]).item():.4f}")

    os.makedirs("results", exist_ok=True)
    plot_trajectory(mean_trace, cov_trace, env,
                    title=cfg.get("label", "Closed Loop"),
                    save_path="results/closed_loop.png")


# =============================================================================
# SIDE-BY-SIDE COMPARISON
# =============================================================================
with skip_run("skip", "Open vs Closed Comparison") as check, check():
    def _run(scenario_path, steerer):
        cfg = yaml.safe_load(open(scenario_path))
        T = cfg["horizon"]
        dyn = DoubleIntegrator(dt=0.1, u_max=2.0, sigma_w=0.1)
        env = _build_env(cfg)
        ic = cfg["initial_state"]
        mu0    = torch.tensor(ic["mean"],     dtype=torch.float32)
        Sigma0 = torch.diag(torch.tensor(ic["cov_diag"], dtype=torch.float32))
        planner_cfg = {**cfg.get("optimizer", {}), **cfg.get("weights", {})}
        planner = ProbabilisticSTLPlanner(dyn, env, T, steerer=steerer, config=planner_cfg)
        return planner.solve(mu0, Sigma0), env

    ol_result, env = _run("configs/scenarios/lane_change.yaml", "open_loop")
    cl_result, _   = _run("configs/scenarios/lane_change_closed.yaml", "closed_loop")

    os.makedirs("results", exist_ok=True)
    plot_comparison(ol_result, cl_result, env, save_path="results/comparison.png")
