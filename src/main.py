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
from visualization.animation import animate_trajectory, make_live_callback


_dyn_cfg = yaml.safe_load(open("config/dynamics/double_integrator.yaml"))


def _build_env(cfg, device="cpu"):
    env = Environment(device=device)
    if "goal" in cfg:
        env.set_goal(cfg["goal"]["x"], cfg["goal"]["y"])
    if "bounds" in cfg:
        env.set_bounds(cfg["bounds"]["x"], cfg["bounds"]["y"])
    for obs in cfg.get("obstacles", []):
        env.add_circle_obstacle(obs["center"], obs["radius"])
    for obs in cfg.get("rect_obstacles", []):
        env.add_obstacle(obs["x"], obs["y"])
    if "moving_obstacle" in cfg:
        env.configure_lane_change(
            road=cfg["road"],
            obstacle=cfg["moving_obstacle"],
            horizon=cfg["horizon"],
            dt=_dyn_cfg["dt"],
            label=cfg.get("label", ""),
        )
    return env


# =============================================================================
# LANE CHANGE — OPEN LOOP
# =============================================================================
with skip_run("run", "Lane Change - Open Loop") as check, check():
    cfg = yaml.safe_load(open("config/scenarios/lane_change.yaml"))
    T = cfg["horizon"]

    dyn = DoubleIntegrator(**{k: _dyn_cfg[k] for k in ("dt", "u_max", "sigma_w")})
    env = _build_env(cfg)

    ic = cfg["initial_state"]
    mu0 = torch.tensor(ic["mean"], dtype=torch.float32)
    Sigma0 = torch.diag(torch.tensor(ic["cov_diag"], dtype=torch.float32))

    planner_cfg = {**cfg.get("optimizer", {}), **cfg.get("weights", {})}
    planner = ProbabilisticSTLPlanner(dyn, env, T, steerer="open_loop", config=planner_cfg)
    mean_trace, cov_trace, best_u, best_K, best_p, history = planner.solve(mu0, Sigma0)

    print(f"\n  P(phi)       = {best_p:.4f}")
    print(f"  cov_trace(T) = {(cov_trace[0, -1, 0, 0] + cov_trace[0, -1, 1, 1]).item():.4f}")

    os.makedirs("results", exist_ok=True)
    plot_trajectory(
        mean_trace,
        cov_trace,
        env,
        title=cfg.get("label", "Open Loop"),
        save_path="results/open_loop.png",
    )


# =============================================================================
# LANE CHANGE — CLOSED LOOP
# =============================================================================
with skip_run("skip", "Lane Change - Closed Loop") as check, check():
    cfg = yaml.safe_load(open("config/scenarios/lane_change_closed.yaml"))
    T = cfg["horizon"]

    dyn = DoubleIntegrator(**{k: _dyn_cfg[k] for k in ("dt", "u_max", "sigma_w")})
    env = _build_env(cfg)

    ic = cfg["initial_state"]
    mu0 = torch.tensor(ic["mean"], dtype=torch.float32)
    Sigma0 = torch.diag(torch.tensor(ic["cov_diag"], dtype=torch.float32))

    planner_cfg = {**cfg.get("optimizer", {}), **cfg.get("weights", {})}
    planner = ProbabilisticSTLPlanner(dyn, env, T, steerer="closed_loop", config=planner_cfg)
    mean_trace, cov_trace, best_u, best_K, best_p, history = planner.solve(mu0, Sigma0)

    print(f"\n  P(phi)       = {best_p:.4f}")
    print(f"  ||K||        = {best_K.norm().item():.4f}")
    print(f"  cov_trace(T) = {(cov_trace[0, -1, 0, 0] + cov_trace[0, -1, 1, 1]).item():.4f}")

    os.makedirs("results", exist_ok=True)
    plot_trajectory(
        mean_trace,
        cov_trace,
        env,
        title=cfg.get("label", "Closed Loop"),
        save_path="results/closed_loop.png",
    )


# =============================================================================
# SIDE-BY-SIDE COMPARISON
# =============================================================================
with skip_run("skip", "Open vs Closed Comparison") as check, check():

    def _run(scenario_path, steerer):
        cfg = yaml.safe_load(open(scenario_path))
        T = cfg["horizon"]
        dyn = DoubleIntegrator(**{k: _dyn_cfg[k] for k in ("dt", "u_max", "sigma_w")})
        env = _build_env(cfg)
        ic = cfg["initial_state"]
        mu0 = torch.tensor(ic["mean"], dtype=torch.float32)
        Sigma0 = torch.diag(torch.tensor(ic["cov_diag"], dtype=torch.float32))
        planner_cfg = {**cfg.get("optimizer", {}), **cfg.get("weights", {})}
        planner = ProbabilisticSTLPlanner(dyn, env, T, steerer=steerer, config=planner_cfg)
        return planner.solve(mu0, Sigma0), env

    ol_result, env = _run("config/scenarios/lane_change.yaml", "open_loop")
    cl_result, _ = _run("config/scenarios/lane_change_closed.yaml", "closed_loop")

    os.makedirs("results", exist_ok=True)
    plot_comparison(ol_result, cl_result, env, save_path="results/comparison.png")


# =============================================================================
# LANE MERGE — OPEN LOOP  (moving obstacle)
# =============================================================================
with skip_run("run", "Lane Merge - Open Loop") as check, check():
    cfg = yaml.safe_load(open("config/scenarios/lane_merge.yaml"))
    T = cfg["horizon"]

    dyn = DoubleIntegrator(**{k: _dyn_cfg[k] for k in ("dt", "u_max", "sigma_w")})
    env = _build_env(cfg)

    ic = cfg["initial_state"]
    mu0 = torch.tensor(ic["mean"], dtype=torch.float32)
    Sigma0 = torch.diag(torch.tensor(ic["cov_diag"], dtype=torch.float32))

    planner_cfg = {**cfg.get("optimizer", {}), **cfg.get("weights", {})}
    planner = ProbabilisticSTLPlanner(dyn, env, T, steerer="open_loop", config=planner_cfg)
    cb = make_live_callback(env, title="Lane Merge — Open Loop Optimization")
    mean_trace, cov_trace, best_u, best_K, best_p, history = planner.solve(
        mu0, Sigma0, callback=cb, callback_every=25
    )

    print(f"\n  rho_lb(phi) = {best_p:.4f}")
    print(f"  cov_trace(T) = {(cov_trace[0, -1, 0, 0] + cov_trace[0, -1, 1, 1]).item():.4f}")

    os.makedirs("results", exist_ok=True)
    plot_trajectory(
        mean_trace,
        cov_trace,
        env,
        title=cfg.get("label", "Lane Merge Open Loop"),
        save_path="results/lane_merge_open.png",
    )
    animate_trajectory(
        mean_trace,
        cov_trace,
        env,
        filename="results/lane_merge_open.gif",
        dt=_dyn_cfg["dt"],
        title=cfg.get("label", "Lane Merge Open Loop"),
    )


# =============================================================================
# LANE MERGE — CLOSED LOOP  (moving obstacle)
# =============================================================================
with skip_run("run", "Lane Merge - Closed Loop") as check, check():
    cfg = yaml.safe_load(open("config/scenarios/lane_merge_closed.yaml"))
    T = cfg["horizon"]

    dyn = DoubleIntegrator(**{k: _dyn_cfg[k] for k in ("dt", "u_max", "sigma_w")})
    env = _build_env(cfg)

    ic = cfg["initial_state"]
    mu0 = torch.tensor(ic["mean"], dtype=torch.float32)
    Sigma0 = torch.diag(torch.tensor(ic["cov_diag"], dtype=torch.float32))

    planner_cfg = {**cfg.get("optimizer", {}), **cfg.get("weights", {})}
    planner = ProbabilisticSTLPlanner(dyn, env, T, steerer="closed_loop", config=planner_cfg)
    cb = make_live_callback(env, title="Lane Merge — Closed Loop Optimization")
    mean_trace, cov_trace, best_u, best_K, best_p, history = planner.solve(
        mu0, Sigma0, callback=cb, callback_every=25
    )

    print(f"\n  rho_lb(phi) = {best_p:.4f}")
    print(f"  ||K||        = {best_K.norm().item():.4f}")
    print(f"  cov_trace(T) = {(cov_trace[0, -1, 0, 0] + cov_trace[0, -1, 1, 1]).item():.4f}")

    os.makedirs("results", exist_ok=True)
    plot_trajectory(
        mean_trace,
        cov_trace,
        env,
        title=cfg.get("label", "Lane Merge Closed Loop"),
        save_path="results/lane_merge_closed.png",
    )
    animate_trajectory(
        mean_trace,
        cov_trace,
        env,
        filename="results/lane_merge_closed.gif",
        dt=_dyn_cfg["dt"],
        title=cfg.get("label", "Lane Merge Closed Loop"),
    )


# =============================================================================
# DOUBLE SLIT — OPEN LOOP
# =============================================================================
with skip_run("run", "Double Slit - Open Loop") as check, check():
    cfg = yaml.safe_load(open("config/scenarios/double_slit.yaml"))
    T = cfg["horizon"]

    dyn = DoubleIntegrator(**{k: _dyn_cfg[k] for k in ("dt", "u_max", "sigma_w")})
    env = _build_env(cfg)

    ic = cfg["initial_state"]
    mu0 = torch.tensor(ic["mean"], dtype=torch.float32)
    Sigma0 = torch.diag(torch.tensor(ic["cov_diag"], dtype=torch.float32))

    planner_cfg = {**cfg.get("optimizer", {}), **cfg.get("weights", {})}
    planner = ProbabilisticSTLPlanner(dyn, env, T, steerer="open_loop", config=planner_cfg)
    cb = make_live_callback(env, title="Double Slit — Open Loop Optimization")
    mean_trace, cov_trace, best_u, best_K, best_p, history = planner.solve(
        mu0, Sigma0, callback=cb, callback_every=25
    )

    print(f"\n  rho_lb(phi) = {best_p:.4f}")
    print(f"  cov_trace(T) = {(cov_trace[0, -1, 0, 0] + cov_trace[0, -1, 1, 1]).item():.4f}")

    os.makedirs("results", exist_ok=True)
    plot_trajectory(
        mean_trace,
        cov_trace,
        env,
        title=cfg.get("label", "Double Slit Open Loop"),
        save_path="results/double_slit_open.png",
    )
    animate_trajectory(
        mean_trace,
        cov_trace,
        env,
        filename="results/double_slit_open.gif",
        dt=_dyn_cfg["dt"],
        title=cfg.get("label", "Double Slit Open Loop"),
    )


# =============================================================================
# DOUBLE SLIT — CLOSED LOOP
# =============================================================================
with skip_run("run", "Double Slit - Closed Loop") as check, check():
    cfg = yaml.safe_load(open("config/scenarios/double_slit_closed.yaml"))
    T = cfg["horizon"]

    dyn = DoubleIntegrator(**{k: _dyn_cfg[k] for k in ("dt", "u_max", "sigma_w")})
    env = _build_env(cfg)

    ic = cfg["initial_state"]
    mu0 = torch.tensor(ic["mean"], dtype=torch.float32)
    Sigma0 = torch.diag(torch.tensor(ic["cov_diag"], dtype=torch.float32))

    planner_cfg = {**cfg.get("optimizer", {}), **cfg.get("weights", {})}
    planner = ProbabilisticSTLPlanner(dyn, env, T, steerer="closed_loop", config=planner_cfg)
    cb = make_live_callback(env, title="Double Slit — Closed Loop Optimization")
    mean_trace, cov_trace, best_u, best_K, best_p, history = planner.solve(
        mu0, Sigma0, callback=cb, callback_every=25
    )

    print(f"\n  rho_lb(phi) = {best_p:.4f}")
    print(f"  ||K||        = {best_K.norm().item():.4f}")
    print(f"  cov_trace(T) = {(cov_trace[0, -1, 0, 0] + cov_trace[0, -1, 1, 1]).item():.4f}")

    os.makedirs("results", exist_ok=True)
    plot_trajectory(
        mean_trace,
        cov_trace,
        env,
        title=cfg.get("label", "Double Slit Closed Loop"),
        save_path="results/double_slit_closed.png",
    )
    animate_trajectory(
        mean_trace,
        cov_trace,
        env,
        filename="results/double_slit_closed.gif",
        dt=_dyn_cfg["dt"],
        title=cfg.get("label", "Double Slit Closed Loop"),
    )
