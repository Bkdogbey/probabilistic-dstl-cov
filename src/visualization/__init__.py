"""Matplotlib helpers for plotting belief trajectories and environment layouts."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse


def _cov_ellipse(mu_xy, Sigma_xy, n_std=2.0, **kwargs):
    """Return an Ellipse patch for a 2×2 covariance matrix."""
    vals, vecs = np.linalg.eigh(Sigma_xy)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(np.abs(vals))
    return Ellipse(xy=mu_xy, width=width, height=height, angle=angle, **kwargs)


def _draw_env(ax, env):
    """Draw obstacles, goal, bounds, lane markings, and moving obstacle onto ax."""
    if env.bounds:
        for y_val in env.bounds["y"]:
            ax.axhline(y_val, color="k", lw=1.5)
        ax.set_xlim(env.bounds["x"][0] - 0.3, env.bounds["x"][1] + 0.3)
        ax.set_ylim(env.bounds["y"][0] - 0.3, env.bounds["y"][1] + 0.3)

    # Lane markings (dashed centre line, solid road edges)
    for lm in getattr(env, "lane_markings", []):
        ls = "--" if lm["style"] == "dashed" else "-"
        ax.axhline(lm["y"], color="grey", lw=1.0, linestyle=ls, zorder=1)

    for obs in env.obstacles:
        xlo, xhi = obs["x"]
        ylo, yhi = obs["y"]
        ax.add_patch(
            patches.Rectangle(
                (xlo, ylo),
                xhi - xlo,
                yhi - ylo,
                edgecolor="darkred",
                facecolor="red",
                alpha=0.3,
                zorder=3,
            )
        )

    for obs in env.circle_obstacles:
        cx, cy = obs["center"]
        ax.add_patch(plt.Circle((cx, cy), obs["radius"], color="red", alpha=0.35, zorder=3))

    # Moving obstacle — draw at t=0 (initial position) with a velocity arrow
    for obs in getattr(env, "moving_obstacles", []):
        x0 = float(obs["x_traj"][0])
        y0 = float(obs["y_traj"][0])
        w, h = obs["width"], obs["height"]
        ax.add_patch(
            patches.Rectangle(
                (x0 - w / 2, y0 - h / 2),
                w,
                h,
                edgecolor="darkorange",
                facecolor="orange",
                alpha=0.5,
                linestyle="--",
                zorder=3,
            )
        )
        # Arrow showing direction of motion (to midpoint of trajectory)
        mid = len(obs["x_traj"]) // 2
        dx = float(obs["x_traj"][mid]) - x0
        ax.annotate(
            "",
            xy=(x0 + dx, y0),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.5),
            zorder=4,
        )

    if env.goal:
        xlo, xhi = env.goal["x"]
        ylo, yhi = env.goal["y"]
        ax.add_patch(
            patches.Rectangle(
                (xlo, ylo),
                xhi - xlo,
                yhi - ylo,
                edgecolor="green",
                facecolor="lightgreen",
                alpha=0.4,
                zorder=2,
            )
        )


def plot_trajectory(mean_trace, cov_trace, env, title="", save_path=None, ellipse_every=3):
    """
    Plot nominal trajectory with covariance ellipses, obstacles, and goal.

    Args:
        mean_trace: [1, T+1, nx] or [T+1, nx]
        cov_trace:  [1, T+1, nx, nx] or [T+1, nx, nx]
        env:        Environment
        title:      figure title
        save_path:  save to file if provided, else show
        ellipse_every: draw a covariance ellipse every N steps
    """
    mu = mean_trace[0].numpy() if mean_trace.dim() == 3 else mean_trace.numpy()
    Sigma = cov_trace[0].numpy() if cov_trace.dim() == 4 else cov_trace.numpy()
    T = mu.shape[0] - 1

    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_env(ax, env)

    for t in range(0, T + 1, ellipse_every):
        S_xy = Sigma[t, :2, :2]
        if np.all(np.isfinite(S_xy)):
            ax.add_patch(
                _cov_ellipse(
                    mu[t, :2],
                    S_xy,
                    n_std=2.0,
                    facecolor="steelblue",
                    alpha=0.15,
                    edgecolor="steelblue",
                    linewidth=0.8,
                )
            )

    ax.plot(mu[:, 0], mu[:, 1], "o-", color="steelblue", ms=4, lw=1.5, label="trajectory")
    ax.plot(mu[0, 0], mu[0, 1], "go", ms=8, label="start")
    ax.plot(mu[-1, 0], mu[-1, 1], "b*", ms=10, label="end")

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_comparison(ol_result, cl_result, env, save_path=None, ellipse_every=3):
    """
    Side-by-side comparison of open-loop and closed-loop results.

    Each result is a tuple: (mean_trace, cov_trace, u, K, best_p, history)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ol_mean, ol_cov, _, _, ol_p, _ = ol_result
    cl_mean, cl_cov, _, cl_K, cl_p, _ = cl_result

    k_norm = cl_K.norm().item() if cl_K is not None else 0.0

    titles = [
        f"Open-Loop   P(φ) = {ol_p:.3f}",
        f"Closed-Loop  P(φ) = {cl_p:.3f}   ||K|| = {k_norm:.3f}",
    ]
    traces = [(ol_mean, ol_cov), (cl_mean, cl_cov)]

    for ax, (mean_trace, cov_trace), title in zip(axes, traces, titles):
        mu = mean_trace[0].numpy()
        Sigma = cov_trace[0].numpy()
        T = mu.shape[0] - 1

        _draw_env(ax, env)

        for t in range(0, T + 1, ellipse_every):
            S_xy = Sigma[t, :2, :2]
            if np.all(np.isfinite(S_xy)):
                ax.add_patch(
                    _cov_ellipse(
                        mu[t, :2],
                        S_xy,
                        n_std=2.0,
                        facecolor="steelblue",
                        alpha=0.15,
                        edgecolor="steelblue",
                        linewidth=0.8,
                    )
                )

        ax.plot(mu[:, 0], mu[:, 1], "o-", color="steelblue", ms=4, lw=1.5)
        ax.plot(mu[0, 0], mu[0, 1], "go", ms=8)
        ax.plot(mu[-1, 0], mu[-1, 1], "b*", ms=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
