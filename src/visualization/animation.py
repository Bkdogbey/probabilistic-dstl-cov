"""Trajectory animation and live-optimization callback for the single-shot planner."""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from matplotlib.animation import FuncAnimation


def _get_helpers():
    """Lazy import of _draw_env to avoid circular dependency with visualization.__init__."""
    from visualization import _draw_env  # noqa: PLC0415

    return _draw_env


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _cov_ellipse_params(cov2x2: np.ndarray) -> tuple[float, float, float]:
    """Return (angle_deg, width, height) for a 2×2 covariance ellipse at 2σ."""
    vals, vecs = np.linalg.eigh(cov2x2)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w = 2 * 2.0 * np.sqrt(np.abs(vals[0]))
    h = 2 * 2.0 * np.sqrt(np.abs(vals[1]))
    return float(angle), float(w), float(h)


# ---------------------------------------------------------------------------
# Post-hoc trajectory animation
# ---------------------------------------------------------------------------


def animate_trajectory(
    mean_trace,
    cov_trace,
    env,
    filename: str = "trajectory.gif",
    dt: float = 0.1,
    title: str = "Trajectory",
    fps: int = 10,
    robot_dims: list | None = None,
) -> None:
    """
    Animate the planned belief trajectory and save as a GIF.

    Args:
        mean_trace:  [1, T+1, nx] or [T+1, nx]   — output of planner.solve()
        cov_trace:   [1, T+1, nx, nx] or [T+1, nx, nx]
        env:         Environment instance
        filename:    output path (must end in .gif)
        dt:          timestep [s] — used in the time label
        title:       figure title
        fps:         frames per second for the saved animation
        robot_dims:  [length, width] — if given, draws a rotated rectangle for ego
    """
    mu = (
        mean_trace[0].detach().cpu().numpy()
        if mean_trace.dim() == 3
        else mean_trace.detach().cpu().numpy()
    )
    Sigma = (
        cov_trace[0].detach().cpu().numpy()
        if cov_trace.dim() == 4
        else cov_trace.detach().cpu().numpy()
    )
    T = mu.shape[0] - 1

    _draw_env = _get_helpers()
    fig, ax = plt.subplots(figsize=(10, 5))
    _draw_env(ax, env)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Moving obstacle patches — one per moving obstacle
    moving_patches = []
    for obs in getattr(env, "moving_obstacles", []):
        xt_raw = obs["x_traj"]
        yt_raw = obs["y_traj"]
        x_traj = np.asarray(xt_raw.cpu() if hasattr(xt_raw, "cpu") else xt_raw)
        y_traj = np.asarray(yt_raw.cpu() if hasattr(yt_raw, "cpu") else yt_raw)
        w, h = obs["width"], obs["height"]
        rect = patches.Rectangle(
            (float(x_traj[0]) - w / 2, float(y_traj[0]) - h / 2),
            w,
            h,
            facecolor="orange",
            edgecolor="darkorange",
            alpha=0.6,
            linestyle="--",
            zorder=6,
        )
        ax.add_patch(rect)
        moving_patches.append((rect, w, h, x_traj, y_traj))

    # Ego representation
    if robot_dims:
        ego_patch = patches.Rectangle(
            (0, 0),
            robot_dims[0],
            robot_dims[1],
            facecolor="steelblue",
            edgecolor="darkblue",
            alpha=0.8,
            zorder=10,
        )
        ax.add_patch(ego_patch)
        ego_dot = None
    else:
        (ego_dot,) = ax.plot([], [], "o", color="steelblue", ms=8, zorder=10, label="Ego")
        ego_patch = None

    (trail,) = ax.plot([], [], "-", color="steelblue", lw=1.5, alpha=0.6, zorder=9)

    ellipse_patch = patches.Ellipse(
        (0, 0),
        width=0,
        height=0,
        angle=0,
        facecolor="steelblue",
        edgecolor="steelblue",
        alpha=0.2,
        zorder=8,
    )
    ax.add_patch(ellipse_patch)

    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
    )

    def init():
        trail.set_data([], [])
        if ego_dot:
            ego_dot.set_data([], [])
        if ego_patch:
            ego_patch.set_visible(False)
        ellipse_patch.set_width(0)
        ellipse_patch.set_height(0)
        time_text.set_text("")
        actors = [trail, ellipse_patch, time_text]
        if ego_dot:
            actors.append(ego_dot)
        if ego_patch:
            actors.append(ego_patch)
        actors.extend(r for r, *_ in moving_patches)
        return actors

    def update(frame):
        x, y = float(mu[frame, 0]), float(mu[frame, 1])

        trail.set_data(mu[: frame + 1, 0], mu[: frame + 1, 1])

        if ego_dot:
            ego_dot.set_data([x], [y])

        if ego_patch:
            ego_patch.set_visible(True)
            if frame < T:
                dx = mu[frame + 1, 0] - x
                dy = mu[frame + 1, 1] - y
            else:
                dx = x - mu[frame - 1, 0]
                dy = y - mu[frame - 1, 1]
            theta = np.degrees(np.arctan2(dy, dx))
            rl, rw = robot_dims
            t = transforms.Affine2D().translate(-rl / 2, -rw / 2).rotate_deg(theta).translate(x, y)
            ego_patch.set_transform(t + ax.transData)

        cov2x2 = Sigma[frame, :2, :2]
        if np.all(np.isfinite(cov2x2)):
            angle, ew, eh = _cov_ellipse_params(cov2x2)
            ellipse_patch.set_center((x, y))
            ellipse_patch.set_width(ew)
            ellipse_patch.set_height(eh)
            ellipse_patch.set_angle(angle)

        time_text.set_text(f"t = {frame * dt:.1f} s  (step {frame}/{T})")

        for rect, w, h, x_traj, y_traj in moving_patches:
            idx = min(frame, len(x_traj) - 1)
            rect.set_xy((float(x_traj[idx]) - w / 2, float(y_traj[idx]) - h / 2))

        actors = [trail, ellipse_patch, time_text]
        if ego_dot:
            actors.append(ego_dot)
        if ego_patch:
            actors.append(ego_patch)
        actors.extend(r for r, *_ in moving_patches)
        return actors

    ani = FuncAnimation(fig, update, frames=range(T + 1), init_func=init, blit=False, interval=100)

    if filename:
        print(f"  Saving animation → {filename} ...", end=" ", flush=True)
        try:
            ani.save(filename, writer="pillow", fps=fps)
            print("done.")
        except Exception as exc:
            print(f"\n  Warning: could not save animation ({exc}). Displaying instead.")
            plt.show()
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Live optimization callback
# ---------------------------------------------------------------------------


def make_live_callback(env, title: str = ""):
    """
    Return a callback for ``planner.solve(callback=...)`` that updates a live
    matplotlib figure each time it is called.

    The returned function has signature::

        callback(k, mean_trace, cov_trace, loss, p_lower)

    where ``k`` is the iteration index.  It is safe to pass this to
    ``solve(callback=cb, callback_every=25)``; the function sets up the figure
    lazily on its first call so that importing this module never creates a window.
    """
    state: dict = {}  # lazy initialisation

    def _setup():
        try:
            matplotlib.use("TkAgg")
        except Exception:
            pass
        plt.ion()
        fig, (ax_map, ax_p) = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle(title or "Optimization Live View", fontsize=10)

        # Left panel — trajectory map
        _draw_env_fn = _get_helpers()
        _draw_env_fn(ax_map, env)
        ax_map.set_aspect("equal")
        ax_map.set_xlabel("x [m]")
        ax_map.set_ylabel("y [m]")
        ax_map.grid(True, alpha=0.3)

        (traj_line,) = ax_map.plot([], [], "o-", color="steelblue", ms=3, lw=1.5, zorder=9)
        ellipse_live = patches.Ellipse(
            (0, 0),
            width=0,
            height=0,
            angle=0,
            facecolor="steelblue",
            edgecolor="steelblue",
            alpha=0.25,
            zorder=8,
        )
        ax_map.add_patch(ellipse_live)

        # Right panel — P(phi) vs iteration
        ax_p.set_xlim(0, 500)
        ax_p.set_ylim(0, 1.05)
        ax_p.set_xlabel("iteration")
        ax_p.set_ylabel("P(φ) lower bound")
        ax_p.set_title("Satisfaction probability")
        ax_p.axhline(0.95, color="red", lw=0.8, linestyle="--", label="α")
        ax_p.grid(True, alpha=0.3)
        (p_line,) = ax_p.plot([], [], color="steelblue", lw=1.5)

        state["fig"] = fig
        state["ax_map"] = ax_map
        state["ax_p"] = ax_p
        state["traj_line"] = traj_line
        state["ellipse"] = ellipse_live
        state["p_line"] = p_line
        state["iters"] = []
        state["p_vals"] = []

    def callback(k: int, mean_trace, cov_trace, loss: float, p_lower: float):
        if not state:
            try:
                _setup()
            except Exception:
                return  # headless / no display — silently skip

        mu = mean_trace[0].cpu().numpy() if mean_trace.dim() == 3 else mean_trace.cpu().numpy()
        Sigma = cov_trace[0].cpu().numpy() if cov_trace.dim() == 4 else cov_trace.cpu().numpy()

        state["traj_line"].set_data(mu[:, 0], mu[:, 1])

        cov2x2 = Sigma[-1, :2, :2]
        if np.all(np.isfinite(cov2x2)):
            angle, ew, eh = _cov_ellipse_params(cov2x2)
            ell = state["ellipse"]
            ell.set_center((float(mu[-1, 0]), float(mu[-1, 1])))
            ell.set_width(ew)
            ell.set_height(eh)
            ell.set_angle(angle)

        state["iters"].append(k)
        state["p_vals"].append(p_lower)
        state["p_line"].set_data(state["iters"], state["p_vals"])
        if k > 0:
            state["ax_p"].set_xlim(0, max(k + 10, 50))

        state["ax_map"].set_title(f"iter {k:4d} | P(φ)={p_lower:.3f} | loss={loss:.4f}")

        try:
            plt.pause(0.01)
        except Exception:
            pass

    return callback
