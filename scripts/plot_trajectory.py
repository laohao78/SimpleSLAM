#!/usr/bin/env python3
"""Plot KITTI-format ground truth and estimated trajectories.

Example:
    python3 scripts/plot_trajectory.py \
        KITTI/dataset/poses/00.txt KITTI/results/kitti00_lo_fast.kitti.txt \
        --out-prefix KITTI/results/kitti00_vis --align se3
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


def load_kitti(path: str) -> np.ndarray:
    poses = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            values = np.fromstring(line, sep=" ")
            if values.size == 0:
                continue
            if values.size != 12:
                raise ValueError(f"{path}: expected 12 values per KITTI pose line")
            T = np.eye(4)
            T[:3, :4] = values.reshape(3, 4)
            poses.append(T)
    if not poses:
        raise ValueError(f"{path}: no poses loaded")
    return np.stack(poses)


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool):
    """Find s, R, t so dst ~= s * R * src + t."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    x = src - mu_src
    y = dst - mu_dst
    cov = (y.T @ x) / n
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1
    r = u @ sign @ vt
    if with_scale:
        var_src = (x * x).sum() / n
        scale = np.trace(np.diag(d) @ sign) / var_src
    else:
        scale = 1.0
    t = mu_dst - scale * r @ mu_src
    return scale, r, t


def align_points(est_xyz: np.ndarray, ref_xyz: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return est_xyz.copy()
    scale, rotation, translation = umeyama(
        est_xyz, ref_xyz, with_scale=(mode == "sim3")
    )
    return (scale * (rotation @ est_xyz.T)).T + translation


def stats(errors: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)),
        "std": float(np.std(errors)),
    }


def plot_trajectory(ref_xyz: np.ndarray, est_xyz: np.ndarray, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ax.plot(ref_xyz[:, 0], ref_xyz[:, 2], label="KITTI ground truth", linewidth=2.0)
    ax.plot(est_xyz[:, 0], est_xyz[:, 2], label="SimpleSLAM LO", linewidth=1.8)
    ax.scatter(ref_xyz[0, 0], ref_xyz[0, 2], s=35, label="start")
    ax.scatter(ref_xyz[-1, 0], ref_xyz[-1, 2], s=35, label="end")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_line_collection(xz: np.ndarray, values: np.ndarray) -> LineCollection:
    segments = np.stack([xz[:-1], xz[1:]], axis=1)
    collection = LineCollection(segments, cmap="viridis")
    collection.set_array(values[:-1])
    collection.set_linewidth(2.4)
    return collection


def xz_bounds(*xyz_arrays: np.ndarray):
    points = np.concatenate([xyz[:, [0, 2]] for xyz in xyz_arrays], axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    span = np.maximum(maxs - mins, 1.0)
    margin = 0.06 * span
    return (mins[0] - margin[0], maxs[0] + margin[0]), (
        mins[1] - margin[1],
        maxs[1] + margin[1],
    )


def style_xz_axis(ax, xlim, zlim) -> None:
    ax.set_xlim(xlim)
    ax.set_ylim(zlim)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)


def add_direction_arrows(
    ax,
    xyz: np.ndarray,
    every: int,
    color: str,
    alpha: float = 0.75,
) -> None:
    if every <= 0:
        return
    xz = xyz[:, [0, 2]]
    step = max(1, every // 8)
    for idx in range(every, len(xz) - step, every):
        start = xz[idx]
        end = xz[idx + step]
        if np.linalg.norm(end - start) < 1e-6:
            continue
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "lw": 1.2,
                "alpha": alpha,
                "mutation_scale": 10,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )


def add_frame_labels(ax, xyz: np.ndarray, every: int) -> None:
    if every <= 0:
        return
    xz = xyz[:, [0, 2]]
    indices = list(range(0, len(xz), every))
    if indices[-1] != len(xz) - 1:
        indices.append(len(xz) - 1)
    for idx in indices:
        ax.text(
            xz[idx, 0],
            xz[idx, 1],
            str(idx),
            fontsize=7,
            color="black",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )


def plot_direction(
    ref_xyz: np.ndarray,
    est_xyz: np.ndarray,
    out_path: Path,
    title: str,
    arrow_every: int,
    label_every: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ref_xz = ref_xyz[:, [0, 2]]
    frames = np.arange(ref_xyz.shape[0])
    progress_line = make_line_collection(ref_xz, frames)
    ax.add_collection(progress_line)
    ax.plot(
        est_xyz[:, 0],
        est_xyz[:, 2],
        color="tab:orange",
        label="SimpleSLAM LO",
        linewidth=1.4,
        alpha=0.75,
    )
    ax.scatter(ref_xyz[0, 0], ref_xyz[0, 2], s=42, color="tab:green", label="start")
    ax.scatter(ref_xyz[-1, 0], ref_xyz[-1, 2], s=48, color="tab:red", label="end")
    add_direction_arrows(ax, ref_xyz, arrow_every, color="black")
    add_frame_labels(ax, ref_xyz, label_every)
    ax.update_datalim(ref_xz)
    ax.autoscale_view()
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    colorbar = fig.colorbar(progress_line, ax=ax, pad=0.01)
    colorbar.set_label("frame")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_time_colored_path(
    ax,
    xyz: np.ndarray,
    frames: np.ndarray,
    title: str,
    arrow_every: int,
    label_every: int,
    xlim,
    zlim,
) -> LineCollection:
    line = make_line_collection(xyz[:, [0, 2]], frames)
    ax.add_collection(line)
    ax.scatter(xyz[0, 0], xyz[0, 2], s=42, color="tab:green", zorder=3)
    ax.scatter(xyz[-1, 0], xyz[-1, 2], s=48, color="tab:red", zorder=3)
    add_direction_arrows(ax, xyz, arrow_every, color="black")
    add_frame_labels(ax, xyz, label_every)
    ax.set_title(title)
    style_xz_axis(ax, xlim, zlim)
    return line


def plot_direction_compare(
    ref_xyz: np.ndarray,
    est_xyz: np.ndarray,
    out_path: Path,
    title: str,
    arrow_every: int,
    label_every: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), dpi=160, constrained_layout=True)
    frames = np.arange(ref_xyz.shape[0])
    xlim, zlim = xz_bounds(ref_xyz, est_xyz)

    line = draw_time_colored_path(
        axes[0],
        ref_xyz,
        frames,
        "KITTI ground truth",
        arrow_every,
        label_every,
        xlim,
        zlim,
    )
    draw_time_colored_path(
        axes[1],
        est_xyz,
        frames,
        "SimpleSLAM LO aligned",
        arrow_every,
        label_every,
        xlim,
        zlim,
    )

    handles = [
        Line2D([0], [0], color="tab:green", marker="o", lw=0, label="start"),
        Line2D([0], [0], color="tab:red", marker="o", lw=0, label="end"),
        Line2D([0], [0], color="black", lw=1.3, label="arrow = driving direction"),
    ]
    axes[0].legend(handles=handles, loc="best")
    fig.suptitle(title)
    colorbar = fig.colorbar(line, ax=axes, pad=0.01)
    colorbar.set_label("frame")
    fig.savefig(out_path)
    plt.close(fig)


def plot_segments(
    ref_xyz: np.ndarray,
    est_xyz: np.ndarray,
    out_path: Path,
    title: str,
    segment_count: int,
) -> None:
    segment_count = max(1, segment_count)
    rows = int(np.ceil(segment_count / 2))
    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(11, max(4.0, rows * 3.2)),
        dpi=160,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    xlim, zlim = xz_bounds(ref_xyz, est_xyz)
    edges = np.linspace(0, ref_xyz.shape[0] - 1, segment_count + 1, dtype=int)

    for seg_idx, ax in enumerate(axes):
        if seg_idx >= segment_count:
            ax.axis("off")
            continue

        start = edges[seg_idx]
        end = edges[seg_idx + 1]
        if seg_idx + 1 == segment_count:
            end = ref_xyz.shape[0] - 1
        segment = slice(start, end + 1)

        ax.plot(
            ref_xyz[:, 0],
            ref_xyz[:, 2],
            color="0.78",
            linewidth=1.0,
            label="full ground truth" if seg_idx == 0 else None,
        )
        ax.plot(
            est_xyz[:, 0],
            est_xyz[:, 2],
            color="peachpuff",
            linewidth=1.0,
            label="full SimpleSLAM" if seg_idx == 0 else None,
        )
        ax.plot(
            ref_xyz[segment, 0],
            ref_xyz[segment, 2],
            color="tab:blue",
            linewidth=2.7,
            label="current ground truth" if seg_idx == 0 else None,
        )
        ax.plot(
            est_xyz[segment, 0],
            est_xyz[segment, 2],
            color="tab:orange",
            linewidth=2.3,
            label="current SimpleSLAM" if seg_idx == 0 else None,
        )

        local_every = max(1, (end - start) // 3)
        add_direction_arrows(ax, ref_xyz[segment], local_every, color="tab:blue", alpha=0.9)
        ax.scatter(ref_xyz[start, 0], ref_xyz[start, 2], s=32, color="tab:green", zorder=3)
        ax.scatter(ref_xyz[end, 0], ref_xyz[end, 2], s=36, color="tab:red", zorder=3)
        ax.text(
            ref_xyz[start, 0],
            ref_xyz[start, 2],
            str(start),
            fontsize=7,
            ha="right",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )
        ax.text(
            ref_xyz[end, 0],
            ref_xyz[end, 2],
            str(end),
            fontsize=7,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )
        ax.set_title(f"frames {start}-{end}")
        style_xz_axis(ax, xlim, zlim)

    axes[0].legend(loc="best")
    fig.suptitle(title)
    fig.savefig(out_path)
    plt.close(fig)


def plot_error(errors: np.ndarray, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=160)
    ax.plot(np.arange(errors.size), errors, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("frame")
    ax.set_ylabel("translation error [m]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize KITTI trajectories")
    parser.add_argument("reference", help="KITTI ground-truth pose file")
    parser.add_argument("estimate", help="KITTI estimated pose file")
    parser.add_argument("--out-prefix", default="KITTI/results/trajectory_vis")
    parser.add_argument("--align", choices=["none", "se3", "sim3"], default="se3")
    parser.add_argument(
        "--arrow-every",
        type=int,
        default=300,
        help="Draw a direction arrow every N frames in the direction plot; 0 disables arrows",
    )
    parser.add_argument(
        "--label-every",
        type=int,
        default=500,
        help="Draw a frame label every N frames in the direction plot; 0 disables labels",
    )
    parser.add_argument(
        "--segments",
        type=int,
        default=6,
        help="Number of driving-order segment panels to draw",
    )
    args = parser.parse_args()

    ref = load_kitti(args.reference)
    est = load_kitti(args.estimate)
    n = min(len(ref), len(est))
    ref = ref[:n]
    est = est[:n]

    ref_xyz = ref[:, :3, 3]
    est_xyz = est[:, :3, 3]
    est_aligned = align_points(est_xyz, ref_xyz, args.align)
    errors = np.linalg.norm(est_aligned - ref_xyz, axis=1)
    summary = stats(errors)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    trajectory_png = out_prefix.with_suffix(".trajectory.png")
    direction_png = out_prefix.with_suffix(".direction.png")
    direction_compare_png = out_prefix.with_suffix(".direction_compare.png")
    segments_png = out_prefix.with_suffix(".segments.png")
    error_png = out_prefix.with_suffix(".error.png")

    title = f"KITTI trajectory, {n} frames, align={args.align}"
    plot_trajectory(ref_xyz, est_aligned, trajectory_png, title)
    plot_direction(
        ref_xyz,
        est_aligned,
        direction_png,
        f"KITTI driving order, {n} frames, align={args.align}",
        args.arrow_every,
        args.label_every,
    )
    plot_direction_compare(
        ref_xyz,
        est_aligned,
        direction_compare_png,
        f"KITTI driving order compare, {n} frames, align={args.align}",
        args.arrow_every,
        args.label_every,
    )
    plot_segments(
        ref_xyz,
        est_aligned,
        segments_png,
        f"KITTI driving order by segments, {n} frames, align={args.align}",
        args.segments,
    )
    plot_error(errors, error_png, f"Translation error, RMSE={summary['rmse']:.3f} m")

    print(f"frames: {n}")
    print(f"align: {args.align}")
    for key, value in summary.items():
        print(f"{key}: {value:.6f} m")
    print(f"wrote: {trajectory_png}")
    print(f"wrote: {direction_png}")
    print(f"wrote: {direction_compare_png}")
    print(f"wrote: {segments_png}")
    print(f"wrote: {error_png}")


if __name__ == "__main__":
    main()
