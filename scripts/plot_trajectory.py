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
    error_png = out_prefix.with_suffix(".error.png")

    title = f"KITTI trajectory, {n} frames, align={args.align}"
    plot_trajectory(ref_xyz, est_aligned, trajectory_png, title)
    plot_error(errors, error_png, f"Translation error, RMSE={summary['rmse']:.3f} m")

    print(f"frames: {n}")
    print(f"align: {args.align}")
    for key, value in summary.items():
        print(f"{key}: {value:.6f} m")
    print(f"wrote: {trajectory_png}")
    print(f"wrote: {error_png}")


if __name__ == "__main__":
    main()
