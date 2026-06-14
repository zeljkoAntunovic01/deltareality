#!/usr/bin/env python3
"""
Inspect the provided computer vision assignment data using NumPy.

This is a shorter/faster version of inspect_data.py. It does not modify files.
It reports PLY structure, point/color statistics, trajectory matrix validity,
camera centers, camera axes, and optional world-space point statistics.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


VectorLike = Iterable[float]


@dataclass(frozen=True)
class PlyHeader:
    path: Path
    file_format: str
    vertex_count: int
    vertex_properties: list[tuple[str, str]]
    header_line_count: int

    @property
    def property_names(self) -> list[str]:
        return [name for _, name in self.vertex_properties]


def fmt_vec(values: VectorLike, precision: int = 5) -> str:
    return "(" + ", ".join(f"{float(x):.{precision}f}" for x in values) + ")"


def parse_ply_header(path: Path) -> PlyHeader:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file. First line was: {first!r}")

        file_format: str | None = None
        vertex_count: int | None = None
        vertex_properties: list[tuple[str, str]] = []
        current_element: str | None = None
        header_line_count = 1

        for raw_line in f:
            header_line_count += 1
            line = raw_line.strip()

            if line == "end_header":
                break
            if not line or line.startswith("comment"):
                continue

            parts = line.split()
            if parts[0] == "format":
                file_format = " ".join(parts[1:])
            elif parts[0] == "element":
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                prop_type = "list" if parts[1] == "list" else parts[1]
                prop_name = parts[-1] if parts[1] == "list" else parts[2]
                vertex_properties.append((prop_type, prop_name))
        else:
            raise ValueError(f"{path} ended before end_header")

    if file_format is None:
        raise ValueError(f"{path} has no PLY format line")
    if vertex_count is None:
        raise ValueError(f"{path} has no vertex element")

    return PlyHeader(
        path=path,
        file_format=file_format,
        vertex_count=vertex_count,
        vertex_properties=vertex_properties,
        header_line_count=header_line_count,
    )


def array_stats(names: list[str], values: np.ndarray) -> dict:
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {values.shape}")

    finite_mask = np.isfinite(values).all(axis=1)
    valid = values[finite_mask]

    if len(valid) == 0:
        nan_values = [float("nan")] * len(names)
        zero_values = [0] * len(names)
        return {
            "count": 0,
            "names": names,
            "min": dict(zip(names, nan_values)),
            "max": dict(zip(names, nan_values)),
            "mean": dict(zip(names, nan_values)),
            "std": dict(zip(names, nan_values)),
            "extent": dict(zip(names, nan_values)),
            "positive_count": dict(zip(names, zero_values)),
            "negative_count": dict(zip(names, zero_values)),
            "zero_count": dict(zip(names, zero_values)),
        }

    return {
        "count": int(valid.shape[0]),
        "names": names,
        "min": dict(zip(names, valid.min(axis=0).tolist())),
        "max": dict(zip(names, valid.max(axis=0).tolist())),
        "mean": dict(zip(names, valid.mean(axis=0).tolist())),
        "std": dict(zip(names, valid.std(axis=0).tolist())),
        "extent": dict(zip(names, np.ptp(valid, axis=0).tolist())),
        "positive_count": dict(zip(names, (valid > 0).sum(axis=0).tolist())),
        "negative_count": dict(zip(names, (valid < 0).sum(axis=0).tolist())),
        "zero_count": dict(zip(names, (valid == 0).sum(axis=0).tolist())),
    }


def load_ply_table(path: Path, header: PlyHeader) -> np.ndarray:
    data = np.loadtxt(
        path,
        dtype=np.float64,
        skiprows=header.header_line_count,
        max_rows=header.vertex_count,
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    expected_cols = len(header.vertex_properties)
    if data.shape[1] != expected_cols:
        raise ValueError(
            f"{path} has {data.shape[1]} data columns, expected {expected_cols}"
        )

    return data


def inspect_ply(path: Path, pose: np.ndarray | None = None) -> dict:
    header = parse_ply_header(path)
    names = header.property_names

    missing = [name for name in ("x", "y", "z") if name not in names]
    if missing:
        raise ValueError(f"{path} is missing required properties: {missing}")

    data = load_ply_table(path, header)
    xyz_indices = [names.index(name) for name in ("x", "y", "z")]
    xyz = data[:, xyz_indices]
    finite_xyz = np.isfinite(xyz).all(axis=1)

    result = {
        "path": str(path),
        "header": {
            "format": header.file_format,
            "vertex_count": header.vertex_count,
            "properties": [
                {"type": prop_type, "name": prop_name}
                for prop_type, prop_name in header.vertex_properties
            ],
            "header_line_count": header.header_line_count,
        },
        "invalid_lines": int(header.vertex_count - finite_xyz.sum()),
        "local_stats": array_stats(["x", "y", "z"], xyz),
    }

    if all(name in names for name in ("red", "green", "blue")):
        rgb_indices = [names.index(name) for name in ("red", "green", "blue")]
        result["color_stats"] = array_stats(
            ["red", "green", "blue"],
            data[:, rgb_indices],
        )

    if pose is not None:
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        world_xyz = xyz @ rotation.T + translation
        result["world_stats_assuming_camera_to_world"] = array_stats(
            ["x", "y", "z"],
            world_xyz,
        )

    return result


def parse_traj(path: Path) -> list[np.ndarray]:
    data = np.loadtxt(path, dtype=np.float64, comments="#", ndmin=2)

    if data.size == 0:
        return []
    if data.shape[1] != 16:
        raise ValueError(f"{path} has {data.shape[1]} columns, expected 16")

    return [row.reshape(4, 4) for row in data]


def inspect_pose(index: int, matrix: np.ndarray) -> dict:
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    cols = rotation.T

    return {
        "index": index,
        "matrix": matrix.tolist(),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "rotation_column_norms": np.linalg.norm(cols, axis=1).tolist(),
        "rotation_column_dot_products": {
            "x_dot_y": float(cols[0] @ cols[1]),
            "x_dot_z": float(cols[0] @ cols[2]),
            "y_dot_z": float(cols[1] @ cols[2]),
        },
        "camera_center_if_camera_to_world": translation.tolist(),
        "camera_center_if_world_to_camera": (-(rotation.T @ translation)).tolist(),
        "local_axes_in_world_if_camera_to_world": {
            "local_x_axis": cols[0].tolist(),
            "local_y_axis": cols[1].tolist(),
            "local_z_axis": cols[2].tolist(),
        },
    }


def inspect_traj(path: Path) -> dict:
    matrices = parse_traj(path)
    pose_reports = [inspect_pose(i + 1, matrix) for i, matrix in enumerate(matrices)]

    deltas = []
    for prev, curr in zip(pose_reports, pose_reports[1:]):
        a = np.asarray(prev["camera_center_if_camera_to_world"])
        b = np.asarray(curr["camera_center_if_camera_to_world"])
        delta = b - a
        deltas.append(
            {
                "from_pose": prev["index"],
                "to_pose": curr["index"],
                "delta": delta.tolist(),
                "distance": float(np.linalg.norm(delta)),
            }
        )

    return {
        "path": str(path),
        "pose_count": len(matrices),
        "poses": pose_reports,
        "camera_center_deltas_assuming_camera_to_world": deltas,
    }


def print_stats_block(title: str, stats: dict) -> None:
    names = stats["names"]

    print(f"\n  {title}")
    print(f"  count:  {stats['count']:,}")

    for name in names:
        count = stats["count"]
        pos = stats["positive_count"][name]
        neg = stats["negative_count"][name]
        zero = stats["zero_count"][name]
        pos_pct = 100.0 * pos / count if count else 0.0
        neg_pct = 100.0 * neg / count if count else 0.0
        zero_pct = 100.0 * zero / count if count else 0.0

        print(
            f"    {name}: "
            f"min={stats['min'][name]: .6f}, "
            f"max={stats['max'][name]: .6f}, "
            f"mean={stats['mean'][name]: .6f}, "
            f"std={stats['std'][name]: .6f}, "
            f"extent={stats['extent'][name]: .6f}, "
            f"+={pos_pct:5.1f}%, -={neg_pct:5.1f}%, 0={zero_pct:5.1f}%"
        )


def print_ply_report(report: dict) -> None:
    print("\n" + "=" * 80)
    print(f"PLY: {report['path']}")
    print("=" * 80)

    header = report["header"]
    print(f"format:        {header['format']}")
    print(f"vertex count:  {header['vertex_count']:,}")
    print(f"header lines:  {header['header_line_count']}")
    print(
        "properties:    "
        + ", ".join(f"{p['type']} {p['name']}" for p in header["properties"])
    )

    if report["invalid_lines"]:
        print(f"invalid lines: {report['invalid_lines']:,}")

    print_stats_block("Local point statistics", report["local_stats"])

    if "color_stats" in report:
        print_stats_block("Color statistics", report["color_stats"])

    if "world_stats_assuming_camera_to_world" in report:
        print_stats_block(
            "World point statistics, assuming traj pose is camera-to-world",
            report["world_stats_assuming_camera_to_world"],
        )


def print_traj_report(report: dict) -> None:
    print("\n" + "=" * 80)
    print(f"TRAJECTORY: {report['path']}")
    print("=" * 80)
    print(f"pose count: {report['pose_count']}")

    for pose in report["poses"]:
        print("\n" + "-" * 80)
        print(f"Pose {pose['index']}")
        print("-" * 80)

        print(f"rotation determinant: {pose['rotation_determinant']:.8f}")
        print(
            "rotation column norms: "
            + fmt_vec(pose["rotation_column_norms"], precision=8)
        )

        dots = pose["rotation_column_dot_products"]
        print(
            "rotation column dot products: "
            f"x.y={dots['x_dot_y']:.8f}, "
            f"x.z={dots['x_dot_z']:.8f}, "
            f"y.z={dots['y_dot_z']:.8f}"
        )

        print(
            "camera center if camera-to-world: "
            + fmt_vec(pose["camera_center_if_camera_to_world"])
        )
        print(
            "camera center if world-to-camera: "
            + fmt_vec(pose["camera_center_if_world_to_camera"])
        )

        axes = pose["local_axes_in_world_if_camera_to_world"]
        print("local axes in world, if camera-to-world:")
        print(f"  local X axis: {fmt_vec(axes['local_x_axis'])}")
        print(f"  local Y axis: {fmt_vec(axes['local_y_axis'])}")
        print(f"  local Z axis: {fmt_vec(axes['local_z_axis'])}")

    if report["camera_center_deltas_assuming_camera_to_world"]:
        print("\nCamera motion, assuming camera-to-world:")
        for delta in report["camera_center_deltas_assuming_camera_to_world"]:
            print(
                f"  pose {delta['from_pose']} -> {delta['to_pose']}: "
                f"delta={fmt_vec(delta['delta'])}, "
                f"distance={delta['distance']:.6f}"
            )


def image_index_from_ply_name(path: Path) -> int | None:
    match = re.search(r"image(\d+)\.ply$", path.name)
    return int(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect assignment PLY files and trajectory matrices with NumPy."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("viewer/ComputerVisionAssignment_Data/StreamingAssets"),
        help="Path to the StreamingAssets directory.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to save the full inspection report as JSON.",
    )
    parser.add_argument(
        "--no-world-stats",
        action="store_true",
        help="Skip applying trajectory poses to estimate world-space point ranges.",
    )

    args = parser.parse_args()

    root = args.root
    points_dir = root / "Points"
    traj_path = root / "traj.txt"

    if not root.exists():
        raise FileNotFoundError(f"StreamingAssets root not found: {root}")
    if not points_dir.exists():
        raise FileNotFoundError(f"Points directory not found: {points_dir}")
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    ply_paths = sorted(points_dir.glob("image*.ply"))
    if not ply_paths:
        raise FileNotFoundError(f"No image*.ply files found in {points_dir}")

    traj_report = inspect_traj(traj_path)
    print_traj_report(traj_report)

    poses = parse_traj(traj_path)
    ply_reports = []

    for ply_path in ply_paths:
        image_index = image_index_from_ply_name(ply_path)
        pose = None

        if (
            not args.no_world_stats
            and image_index is not None
            and 1 <= image_index <= len(poses)
        ):
            pose = poses[image_index - 1]

        print(f"\nLoading {ply_path}...")
        report = inspect_ply(ply_path, pose=pose)
        print_ply_report(report)
        ply_reports.append(report)

    full_report = {
        "root": str(root),
        "trajectory": traj_report,
        "ply_files": ply_reports,
    }

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)
        print(f"\nSaved JSON report to: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
