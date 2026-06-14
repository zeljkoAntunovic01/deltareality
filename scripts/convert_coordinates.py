#!/usr/bin/env python3
"""
Convert assignment point clouds and trajectory poses with a proposed transform.

The default model is a coordinate-system conversion:

    p_viewer = C @ p_source
    T_viewer = C @ T_source @ inv(C)

where C is the proposed source-to-viewer homogeneous transform. The script writes
replacement versions of image1.ply, image2.ply, image3.ply, and traj.txt into an
output folder without modifying the original viewer data.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path("viewer/ComputerVisionAssignment_Data/StreamingAssets")
DEFAULT_OUTPUT_ROOT = Path("outputs/converted")


@dataclass(frozen=True)
class PlyHeader:
    lines: list[str]
    vertex_count: int
    vertex_properties: list[tuple[str, str]]

    @property
    def property_names(self) -> list[str]:
        return [name for _, name in self.vertex_properties]


def parse_ply_header(path: Path) -> PlyHeader:
    lines: list[str] = []
    vertex_count: int | None = None
    vertex_properties: list[tuple[str, str]] = []
    current_element: str | None = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
        if first.strip() != "ply":
            raise ValueError(f"{path} is not a PLY file")

        lines.append(first)

        for raw_line in f:
            lines.append(raw_line)
            line = raw_line.strip()

            if line == "end_header":
                break
            if not line or line.startswith("comment"):
                continue

            parts = line.split()
            if parts[0] == "element":
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                prop_type = "list" if parts[1] == "list" else parts[1]
                prop_name = parts[-1] if parts[1] == "list" else parts[2]
                vertex_properties.append((prop_type, prop_name))
        else:
            raise ValueError(f"{path} ended before end_header")

    if vertex_count is None:
        raise ValueError(f"{path} has no vertex element")

    return PlyHeader(
        lines=lines,
        vertex_count=vertex_count,
        vertex_properties=vertex_properties,
    )


def parse_matrix_values(values: list[float]) -> np.ndarray:
    if len(values) == 9:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.asarray(values, dtype=np.float64).reshape(3, 3)
        return matrix

    if len(values) == 12:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :4] = np.asarray(values, dtype=np.float64).reshape(3, 4)
        return matrix

    if len(values) == 16:
        return np.asarray(values, dtype=np.float64).reshape(4, 4)

    raise ValueError(
        f"Expected 9, 12, or 16 matrix values, got {len(values)}"
    )


def load_matrix_from_file(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    values = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)]
    return parse_matrix_values(values)


def format_for_property(prop_type: str) -> str:
    if prop_type in {"char", "uchar", "short", "ushort", "int", "uint"}:
        return "%d"
    if prop_type in {"float", "double"}:
        return "%.9g"
    raise ValueError(f"Unsupported PLY property type: {prop_type}")


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    rotation_or_linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    transformed = points @ rotation_or_linear.T + translation

    w_row = matrix[3, :]
    if not np.allclose(w_row, [0.0, 0.0, 0.0, 1.0]):
        weights = points @ w_row[:3] + w_row[3]
        transformed = transformed / weights[:, None]

    return transformed


def convert_ply(input_path: Path, output_path: Path, matrix: np.ndarray) -> None:
    header = parse_ply_header(input_path)
    names = header.property_names

    missing = [name for name in ("x", "y", "z") if name not in names]
    if missing:
        raise ValueError(f"{input_path} is missing required properties: {missing}")

    data = np.loadtxt(
        input_path,
        dtype=np.float64,
        skiprows=len(header.lines),
        max_rows=header.vertex_count,
    )
    if data.ndim == 1:
        data = data.reshape(1, -1)

    expected_cols = len(header.vertex_properties)
    if data.shape[1] != expected_cols:
        raise ValueError(
            f"{input_path} has {data.shape[1]} columns, expected {expected_cols}"
        )

    xyz_indices = [names.index(name) for name in ("x", "y", "z")]
    data[:, xyz_indices] = transform_points(data[:, xyz_indices], matrix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    formats = [format_for_property(prop_type) for prop_type, _ in header.vertex_properties]

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(line.rstrip("\r\n") + "\n" for line in header.lines)
        np.savetxt(f, data, fmt=formats)


def load_traj(path: Path) -> np.ndarray:
    traj = np.loadtxt(path, dtype=np.float64, comments="#", ndmin=2)
    if traj.shape[1] != 16:
        raise ValueError(f"{path} has {traj.shape[1]} columns, expected 16")
    return traj.reshape(-1, 4, 4)


def convert_traj(
    input_path: Path,
    output_path: Path,
    matrix: np.ndarray,
    mode: str,
) -> None:
    poses = load_traj(input_path)
    inv_matrix = np.linalg.inv(matrix)

    if mode == "conjugate":
        converted = np.asarray([matrix @ pose @ inv_matrix for pose in poses])
    elif mode == "left":
        converted = np.asarray([matrix @ pose for pose in poses])
    elif mode == "right":
        converted = np.asarray([pose @ inv_matrix for pose in poses])
    else:
        raise ValueError(f"Unsupported trajectory mode: {mode}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, converted.reshape(len(converted), 16), fmt="%.18e")


def copy_optional_images(input_points_dir: Path, output_points_dir: Path) -> None:
    output_points_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(input_points_dir.glob("*.png")):
        shutil.copy2(image_path, output_points_dir / image_path.name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a proposed source-to-viewer coordinate transform to assignment "
            "PLY files and traj.txt."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Input StreamingAssets directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output StreamingAssets directory for converted files.",
    )
    parser.add_argument(
        "--matrix",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Source-to-viewer transform in row-major order. Accepts 9 values "
            "(3x3), 12 values (3x4), or 16 values (4x4)."
        ),
    )
    parser.add_argument(
        "--matrix-file",
        type=Path,
        default=None,
        help="Text file containing 9, 12, or 16 matrix values.",
    )
    parser.add_argument(
        "--traj-mode",
        choices=["conjugate", "left", "right"],
        default="conjugate",
        help=(
            "How to transform poses. Default 'conjugate' uses C @ T @ inv(C), "
            "which is the usual coordinate-basis conversion."
        ),
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Also copy unchanged PNG files into the output Points directory.",
    )

    args = parser.parse_args()

    if (args.matrix is None) == (args.matrix_file is None):
        raise ValueError("Provide exactly one of --matrix or --matrix-file")

    matrix = (
        parse_matrix_values(args.matrix)
        if args.matrix is not None
        else load_matrix_from_file(args.matrix_file)
    )

    root = args.root
    points_dir = root / "Points"
    traj_path = root / "traj.txt"
    output_points_dir = args.output_root / "Points"
    output_traj_path = args.output_root / "traj.txt"

    if not points_dir.exists():
        raise FileNotFoundError(f"Points directory not found: {points_dir}")
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    ply_paths = sorted(points_dir.glob("image*.ply"))
    if not ply_paths:
        raise FileNotFoundError(f"No image*.ply files found in {points_dir}")

    print("Using source-to-viewer matrix:")
    print(matrix)
    print(f"\nWriting converted files to: {args.output_root}")

    for input_ply in ply_paths:
        output_ply = output_points_dir / input_ply.name
        print(f"Converting {input_ply} -> {output_ply}")
        convert_ply(input_ply, output_ply, matrix)

    print(f"Converting {traj_path} -> {output_traj_path}")
    convert_traj(traj_path, output_traj_path, matrix, args.traj_mode)

    if args.copy_images:
        print(f"Copying PNG files to {output_points_dir}")
        copy_optional_images(points_dir, output_points_dir)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
