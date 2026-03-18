from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .wrapper import (
    CPP_PROJECT_DIR,
    build_surface_filling_curve_flows,
    run_capsule_scene,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run autorigami C++ binaries through a Python wrapper"
    )
    parser.add_argument(
        "--binary",
        choices=["curve_on_surface", "mesh_deformation"],
        default="curve_on_surface",
        help="C++ executable to run",
    )
    parser.add_argument("--build", action="store_true", help="Build C++ binaries before running")
    parser.add_argument(
        "--build-dir", default="build", help="CMake build directory under the C++ project"
    )
    parser.add_argument("--jobs", type=int, default=6, help="Parallel build jobs")
    parser.add_argument("--radius", type=float, default=0.1, help="Scene radius value")
    parser.add_argument("--h", type=float, default=0.02, help="Scene h value")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to mesh model file; defaults to packaged capsule asset",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="Base outputs directory (defaults to <repo>/outputs)",
    )
    parser.add_argument(
        "--disable-execute-only",
        action="store_true",
        help="Do not write 'excecute_only' to the generated scene",
    )
    parser.add_argument(
        "--arg",
        action="append",
        dest="extra_args",
        default=[],
        help="Extra argument passed through to the C++ executable (repeatable)",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.build:
        build_result = build_surface_filling_curve_flows(
            build_dir=args.build_dir,
            jobs=args.jobs,
            source_dir=CPP_PROJECT_DIR,
        )
        if build_result.stdout:
            print(build_result.stdout, end="")
        if build_result.stderr:
            print(build_result.stderr, end="")

    run_paths, process = run_capsule_scene(
        binary_name=args.binary,
        model_path=args.model,
        build_dir=args.build_dir,
        source_dir=CPP_PROJECT_DIR,
        base_outputs_dir=args.outputs_dir,
        radius=args.radius,
        h=args.h,
        execute_only=not args.disable_execute_only,
        extra_args=args.extra_args,
        check=False,
        capture_output=True,
    )

    print(f"Run completed in: {run_paths.output_dir}")
    print(f"Generated scene: {run_paths.scene_file}")
    print(f"Copied mesh: {run_paths.mesh_file}")
    if process.returncode != 0:
        if process.stdout:
            print(process.stdout, end="")
        if process.stderr:
            print(process.stderr, end="", file=sys.stderr)
        print(f"Executable exited with code {process.returncode}", file=sys.stderr)
        return process.returncode

    if process.stdout:
        print(process.stdout, end="")
    if process.stderr:
        print(process.stderr, end="", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
