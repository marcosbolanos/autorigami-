from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .wrapper import (
    CPP_PROJECT_DIR,
    build_surface_filling_curve_flows,
    generate_beziers_for_run,
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
        "--nm-per-unit",
        type=float,
        default=None,
        help="Nanometers per mesh unit; enables physical scene parameters and Bezier validation",
    )
    parser.add_argument("--rmin-nm", type=float, default=None, help="Physical minimum radius in nm")
    parser.add_argument(
        "--init-mode",
        choices=["rings", "legacy_triangle", "staged_triangle"],
        default=None,
        help="Curve initialization strategy",
    )
    parser.add_argument(
        "--init-rings",
        type=int,
        default=None,
        help="Initialize this many parallel ring curves",
    )
    parser.add_argument(
        "--init-ring-vertices",
        type=int,
        default=None,
        help="Vertex count used for each initialized ring",
    )
    parser.add_argument(
        "--init-ring-radius-nm",
        type=float,
        default=None,
        help="Initial ring radius in nanometers",
    )
    parser.add_argument(
        "--init-ring-spacing-nm",
        type=float,
        default=None,
        help="Initial ring spacing in nanometers",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum execute-only optimization iterations",
    )
    parser.add_argument(
        "--init-stage-iterations",
        type=int,
        default=None,
        help="Iterations per staged triangle before inserting the next one",
    )
    parser.add_argument(
        "--keep-previous-active",
        action="store_true",
        help="For staged triangle init, keep previous triangles active instead of freezing them",
    )
    parser.add_argument(
        "--init-stage-clearance-nm",
        type=float,
        default=None,
        help="Minimum nm clearance used when placing staged triangles",
    )
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
    parser.add_argument(
        "--disable-bezier",
        action="store_true",
        help="Do not generate Bezier outputs from the latest exported curve OBJ",
    )
    parser.add_argument(
        "--skip-bezier-validation",
        action="store_true",
        help="Generate Bezier outputs without validation metrics",
    )
    parser.add_argument(
        "--keep-rejected",
        action="store_true",
        help="Keep output folders for failed or non-validated runs instead of deleting them",
    )
    parser.add_argument(
        "--bezier-validation-tol-nm",
        type=float,
        default=0.01,
        help="Bezier validation tolerance in nanometers when validation is enabled",
    )
    parser.add_argument(
        "--min-separation-nm",
        type=float,
        default=2.6,
        help="Minimum allowed separation between distinct curve components in nanometers",
    )
    parser.add_argument(
        "--bezier-tube-radius-nm",
        type=float,
        default=0.5,
        help="Physical radius for the rendered Bezier tube GLB geometry",
    )
    parser.add_argument(
        "--bezier-glb-tessellation-tol-nm",
        type=float,
        default=0.005,
        help="Adaptive subdivision tolerance in nanometers for the Bezier GLB preview mesh",
    )
    parser.add_argument(
        "--bezier-glb-tessellation-max-depth",
        type=int,
        default=9,
        help="Maximum adaptive subdivision depth for the Bezier GLB preview mesh",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    scene_lines_extra: list[str] = []

    if args.rmin_nm is not None:
        scene_lines_extra.append(f"rmin_nm {args.rmin_nm}")
    if args.nm_per_unit is not None:
        scene_lines_extra.append(f"nm_per_unit {args.nm_per_unit}")
    if args.rmin_nm is not None and args.nm_per_unit is not None:
        scene_lines_extra.extend(
            [
                "curvature_constraint barrier",
                "curvature_barrier 1.0",
                "curvature_barrier_epsilon 1e-6",
                f"curvature_barrier_threshold {args.nm_per_unit / args.rmin_nm}",
                f"curvature_barrier_min_length {args.rmin_nm / args.nm_per_unit}",
            ]
        )
    if args.init_mode is not None:
        scene_lines_extra.append(f"init_mode {args.init_mode}")
    if args.init_rings is not None:
        scene_lines_extra.append(f"init_triangles {args.init_rings}")
    if args.init_stage_iterations is not None:
        scene_lines_extra.append(f"init_stage_iterations {args.init_stage_iterations}")
    if args.keep_previous_active:
        scene_lines_extra.append("init_keep_previous_active")
    if args.init_ring_vertices is not None:
        scene_lines_extra.append(f"init_ring_vertices {args.init_ring_vertices}")
    if args.init_ring_radius_nm is not None:
        scene_lines_extra.append(f"init_ring_radius_nm {args.init_ring_radius_nm}")
    if args.init_ring_spacing_nm is not None:
        scene_lines_extra.append(f"init_ring_spacing_nm {args.init_ring_spacing_nm}")
    if args.init_stage_clearance_nm is not None:
        scene_lines_extra.append(f"init_stage_clearance_nm {args.init_stage_clearance_nm}")
    if args.max_iterations is not None:
        scene_lines_extra.append(f"max_iterations {args.max_iterations}")
    elif args.init_mode == "staged_triangle" and args.init_rings is not None:
        stage_iterations = args.init_stage_iterations or 100
        scene_lines_extra.append(f"max_iterations {args.init_rings * stage_iterations}")

    def reject_run(message: str, exit_code: int) -> int:
        print(message, file=sys.stderr)
        if not args.keep_rejected and run_paths.output_dir.exists():
            shutil.rmtree(run_paths.output_dir)
            print(f"Deleted rejected run folder: {run_paths.output_dir}", file=sys.stderr)
        return exit_code

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
        scene_lines_extra=scene_lines_extra,
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
        return reject_run(f"Executable exited with code {process.returncode}", process.returncode)

    if process.stdout:
        print(process.stdout, end="")
    if process.stderr:
        print(process.stderr, end="", file=sys.stderr)

    if not args.disable_bezier:
        if args.nm_per_unit is None:
            return reject_run(
                "Skipping Bezier generation because --nm-per-unit was not provided; "
                "run rejected because outputs are not validated.",
                2,
            )
        else:
            artifacts = generate_beziers_for_run(
                run_paths.output_dir,
                nm_per_unit=args.nm_per_unit,
                validation_tol_nm=(
                    None if args.skip_bezier_validation else args.bezier_validation_tol_nm
                ),
                glb_tessellation_tol_nm=args.bezier_glb_tessellation_tol_nm,
                glb_tessellation_max_depth=args.bezier_glb_tessellation_max_depth,
                required_rmin_nm=args.rmin_nm,
                required_min_separation_nm=args.min_separation_nm,
                tube_radius_nm=args.bezier_tube_radius_nm,
            )
            print(f"Latest curve OBJ: {artifacts.curve_obj}")
            print(f"Bezier curves: {artifacts.fit.output_json}")
            print(f"Bezier samples: {artifacts.fit.sampled_obj}")
            print(f"Bezier USDA: {artifacts.fit.usda_curves}")
            print(f"Bezier validation: {artifacts.fit.validation_json}")
            print(f"Bezier tube GLB preview: {artifacts.fit.glb_mesh}")
            if args.skip_bezier_validation:
                return reject_run(
                    "Bezier validation was skipped; "
                    "run rejected because outputs are not validated.",
                    3,
                )
            if not artifacts.fit.report.passes_all:
                details = []
                if not artifacts.fit.report.passes_rmin:
                    details.append("Bezier minimum radius check failed")
                if not artifacts.fit.report.passes_min_separation:
                    details.append(
                        "Bezier minimum separation check failed "
                        f"(min={artifacts.fit.report.min_separation_nm})"
                    )
                return reject_run("; ".join(details), 5)
    elif not args.keep_rejected:
        return reject_run(
            "Bezier generation was disabled; run rejected because outputs are not validated.",
            4,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
