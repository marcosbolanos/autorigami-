from __future__ import annotations

import argparse

from .bezier_postprocess import fit_curve_obj_with_beziers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit cubic Bezier curves to an optimizer polyline and validate densely"
    )
    parser.add_argument("curve", help="Path to the input OBJ curve file")
    parser.add_argument(
        "--nm-per-unit",
        type=float,
        default=25.0,
        help="Nanometers represented by one mesh unit",
    )
    parser.add_argument(
        "--handle-scale",
        type=float,
        default=1.0 / 3.0,
        help="Bezier handle length as a fraction of the local chord length",
    )
    parser.add_argument(
        "--smooth-tangent-iters",
        type=int,
        default=2,
        help="Number of local tangent smoothing iterations before fitting",
    )
    parser.add_argument(
        "--validation-tol-nm",
        type=float,
        default=0.01,
        help="Adaptive subdivision tolerance in nanometers",
    )
    parser.add_argument(
        "--required-rmin-nm",
        type=float,
        default=None,
        help="Reject the fit if any component falls below this minimum radius",
    )
    parser.add_argument(
        "--required-min-separation-nm",
        type=float,
        default=None,
        help="Reject the fit if the minimum separation between components falls below this value",
    )
    parser.add_argument(
        "--tube-radius-nm",
        type=float,
        default=0.5,
        help="Physical radius for the rendered GLB tube geometry",
    )
    parser.add_argument(
        "--glb-tessellation-tol-nm",
        type=float,
        default=0.005,
        help="Adaptive subdivision tolerance in nanometers for the GLB preview mesh",
    )
    parser.add_argument(
        "--glb-tessellation-max-depth",
        type=int,
        default=9,
        help="Maximum adaptive subdivision depth for the GLB preview mesh",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip expensive validation metrics and only emit fitted Bezier outputs",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the Bezier JSON, validation JSON, and sampled OBJ outputs",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = fit_curve_obj_with_beziers(
        args.curve,
        nm_per_unit=args.nm_per_unit,
        handle_scale=args.handle_scale,
        smooth_tangent_iters=args.smooth_tangent_iters,
        validation_tol_nm=None if args.skip_validation else args.validation_tol_nm,
        glb_tessellation_tol_nm=args.glb_tessellation_tol_nm,
        glb_tessellation_max_depth=args.glb_tessellation_max_depth,
        required_rmin_nm=args.required_rmin_nm,
        required_min_separation_nm=args.required_min_separation_nm,
        tube_radius_nm=args.tube_radius_nm,
        output_dir=args.output_dir,
    )

    print(f"Bezier curves: {result.output_json}")
    print(f"Validation report: {result.validation_json}")
    print(f"Sampled curve OBJ: {result.sampled_obj}")
    print(f"Bezier USDA: {result.usda_curves}")
    print(f"Tube GLB preview: {result.glb_mesh}")
    for summary in result.validation:
        radius_text = "inf" if summary.min_radius_nm is None else f"{summary.min_radius_nm:.6f}"
        print(
            "component "
            f"{summary.component}: "
            f"segments={summary.segments}, "
            f"samples={summary.sample_count}, "
            f"max_dist_nm={summary.max_distance_nm:.6f}, "
            f"max_tangent_deg={summary.max_tangent_error_deg:.6f}, "
            f"min_radius_nm={radius_text}"
        )
    if not result.validation:
        print("validation skipped")
    else:
        min_sep = result.report.min_separation_nm
        min_sep_text = "inf" if min_sep is None else f"{min_sep:.6f}"
        print(
            "overall "
            f"passes_all={result.report.passes_all}, "
            f"passes_rmin={result.report.passes_rmin}, "
            f"passes_min_separation={result.report.passes_min_separation}, "
            f"min_separation_nm={min_sep_text}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
