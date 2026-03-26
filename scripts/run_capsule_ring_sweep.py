from __future__ import annotations

import argparse
import csv
import subprocess
from datetime import datetime
from pathlib import Path

from collect_capsule_ring_sweep_metrics import collect_case_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def _dated_sweep_root(base_dir: Path, label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = base_dir / f"{label}_{stamp}"
    suffix = 1
    while root.exists():
        root = base_dir / f"{label}_{stamp}_{suffix}"
        suffix += 1
    root.mkdir(parents=True, exist_ok=False)
    return root


def _write_summary_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    if not rows:
        return
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the strict capsule staged-triangle sweep"
    )
    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Parent directory where the dated sweep folder will be created",
    )
    parser.add_argument(
        "--label",
        default="capsule_staged_triangle_sweep_strict",
        help="Prefix for the dated sweep folder",
    )
    parser.add_argument(
        "--nm-per-unit",
        type=float,
        nargs="+",
        default=[25.0, 100.0],
        help="Physical scales to sweep",
    )
    parser.add_argument(
        "--triangle-min", type=int, default=1, help="Minimum staged triangle count"
    )
    parser.add_argument(
        "--triangle-max", type=int, default=6, help="Maximum staged triangle count"
    )
    parser.add_argument("--rmin-nm", type=float, default=6.0, help="Minimum radius in nm")
    parser.add_argument(
        "--min-separation-nm",
        type=float,
        default=2.6,
        help="Minimum allowed separation between distinct curve components in nm",
    )
    parser.add_argument("--h", type=float, default=0.1, help="Optimizer h value")
    parser.add_argument(
        "--init-stage-iterations",
        type=int,
        default=100,
        help="Iterations between staged triangle insertions",
    )
    parser.add_argument(
        "--init-stage-clearance-nm",
        type=float,
        default=None,
        help="Optional physical clearance target when placing the next staged triangle",
    )
    parser.add_argument(
        "--bezier-validation-tol-nm",
        type=float,
        default=0.01,
        help="Bezier validation tolerance in nm",
    )
    parser.add_argument(
        "--bezier-tube-radius-nm",
        type=float,
        default=0.5,
        help="Tube radius for rendered GLB geometry",
    )
    parser.add_argument(
        "--build-dir",
        default="build",
        help="CMake build directory for the C++ binary",
    )
    parser.add_argument(
        "--binary",
        default="curve_on_surface",
        choices=["curve_on_surface", "mesh_deformation"],
        help="C++ binary to run",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue the sweep even if a subprocess call raises unexpectedly",
    )
    args = parser.parse_args()

    base_output_dir = args.base_output_dir.expanduser().resolve()
    sweep_root = _dated_sweep_root(base_output_dir, args.label)
    summary_csv = sweep_root / "sweep_summary.csv"
    metrics_csv = sweep_root / "capsule_staged_triangle_sweep_metrics.csv"

    summary_rows: list[dict[str, object]] = []
    print(f"sweep_root={sweep_root}")

    for nm_per_unit in args.nm_per_unit:
        nm_dir = sweep_root / f"capsule_nm_per_unit_{int(nm_per_unit)}"
        nm_dir.mkdir(parents=True, exist_ok=True)
        for triangle_count in range(args.triangle_min, args.triangle_max + 1):
            case_dir = nm_dir / (
                f"staged_triangles_{triangle_count}"
                f"_stage_iterations_{args.init_stage_iterations}"
                f"_rmin_nm_{str(args.rmin_nm).replace('.', 'p')}"
            )
            case_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"=== nm_per_unit={nm_per_unit} staged_triangles={triangle_count} ===",
                flush=True,
            )
            cmd = [
                str(PYTHON),
                "-m",
                "autorigami_cpp.cli",
                "--build-dir",
                args.build_dir,
                "--binary",
                args.binary,
                "--nm-per-unit",
                str(nm_per_unit),
                "--rmin-nm",
                str(args.rmin_nm),
                "--min-separation-nm",
                str(args.min_separation_nm),
                "--h",
                str(args.h),
                "--init-mode",
                "staged_triangle",
                "--init-rings",
                str(triangle_count),
                "--init-stage-iterations",
                str(args.init_stage_iterations),
                "--keep-previous-active",
                "--bezier-validation-tol-nm",
                str(args.bezier_validation_tol_nm),
                "--bezier-tube-radius-nm",
                str(args.bezier_tube_radius_nm),
                "--outputs-dir",
                str(case_dir),
            ]
            if args.init_stage_clearance_nm is not None:
                cmd.extend(
                    ["--init-stage-clearance-nm", str(args.init_stage_clearance_nm)]
                )
            try:
                result = subprocess.run(
                    cmd,
                    cwd=REPO_ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
            except Exception as exc:  # noqa: BLE001
                if not args.keep_going:
                    raise
                summary_rows.append(
                    {
                        "nm_per_unit": nm_per_unit,
                        "triangle_count": triangle_count,
                        "case_dir": str(case_dir),
                        "returncode": "",
                        "status": f"runner_exception: {exc}",
                    }
                )
                _write_summary_csv(summary_rows, summary_csv)
                continue

            (case_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
            (case_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
            (case_dir / "last_status.txt").write_text(
                f"status={result.returncode}\n", encoding="utf-8"
            )

            metrics_row = collect_case_metrics(case_dir, nm_per_unit, triangle_count)
            summary_row = {
                "nm_per_unit": nm_per_unit,
                "triangle_count": triangle_count,
                "case_dir": str(case_dir),
                "returncode": result.returncode,
                "status": metrics_row["status"],
                "run_dir": metrics_row["run_dir"],
                "bezier_passes_all": metrics_row["bezier_passes_all"],
                "bezier_passes_rmin": metrics_row["bezier_passes_rmin"],
                "bezier_passes_min_separation": metrics_row["bezier_passes_min_separation"],
                "final_total_length_nm": metrics_row["final_total_length_nm"],
                "min_inter_ring_separation_nm": metrics_row["min_inter_ring_separation_nm"],
                "bezier_validation_min_separation_nm": metrics_row[
                    "bezier_validation_min_separation_nm"
                ],
            }
            summary_rows.append(summary_row)
            _write_summary_csv(summary_rows, summary_csv)

    metrics_rows: list[dict[str, object]] = []
    for nm_per_unit in args.nm_per_unit:
        for triangle_count in range(args.triangle_min, args.triangle_max + 1):
            case_dir = sweep_root / f"capsule_nm_per_unit_{int(nm_per_unit)}" / (
                f"staged_triangles_{triangle_count}"
                f"_stage_iterations_{args.init_stage_iterations}"
                f"_rmin_nm_{str(args.rmin_nm).replace('.', 'p')}"
            )
            metrics_rows.append(collect_case_metrics(case_dir, nm_per_unit, triangle_count))

    _write_summary_csv(metrics_rows, metrics_csv)
    print(f"summary_csv={summary_csv}")
    print(f"metrics_csv={metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
