from __future__ import annotations

import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import Any

from .bezier_postprocess import FitResult, fit_curve_obj_with_beziers

REPO_ROOT = Path(__file__).resolve().parents[2]
CPP_PROJECT_DIR = REPO_ROOT / "surface-filling-curve-flows"
PACKAGE_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_ROOT / "assets"
CAPSULE_ASSET_NAME = "simple_capsule_basic_watertight.obj"
DEFAULT_MODEL_PATH = ASSETS_DIR / CAPSULE_ASSET_NAME


@dataclass(frozen=True)
class CapsuleRunPaths:
    output_dir: Path
    mesh_file: Path
    scene_file: Path


@dataclass(frozen=True)
class BezierRunArtifacts:
    curve_obj: Path
    fit: FitResult


def build_surface_filling_curve_flows(
    build_dir: str = "build",
    jobs: int = 6,
    source_dir: Path | None = None,
    check: bool = True,
) -> CompletedProcess[str]:
    project_dir = source_dir if source_dir is not None else CPP_PROJECT_DIR
    project_dir = Path(project_dir)

    return run(
        [
            "bash",
            "-lc",
            f"cmake -S . -B {build_dir} && cmake --build {build_dir} -j{jobs}",
        ],
        cwd=project_dir,
        text=True,
        capture_output=True,
        check=check,
    )


def run_surface_filling_binary(
    binary_name: str,
    scene_file: str | Path,
    *,
    build_dir: str = "build",
    source_dir: Path | None = None,
    run_dir: Path | None = None,
    extra_args: Sequence[str] = (),
    check: bool = True,
    capture_output: bool = False,
    env: Mapping[str, str] | None = None,
) -> CompletedProcess[str]:
    project_dir = source_dir if source_dir is not None else CPP_PROJECT_DIR
    project_dir = Path(project_dir)

    executable = project_dir / build_dir / "bin" / binary_name
    if not executable.exists():
        raise FileNotFoundError(
            f"Could not find executable: {executable}. "
            f"Build first with build_surface_filling_curve_flows(build_dir='{build_dir}')."
        )

    scene_path = Path(scene_file)
    if not scene_path.is_absolute():
        scene_path = project_dir / scene_path

    execution_dir = run_dir if run_dir is not None else project_dir
    cmd = [str(executable), str(scene_path), *extra_args]
    return run(
        cmd,
        cwd=execution_dir,
        text=True,
        capture_output=capture_output,
        env=dict(env) if env is not None else None,
        check=check,
    )


def make_output_run_dir(base_outputs_dir: str | Path | None = None) -> Path:
    outputs_dir = (
        Path(base_outputs_dir).expanduser().resolve()
        if base_outputs_dir is not None
        else REPO_ROOT / "outputs"
    )
    outputs_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = outputs_dir / f"outputs_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = outputs_dir / f"outputs_{stamp}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def prepare_scene_run(
    *,
    model_path: str | Path | None = None,
    base_outputs_dir: str | Path | None = None,
    radius: float = 0.1,
    h: float = 0.02,
    execute_only: bool = True,
    scene_lines_extra: Sequence[str] = (),
) -> CapsuleRunPaths:
    output_dir = make_output_run_dir(base_outputs_dir)

    source_mesh = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    source_mesh = source_mesh.expanduser().resolve()
    if not source_mesh.exists():
        raise FileNotFoundError(f"Model file not found: {source_mesh}")

    destination_mesh = output_dir / source_mesh.name
    shutil.copy2(source_mesh, destination_mesh)

    scene_lines = [
        f"mesh {destination_mesh.name}",
        f"radius {radius}",
        f"h {h}",
    ]
    scene_lines.extend(scene_lines_extra)
    if execute_only:
        scene_lines.append("excecute_only")

    scene_file = output_dir / "scene.txt"
    scene_file.write_text("\n".join(scene_lines) + "\n", encoding="utf-8")

    return CapsuleRunPaths(output_dir=output_dir, mesh_file=destination_mesh, scene_file=scene_file)


def run_capsule_scene(
    *,
    binary_name: str = "curve_on_surface",
    model_path: str | Path | None = None,
    build_dir: str = "build",
    source_dir: Path | None = None,
    base_outputs_dir: str | Path | None = None,
    radius: float = 0.1,
    h: float = 0.02,
    execute_only: bool = True,
    scene_lines_extra: Sequence[str] = (),
    extra_args: Sequence[str] = (),
    check: bool = True,
    capture_output: bool = False,
    env: Mapping[str, str] | None = None,
) -> tuple[CapsuleRunPaths, CompletedProcess[str]]:
    run_paths = prepare_scene_run(
        model_path=model_path,
        base_outputs_dir=base_outputs_dir,
        radius=radius,
        h=h,
        execute_only=execute_only,
        scene_lines_extra=scene_lines_extra,
    )

    merged_env = dict(os.environ)
    if env is not None:
        merged_env.update(env)
    if execute_only:
        merged_env.setdefault("POLYSCOPE_BACKEND", "openGL_mock")

    process = run_surface_filling_binary(
        binary_name,
        run_paths.scene_file,
        build_dir=build_dir,
        source_dir=source_dir,
        run_dir=run_paths.output_dir,
        extra_args=extra_args,
        check=check,
        capture_output=capture_output,
        env=merged_env,
    )

    return run_paths, process


def prepare_capsule_run(
    *,
    base_outputs_dir: str | Path | None = None,
    radius: float = 0.1,
    h: float = 0.02,
    execute_only: bool = True,
) -> CapsuleRunPaths:
    return prepare_scene_run(
        model_path=DEFAULT_MODEL_PATH,
        base_outputs_dir=base_outputs_dir,
        radius=radius,
        h=h,
        execute_only=execute_only,
    )


def find_latest_curve_obj(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    curve_paths = sorted(
        (run_dir / "objs").glob("curve_*.obj"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    if not curve_paths:
        raise FileNotFoundError(f"No exported curve OBJ found under {run_dir / 'objs'}")
    return curve_paths[-1]


def generate_beziers_for_run(
    run_dir: str | Path,
    *,
    nm_per_unit: float,
    validation_tol_nm: float | None = 0.01,
    glb_tessellation_tol_nm: float = 0.005,
    glb_tessellation_max_depth: int = 9,
    required_rmin_nm: float | None = None,
    required_min_separation_nm: float | None = None,
    tube_radius_nm: float = 0.5,
    output_subdir: str = "bezier_final",
) -> BezierRunArtifacts:
    run_dir = Path(run_dir)
    curve_obj = find_latest_curve_obj(run_dir)
    fit = fit_curve_obj_with_beziers(
        curve_obj,
        nm_per_unit=nm_per_unit,
        validation_tol_nm=validation_tol_nm,
        glb_tessellation_tol_nm=glb_tessellation_tol_nm,
        glb_tessellation_max_depth=glb_tessellation_max_depth,
        required_rmin_nm=required_rmin_nm,
        required_min_separation_nm=required_min_separation_nm,
        tube_radius_nm=tube_radius_nm,
        output_dir=run_dir / output_subdir,
    )
    return BezierRunArtifacts(curve_obj=curve_obj, fit=fit)


def run_curve_on_surface(
    scene_file: str | Path,
    **kwargs: Any,
) -> CompletedProcess[str]:
    return run_surface_filling_binary("curve_on_surface", scene_file, **kwargs)


def run_mesh_deformation(
    scene_file: str | Path,
    **kwargs: Any,
) -> CompletedProcess[str]:
    return run_surface_filling_binary("mesh_deformation", scene_file, **kwargs)
