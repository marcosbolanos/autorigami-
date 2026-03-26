from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CurveData:
    vertices: list[tuple[float, float, float]]
    segments: list[tuple[int, int]]


def parse_obj_vertices_and_lines(path: Path) -> CurveData:
    vertices: list[tuple[float, float, float]] = []
    segments: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif parts[0] == "l" and len(parts) >= 3:
                indices = [int(token.split("/")[0]) - 1 for token in parts[1:]]
                for idx in range(len(indices) - 1):
                    a = indices[idx]
                    b = indices[idx + 1]
                    if a != b:
                        segments.append((a, b))
    return CurveData(vertices=vertices, segments=segments)


def parse_obj_vertices(path: Path) -> list[tuple[float, float, float]]:
    vertices: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return vertices


def vector_sub(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vector_add(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vector_scale(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(dot(a, a))


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return norm(vector_sub(a, b))


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def point_segment_distance(
    point: tuple[float, float, float],
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    ab = vector_sub(b, a)
    ab_len_sq = dot(ab, ab)
    if ab_len_sq <= 1e-18:
        return distance(point, a)
    t = max(0.0, min(1.0, dot(vector_sub(point, a), ab) / ab_len_sq))
    closest = vector_add(a, vector_scale(ab, t))
    return distance(point, closest)


def segment_segment_distance(
    p1: tuple[float, float, float],
    q1: tuple[float, float, float],
    p2: tuple[float, float, float],
    q2: tuple[float, float, float],
) -> float:
    u = vector_sub(q1, p1)
    v = vector_sub(q2, p2)
    w = vector_sub(p1, p2)
    a = dot(u, u)
    b = dot(u, v)
    c = dot(v, v)
    d = dot(u, w)
    e = dot(v, w)
    denom = a * c - b * b
    small = 1e-18

    if a <= small and c <= small:
        return distance(p1, p2)
    if a <= small:
        s = 0.0
        t = max(0.0, min(1.0, e / c if c > small else 0.0))
    else:
        if c <= small:
            t = 0.0
            s = max(0.0, min(1.0, -d / a))
        else:
            if abs(denom) > small:
                s = max(0.0, min(1.0, (b * e - c * d) / denom))
            else:
                s = 0.0
            t = (b * s + e) / c
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -d / a))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - d) / a))

    c1 = vector_add(p1, vector_scale(u, s))
    c2 = vector_add(p2, vector_scale(v, t))
    return distance(c1, c2)


def connected_components(vertex_count: int, segments: list[tuple[int, int]]) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(vertex_count)]
    for a, b in segments:
        adjacency[a].append(b)
        adjacency[b].append(a)
    seen = [False] * vertex_count
    components: list[list[int]] = []
    for start in range(vertex_count):
        if seen[start] or not adjacency[start]:
            continue
        stack = [start]
        seen[start] = True
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(component)
    isolated = [[index] for index, neighbors in enumerate(adjacency) if not neighbors]
    components.extend(isolated)
    return components


def total_length(
    vertices: list[tuple[float, float, float]], segments: list[tuple[int, int]]
) -> float:
    return sum(distance(vertices[a], vertices[b]) for a, b in segments)


def min_inter_component_separation(
    vertices: list[tuple[float, float, float]],
    segments: list[tuple[int, int]],
    components: list[list[int]],
) -> float | None:
    if len(components) < 2:
        return None

    component_by_vertex: dict[int, int] = {}
    for comp_index, component in enumerate(components):
        for vertex in component:
            component_by_vertex[vertex] = comp_index

    component_segments: dict[int, list[tuple[int, int]]] = {
        index: [] for index in range(len(components))
    }
    for a, b in segments:
        comp_index = component_by_vertex.get(a)
        if comp_index is None or component_by_vertex.get(b) != comp_index:
            continue
        component_segments[comp_index].append((a, b))

    best: float | None = None
    for left in range(len(components)):
        left_segments = component_segments[left]
        if not left_segments:
            continue
        for right in range(left + 1, len(components)):
            right_segments = component_segments[right]
            if not right_segments:
                continue
            for a0, a1 in left_segments:
                for b0, b1 in right_segments:
                    candidate = segment_segment_distance(
                        vertices[a0],
                        vertices[a1],
                        vertices[b0],
                        vertices[b1],
                    )
                    if best is None or candidate < best:
                        best = candidate
    return best


def coverage_stats(
    mesh_vertices: list[tuple[float, float, float]],
    curve_vertices: list[tuple[float, float, float]],
    segments: list[tuple[int, int]],
) -> tuple[float, float, float]:
    if not mesh_vertices or not segments:
        return (0.0, 0.0, 0.0)
    mesh = np.asarray(mesh_vertices, dtype=float)
    if mesh.shape[0] > 4096:
        sample_indices = np.linspace(0, mesh.shape[0] - 1, num=4096, dtype=int)
        mesh = mesh[sample_indices]
    seg_start = np.asarray([curve_vertices[a] for a, _ in segments], dtype=float)
    seg_end = np.asarray([curve_vertices[b] for _, b in segments], dtype=float)
    seg_vec = seg_end - seg_start
    seg_len_sq = np.sum(seg_vec * seg_vec, axis=1)
    min_dist_sq = np.full(mesh.shape[0], np.inf, dtype=float)

    # Process segments in chunks to keep memory bounded on the denser runs.
    chunk_size = 256
    for start in range(0, len(segments), chunk_size):
        stop = min(start + chunk_size, len(segments))
        a = seg_start[start:stop]
        v = seg_vec[start:stop]
        vv = seg_len_sq[start:stop]
        diff = mesh[:, None, :] - a[None, :, :]
        proj_num = np.sum(diff * v[None, :, :], axis=2)
        safe_vv = np.where(vv > 1e-18, vv, 1.0)
        t = np.clip(proj_num / safe_vv[None, :], 0.0, 1.0)
        closest = a[None, :, :] + t[:, :, None] * v[None, :, :]
        dist_sq = np.sum((mesh[:, None, :] - closest) ** 2, axis=2)
        min_dist_sq = np.minimum(min_dist_sq, np.min(dist_sq, axis=1))

    distances = np.sqrt(min_dist_sq)
    return (
        float(np.mean(distances)),
        float(np.quantile(distances, 0.95)),
        float(np.max(distances)),
    )


def read_final_data_row(data_csv: Path) -> dict[str, str]:
    with data_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, skipinitialspace=True))
    return rows[-1] if rows else {}


def aggregate_validation(validation_json: Path) -> dict[str, float | int | None]:
    payload = json.loads(validation_json.read_text(encoding="utf-8"))
    entries = payload.get("validation", [])
    report = payload.get("report", {})
    if not entries:
        return {
            "validation_components": 0,
            "bezier_max_distance_nm": None,
            "bezier_mean_distance_nm": None,
            "bezier_max_tangent_error_deg": None,
            "bezier_mean_tangent_error_deg": None,
            "bezier_min_radius_nm": None,
            "bezier_validation_min_separation_nm": report.get("min_separation_nm"),
            "bezier_passes_rmin": report.get("passes_rmin"),
            "bezier_passes_min_separation": report.get("passes_min_separation"),
            "bezier_passes_all": report.get("passes_all"),
        }
    return {
        "validation_components": len(entries),
        "bezier_max_distance_nm": max(entry["max_distance_nm"] for entry in entries),
        "bezier_mean_distance_nm": (
            sum(entry["mean_distance_nm"] for entry in entries) / len(entries)
        ),
        "bezier_max_tangent_error_deg": max(entry["max_tangent_error_deg"] for entry in entries),
        "bezier_mean_tangent_error_deg": sum(entry["mean_tangent_error_deg"] for entry in entries)
        / len(entries),
        "bezier_min_radius_nm": min(
            entry["min_radius_nm"]
            for entry in entries
            if entry.get("min_radius_nm") is not None
        ),
        "bezier_validation_min_separation_nm": report.get("min_separation_nm"),
        "bezier_passes_rmin": report.get("passes_rmin"),
        "bezier_passes_min_separation": report.get("passes_min_separation"),
        "bezier_passes_all": report.get("passes_all"),
    }


def collect_case_metrics(case_dir: Path, nm_per_unit: float, init_rings: int) -> dict[str, object]:
    run_dirs = sorted(case_dir.glob("outputs_*"))
    base_row: dict[str, object] = {
        "nm_per_unit": nm_per_unit,
        "init_rings": init_rings,
        "case_dir": str(case_dir),
        "status": "rejected",
        "run_dir": "",
        "curve_obj": "",
        "node_count": "",
        "segment_count": "",
        "component_count": "",
        "final_total_length_units": "",
        "final_total_length_nm": "",
        "min_inter_ring_separation_units": "",
        "min_inter_ring_separation_nm": "",
        "coverage_mean_distance_units": "",
        "coverage_p95_distance_units": "",
        "coverage_max_distance_units": "",
        "coverage_mean_distance_nm": "",
        "coverage_p95_distance_nm": "",
        "coverage_max_distance_nm": "",
        "final_energy": "",
        "final_descent_l2": "",
        "final_descent_l1": "",
        "final_descent_linf": "",
        "final_max_curvature": "",
        "bezier_validation_path": "",
        "validation_components": "",
        "bezier_max_distance_nm": "",
        "bezier_mean_distance_nm": "",
        "bezier_max_tangent_error_deg": "",
        "bezier_mean_tangent_error_deg": "",
        "bezier_min_radius_nm": "",
        "bezier_validation_min_separation_nm": "",
        "bezier_passes_rmin": "",
        "bezier_passes_min_separation": "",
        "bezier_passes_all": "",
        "glb_path": "",
    }

    if not run_dirs:
        return base_row

    run_dir = run_dirs[-1]
    curve_obj = run_dir / "objs" / "curve_100.obj"
    validation_json = run_dir / "bezier_final" / "curve_100_bezier_validation.json"
    if not curve_obj.exists():
        base_row["status"] = "missing_curve"
        base_row["run_dir"] = str(run_dir)
        return base_row

    curve = parse_obj_vertices_and_lines(curve_obj)
    components = connected_components(len(curve.vertices), curve.segments)
    total_length_units = total_length(curve.vertices, curve.segments)
    min_sep_units = min_inter_component_separation(curve.vertices, curve.segments, components)
    mesh_vertices = parse_obj_vertices(run_dir / "object.obj")
    coverage_mean_units, coverage_p95_units, coverage_max_units = coverage_stats(
        mesh_vertices,
        curve.vertices,
        curve.segments,
    )
    final_row = read_final_data_row(run_dir / "data.csv")

    status = "missing_validation"
    validation_fields: dict[str, object] = {}
    if validation_json.exists():
        validation_fields = aggregate_validation(validation_json)
        passes_all = validation_fields.get("bezier_passes_all")
        status = "accepted" if passes_all is True else "rejected"

    base_row.update(
        {
            "status": status,
            "run_dir": str(run_dir),
            "curve_obj": str(curve_obj),
            "node_count": len(curve.vertices),
            "segment_count": len(curve.segments),
            "component_count": len(components),
            "final_total_length_units": total_length_units,
            "final_total_length_nm": total_length_units * nm_per_unit,
            "min_inter_ring_separation_units": min_sep_units,
            "min_inter_ring_separation_nm": (
                None if min_sep_units is None else min_sep_units * nm_per_unit
            ),
            "coverage_mean_distance_units": coverage_mean_units,
            "coverage_p95_distance_units": coverage_p95_units,
            "coverage_max_distance_units": coverage_max_units,
            "coverage_mean_distance_nm": coverage_mean_units * nm_per_unit,
            "coverage_p95_distance_nm": coverage_p95_units * nm_per_unit,
            "coverage_max_distance_nm": coverage_max_units * nm_per_unit,
            "final_energy": final_row.get("f", ""),
            "final_descent_l2": final_row.get("descent norm (L2)", ""),
            "final_descent_l1": final_row.get("descent norm (L1)", ""),
            "final_descent_linf": final_row.get("descent norm (L∞)", ""),
            "final_max_curvature": final_row.get("max curvature", ""),
            "bezier_validation_path": str(validation_json) if validation_json.exists() else "",
            "glb_path": (
                str(run_dir / "bezier_final" / "curve_100_bezier_tube.glb")
                if (run_dir / "bezier_final" / "curve_100_bezier_tube.glb").exists()
                else ""
            ),
        }
    )

    if validation_fields:
        base_row.update(validation_fields)
    return base_row


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect metrics for the capsule ring sweep")
    parser.add_argument("sweep_root", type=Path, help="Sweep root directory")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path; defaults to <sweep_root>/capsule_ring_sweep_metrics.csv",
    )
    args = parser.parse_args()

    sweep_root = args.sweep_root.resolve()
    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else sweep_root / "capsule_ring_sweep_metrics.csv"
    )

    rows: list[dict[str, object]] = []
    for nm_per_unit in (25.0, 100.0):
        nm_dir = sweep_root / f"capsule_nm_per_unit_{int(nm_per_unit)}"
        for init_rings in range(2, 9):
            case_dir = nm_dir / f"init_rings_{init_rings}_max_iterations_100_rmin_nm_6p0"
            rows.append(collect_case_metrics(case_dir, nm_per_unit, init_rings))

    fieldnames = list(rows[0].keys()) if rows else []
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
