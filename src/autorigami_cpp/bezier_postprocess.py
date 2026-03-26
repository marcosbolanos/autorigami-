# pyright: reportMissingImports=false

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

type Vec3 = Any
type Ctrl = tuple[Any, Any, Any, Any]


@dataclass(frozen=True)
class CubicBezierCurve:
    p0: tuple[float, float, float]
    p1: tuple[float, float, float]
    p2: tuple[float, float, float]
    p3: tuple[float, float, float]
    component: int
    segment: int


@dataclass(frozen=True)
class ValidationSummary:
    component: int
    segments: int
    sample_count: int
    max_distance_nm: float
    mean_distance_nm: float
    max_tangent_error_deg: float
    mean_tangent_error_deg: float
    max_curvature_per_unit: float
    min_radius_nm: float | None


@dataclass(frozen=True)
class ValidationReport:
    components: list[ValidationSummary]
    min_separation_nm: float | None
    required_rmin_nm: float | None
    required_min_separation_nm: float | None
    passes_rmin: bool
    passes_min_separation: bool
    passes_all: bool


@dataclass(frozen=True)
class FitResult:
    curves: list[CubicBezierCurve]
    validation: list[ValidationSummary]
    report: ValidationReport
    output_json: Path
    validation_json: Path
    sampled_obj: Path
    usda_curves: Path
    glb_mesh: Path


@dataclass(frozen=True)
class _ComponentFitInput:
    points: np.ndarray
    tangents: np.ndarray
    closed: bool
    component_id: int


def _as_array(values: tuple[float, float, float] | Vec3) -> Vec3:
    return np.asarray(values, dtype=float)


def _vec_tuple(vec: Vec3) -> tuple[float, float, float]:
    return (float(vec[0]), float(vec[1]), float(vec[2]))


def _norm(vec: Vec3) -> float:
    return float(np.linalg.norm(vec))


def _unit(vec: Vec3) -> Vec3:
    norm = _norm(vec)
    if norm <= 1e-12:
        return np.zeros(3, dtype=float)
    return vec / norm


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return np.cross(a, b)


def _point_line_distance(point: Vec3, start: Vec3, end: Vec3) -> float:
    delta = end - start
    denom = float(np.dot(delta, delta))
    if denom <= 1e-16:
        return _norm(point - start)
    t = _clamp(float(np.dot(point - start, delta) / denom), 0.0, 1.0)
    projection = start + t * delta
    return _norm(point - projection)


def _bezier_eval(ctrl: Ctrl, t: float) -> Vec3:
    omt = 1.0 - t
    p0, p1, p2, p3 = ctrl
    return (
        (omt**3) * p0
        + 3.0 * (omt**2) * t * p1
        + 3.0 * omt * (t**2) * p2
        + (t**3) * p3
    )


def _bezier_first_derivative(ctrl: Ctrl, t: float) -> Vec3:
    omt = 1.0 - t
    p0, p1, p2, p3 = ctrl
    return (
        3.0 * (omt**2) * (p1 - p0)
        + 6.0 * omt * t * (p2 - p1)
        + 3.0 * (t**2) * (p3 - p2)
    )


def _bezier_second_derivative(ctrl: Ctrl, t: float) -> Vec3:
    p0, p1, p2, p3 = ctrl
    omt = 1.0 - t
    return 6.0 * omt * (p2 - 2.0 * p1 + p0) + 6.0 * t * (p3 - 2.0 * p2 + p1)


def _adaptive_sample(
    ctrl: Ctrl,
    tol_units: float,
    depth: int = 0,
    max_depth: int = 12,
) -> list[Vec3]:
    p0, p1, p2, p3 = ctrl
    flatness = max(_point_line_distance(p1, p0, p3), _point_line_distance(p2, p0, p3))
    if flatness <= tol_units or depth >= max_depth:
        return [p0, p3]

    p01 = 0.5 * (p0 + p1)
    p12 = 0.5 * (p1 + p2)
    p23 = 0.5 * (p2 + p3)
    p012 = 0.5 * (p01 + p12)
    p123 = 0.5 * (p12 + p23)
    pmid = 0.5 * (p012 + p123)

    left = (p0, p01, p012, pmid)
    right = (pmid, p123, p23, p3)
    left_points = _adaptive_sample(left, tol_units, depth + 1, max_depth)
    right_points = _adaptive_sample(right, tol_units, depth + 1, max_depth)
    return left_points[:-1] + right_points


def _ordered_component(edges: list[tuple[int, int]]) -> tuple[list[int], bool]:
    neighbors: dict[int, list[int]] = {}
    for a, b in edges:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)

    endpoints = [node for node, adjacent in neighbors.items() if len(adjacent) == 1]
    start = endpoints[0] if endpoints else min(neighbors)
    ordered = [start]
    closed = not endpoints
    prev = -1
    current = start

    while True:
        next_candidates = [node for node in neighbors[current] if node != prev]
        if not next_candidates:
            break
        nxt = next_candidates[0]
        if nxt == start:
            break
        ordered.append(nxt)
        prev, current = current, nxt

    return ordered, closed


def _connected_components(
    num_nodes: int, edges: list[tuple[int, int]]
) -> list[list[tuple[int, int]]]:
    parent = list(range(num_nodes))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    by_root: dict[int, list[tuple[int, int]]] = {}
    for edge in edges:
        by_root.setdefault(find(edge[0]), []).append(edge)
    return list(by_root.values())


def _segment_segment_distance(
    p1: Vec3,
    q1: Vec3,
    p2: Vec3,
    q2: Vec3,
) -> float:
    u = q1 - p1
    v = q2 - p2
    w = p1 - p2
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denom = a * c - b * b
    small = 1e-18

    if a <= small and c <= small:
        return _norm(p1 - p2)
    if a <= small:
        s = 0.0
        t = _clamp(e / c if c > small else 0.0, 0.0, 1.0)
    elif c <= small:
        t = 0.0
        s = _clamp(-d / a, 0.0, 1.0)
    else:
        if abs(denom) > small:
            s = _clamp((b * e - c * d) / denom, 0.0, 1.0)
        else:
            s = 0.0
        t = (b * s + e) / c
        if t < 0.0:
            t = 0.0
            s = _clamp(-d / a, 0.0, 1.0)
        elif t > 1.0:
            t = 1.0
            s = _clamp((b - d) / a, 0.0, 1.0)

    c1 = p1 + s * u
    c2 = p2 + t * v
    return _norm(c1 - c2)


def read_curve_obj(curve_path: str | Path) -> tuple[np.ndarray, list[tuple[int, int]]]:
    curve_path = Path(curve_path)
    vertices: list[tuple[float, float, float]] = []
    edges: list[tuple[int, int]] = []

    for raw_line in curve_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("v "):
            _, x, y, z = line.split()
            vertices.append((float(x), float(y), float(z)))
        elif line.startswith("l "):
            parts = [int(token) - 1 for token in line.split()[1:]]
            for a, b in zip(parts[:-1], parts[1:], strict=True):
                edges.append((a, b))

    if not vertices or not edges:
        raise ValueError(f"Curve file {curve_path} does not contain vertices and polyline edges")

    return np.asarray(vertices, dtype=float), edges


def _component_points(
    vertices: np.ndarray, edges: list[tuple[int, int]]
) -> tuple[np.ndarray, bool]:
    ordered, closed = _ordered_component(edges)
    return vertices[np.asarray(ordered, dtype=int)], closed


def _arclengths(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=float)
    lengths = np.zeros(len(points), dtype=float)
    if len(points) > 1:
        lengths[1:] = np.cumsum(np.linalg.norm(points[1:] - points[:-1], axis=1))
    return lengths


def _estimate_tangents(points: np.ndarray, smooth_iters: int, closed: bool) -> np.ndarray:
    tangents = np.zeros_like(points)
    count = len(points)
    if count < 2:
        return tangents

    for idx in range(count):
        if closed:
            tangents[idx] = _unit(points[(idx + 1) % count] - points[(idx - 1) % count])
        elif idx == 0:
            tangents[idx] = _unit(points[1] - points[0])
        elif idx == count - 1:
            tangents[idx] = _unit(points[-1] - points[-2])
        else:
            tangents[idx] = _unit(points[idx + 1] - points[idx - 1])

    for _ in range(max(smooth_iters, 0)):
        updated = tangents.copy()
        if closed:
            for idx in range(count):
                updated[idx] = _unit(
                    tangents[(idx - 1) % count]
                    + 2.0 * tangents[idx]
                    + tangents[(idx + 1) % count]
                )
        else:
            for idx in range(1, count - 1):
                updated[idx] = _unit(tangents[idx - 1] + 2.0 * tangents[idx] + tangents[idx + 1])
        tangents = updated

    return tangents


def _fit_cubic_chain(
    points: np.ndarray,
    tangents: np.ndarray,
    handle_scale: float,
    component_id: int,
    closed: bool,
) -> list[CubicBezierCurve]:
    curves: list[CubicBezierCurve] = []
    segment_count = len(points) if closed else len(points) - 1
    for segment_id in range(segment_count):
        p0 = points[segment_id]
        p3 = points[(segment_id + 1) % len(points)]
        chord = _norm(p3 - p0)
        if chord <= 1e-12:
            continue
        chord_dir = _unit(p3 - p0)

        if handle_scale <= 1e-12:
            p1 = p0 + (chord / 3.0) * chord_dir
            p2 = p0 + (2.0 * chord / 3.0) * chord_dir
            curves.append(
                CubicBezierCurve(
                    p0=_vec_tuple(p0),
                    p1=_vec_tuple(p1),
                    p2=_vec_tuple(p2),
                    p3=_vec_tuple(p3),
                    component=component_id,
                    segment=segment_id,
                )
            )
            continue

        if closed:
            prev_len = _norm(points[segment_id] - points[(segment_id - 1) % len(points)])
            next_len = _norm(
                points[(segment_id + 2) % len(points)] - points[(segment_id + 1) % len(points)]
            )
        else:
            prev_len = (
                chord if segment_id == 0 else _norm(points[segment_id] - points[segment_id - 1])
            )
            next_len = (
                chord
                if segment_id + 2 >= len(points)
                else _norm(points[segment_id + 2] - points[segment_id + 1])
            )
        left_handle = handle_scale * min(prev_len, chord)
        right_handle = handle_scale * min(next_len, chord)

        p1 = p0 + left_handle * tangents[segment_id]
        p2 = p3 - right_handle * tangents[(segment_id + 1) % len(points)]
        curves.append(
            CubicBezierCurve(
                p0=_vec_tuple(p0),
                p1=_vec_tuple(p1),
                p2=_vec_tuple(p2),
                p3=_vec_tuple(p3),
                component=component_id,
                segment=segment_id,
            )
        )
    return curves


def _nearest_polyline_tangent(point: Vec3, polyline: np.ndarray) -> tuple[float, Vec3]:
    best_distance = math.inf
    best_tangent = np.zeros(3, dtype=float)
    for start, end in zip(polyline[:-1], polyline[1:], strict=True):
        delta = end - start
        denom = float(np.dot(delta, delta))
        if denom <= 1e-16:
            continue
        t = _clamp(float(np.dot(point - start, delta) / denom), 0.0, 1.0)
        projection = start + t * delta
        distance = _norm(point - projection)
        if distance < best_distance:
            best_distance = distance
            best_tangent = _unit(delta)
    return best_distance, best_tangent


def _nearest_polyline_tangent_closed(
    point: Vec3, polyline: np.ndarray, closed: bool
) -> tuple[float, Vec3]:
    if not closed:
        return _nearest_polyline_tangent(point, polyline)

    best_distance = math.inf
    best_tangent = np.zeros(3, dtype=float)
    count = len(polyline)
    for idx in range(count):
        start = polyline[idx]
        end = polyline[(idx + 1) % count]
        delta = end - start
        denom = float(np.dot(delta, delta))
        if denom <= 1e-16:
            continue
        t = _clamp(float(np.dot(point - start, delta) / denom), 0.0, 1.0)
        projection = start + t * delta
        distance = _norm(point - projection)
        if distance < best_distance:
            best_distance = distance
            best_tangent = _unit(delta)
    return best_distance, best_tangent


def _curvature_from_derivatives(first: Vec3, second: Vec3) -> float:
    denom = _norm(first) ** 3
    if denom <= 1e-16:
        return 0.0
    return _norm(np.cross(first, second)) / denom


def _validate_component(
    curves: list[CubicBezierCurve],
    polyline: np.ndarray,
    closed: bool,
    nm_per_unit: float,
    validation_tol_nm: float,
) -> tuple[ValidationSummary, list[Vec3]]:
    tol_units = validation_tol_nm / nm_per_unit
    sampled_points: list[Vec3] = []
    tangent_errors_deg: list[float] = []
    point_distances_nm: list[float] = []
    curvatures: list[float] = []

    for curve in curves:
        ctrl = tuple(_as_array(getattr(curve, name)) for name in ("p0", "p1", "p2", "p3"))
        local_points = _adaptive_sample(ctrl, tol_units)
        if sampled_points:
            local_points = local_points[1:]
        sampled_points.extend(local_points)

        sample_count = min(max(8, len(local_points) * 2), 96)
        for t in np.linspace(0.0, 1.0, sample_count):
            point = _bezier_eval(ctrl, float(t))
            first = _bezier_first_derivative(ctrl, float(t))
            second = _bezier_second_derivative(ctrl, float(t))
            distance_units, ref_tangent = _nearest_polyline_tangent_closed(point, polyline, closed)
            bezier_tangent = _unit(first)

            dot_value = _clamp(float(np.dot(bezier_tangent, ref_tangent)), -1.0, 1.0)
            tangent_errors_deg.append(math.degrees(math.acos(dot_value)))
            point_distances_nm.append(distance_units * nm_per_unit)
            curvatures.append(_curvature_from_derivatives(first, second))

    max_curvature = max(curvatures, default=0.0)
    min_radius_nm = None if max_curvature <= 1e-12 else nm_per_unit / max_curvature
    summary = ValidationSummary(
        component=curves[0].component if curves else 0,
        segments=len(curves),
        sample_count=len(sampled_points),
        max_distance_nm=max(point_distances_nm, default=0.0),
        mean_distance_nm=float(np.mean(point_distances_nm)) if point_distances_nm else 0.0,
        max_tangent_error_deg=max(tangent_errors_deg, default=0.0),
        mean_tangent_error_deg=float(np.mean(tangent_errors_deg)) if tangent_errors_deg else 0.0,
        max_curvature_per_unit=max_curvature,
        min_radius_nm=min_radius_nm,
    )
    return summary, sampled_points


def _default_output_paths(
    curve_path: Path, output_dir: Path | None
) -> tuple[Path, Path, Path, Path]:
    base_dir = output_dir if output_dir is not None else curve_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = curve_path.stem
    return (
        base_dir / f"{stem}_bezier.json",
        base_dir / f"{stem}_bezier_validation.json",
        base_dir / f"{stem}_bezier_samples.obj",
        base_dir / f"{stem}_bezier_curves.usda",
    )


def _default_glb_path(curve_path: Path, output_dir: Path | None) -> Path:
    base_dir = output_dir if output_dir is not None else curve_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{curve_path.stem}_bezier_tube.glb"


def _sample_component_curves(curves: list[CubicBezierCurve], tol_units: float) -> list[Vec3]:
    return _sample_component_curves_with_max_depth(curves, tol_units, 12)


def _sample_component_curves_with_max_depth(
    curves: list[CubicBezierCurve], tol_units: float, max_depth: int
) -> list[Vec3]:
    sampled_points: list[Vec3] = []
    for curve in curves:
        ctrl = tuple(_as_array(getattr(curve, name)) for name in ("p0", "p1", "p2", "p3"))
        local_points = _adaptive_sample(ctrl, tol_units, max_depth=max_depth)
        if sampled_points:
            local_points = local_points[1:]
        sampled_points.extend(local_points)
    return sampled_points


def _component_polyline_segments(points: list[Vec3], closed: bool) -> list[tuple[Vec3, Vec3]]:
    if len(points) < 2:
        return []
    segments = [(points[idx], points[idx + 1]) for idx in range(len(points) - 1)]
    if closed:
        segments.append((points[-1], points[0]))
    return segments


def _min_inter_component_separation_nm(
    sampled_components: list[tuple[list[Vec3], bool]],
    nm_per_unit: float,
) -> float | None:
    if len(sampled_components) < 2:
        return None
    component_segments = [
        _component_polyline_segments(points, closed) for points, closed in sampled_components
    ]
    best: float | None = None
    for left in range(len(component_segments)):
        if not component_segments[left]:
            continue
        for right in range(left + 1, len(component_segments)):
            if not component_segments[right]:
                continue
            for a0, a1 in component_segments[left]:
                for b0, b1 in component_segments[right]:
                    candidate = _segment_segment_distance(a0, a1, b0, b1) * nm_per_unit
                    if best is None or candidate < best:
                        best = candidate
    return best


def _tube_mesh_for_component(
    points: list[Vec3],
    closed: bool,
    radius_units: float,
    radial_segments: int,
) -> trimesh.Trimesh | None:
    if len(points) < 2 or radius_units <= 0.0:
        return None
    pts = [np.asarray(point, dtype=float) for point in points]
    tangents: list[np.ndarray] = []
    count = len(pts)
    for idx in range(count):
        if closed:
            tangent = _unit(pts[(idx + 1) % count] - pts[idx - 1])
        elif idx == 0:
            tangent = _unit(pts[1] - pts[0])
        elif idx == count - 1:
            tangent = _unit(pts[-1] - pts[-2])
        else:
            tangent = _unit(pts[idx + 1] - pts[idx - 1])
        tangents.append(tangent)

    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(reference, tangents[0]))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0], dtype=float)
    normal = _unit(_cross(tangents[0], reference))
    if _norm(normal) <= 1e-12:
        normal = np.array([1.0, 0.0, 0.0], dtype=float)
    binormal = _unit(_cross(tangents[0], normal))

    frames: list[tuple[np.ndarray, np.ndarray]] = [(normal, binormal)]
    for idx in range(1, count):
        tangent = tangents[idx]
        prev_normal, _ = frames[-1]
        normal = prev_normal - tangent * float(np.dot(prev_normal, tangent))
        if _norm(normal) <= 1e-12:
            fallback = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(float(np.dot(fallback, tangent))) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0], dtype=float)
            normal = _unit(_cross(tangent, fallback))
        else:
            normal = _unit(normal)
        binormal = _unit(_cross(tangent, normal))
        frames.append((normal, binormal))

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    angles = np.linspace(0.0, 2.0 * math.pi, num=radial_segments, endpoint=False)
    for center, (normal, binormal) in zip(pts, frames, strict=True):
        for angle in angles:
            offset = radius_units * (
                math.cos(float(angle)) * normal + math.sin(float(angle)) * binormal
            )
            vertices.append((center + offset).tolist())

    ring_count = count if closed else count - 1
    for ring in range(ring_count):
        next_ring = (ring + 1) % count
        for side in range(radial_segments):
            next_side = (side + 1) % radial_segments
            a = ring * radial_segments + side
            b = ring * radial_segments + next_side
            c = next_ring * radial_segments + side
            d = next_ring * radial_segments + next_side
            faces.append([a, c, b])
            faces.append([b, c, d])

    if not closed:
        start_center_idx = len(vertices)
        vertices.append(pts[0].tolist())
        end_center_idx = len(vertices)
        vertices.append(pts[-1].tolist())
        for side in range(radial_segments):
            next_side = (side + 1) % radial_segments
            faces.append([start_center_idx, next_side, side])
            a = (count - 1) * radial_segments + side
            b = (count - 1) * radial_segments + next_side
            faces.append([end_center_idx, a, b])

    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)


def _write_glb_tube(
    path: Path,
    sampled_components: list[tuple[list[Vec3], bool]],
    nm_per_unit: float,
    tube_radius_nm: float,
    radial_segments: int,
) -> None:
    radius_units = tube_radius_nm / nm_per_unit
    meshes = []
    for points, closed in sampled_components:
        mesh = _tube_mesh_for_component(points, closed, radius_units, radial_segments)
        if mesh is not None:
            meshes.append(mesh)
    if not meshes:
        trimesh.Scene().export(path)
        return
    scene = trimesh.Scene()
    for index, mesh in enumerate(meshes):
        scene.add_geometry(mesh, node_name=f"component_{index}")
    scene.export(path)


def _write_usda_curves(
    path: Path,
    curves: list[CubicBezierCurve],
    nm_per_unit: float,
    tube_radius_nm: float,
) -> None:
    width_units = 2.0 * tube_radius_nm / nm_per_unit
    point_rows = []
    curve_counts = []
    widths = []
    for curve in curves:
        point_rows.extend(
            [
                curve.p0,
                curve.p1,
                curve.p2,
                curve.p3,
            ]
        )
        curve_counts.append(4)
        widths.append(width_units)

    def fmt_vec3(point: tuple[float, float, float]) -> str:
        return f"({point[0]:.9g}, {point[1]:.9g}, {point[2]:.9g})"

    points_text = ",\n            ".join(fmt_vec3(point) for point in point_rows)
    counts_text = ", ".join(str(count) for count in curve_counts)
    widths_text = ", ".join(f"{width:.9g}" for width in widths)

    payload = f"""#usda 1.0
(
    defaultPrim = "BezierCurves"
)

def BasisCurves "BezierCurves"
{{
    uniform token basis = "bezier"
    uniform token type = "cubic"
    uniform token wrap = "nonperiodic"
    int[] curveVertexCounts = [{counts_text}]
    point3f[] points = [
            {points_text}
    ]
    float[] widths = [{widths_text}] (
        interpolation = "uniform"
    )
}}
"""
    path.write_text(payload, encoding="utf-8")


def fit_curve_obj_with_beziers(
    curve_path: str | Path,
    *,
    nm_per_unit: float,
    handle_scale: float = 1.0 / 3.0,
    smooth_tangent_iters: int = 2,
    validation_tol_nm: float | None = 0.01,
    glb_tessellation_tol_nm: float = 0.005,
    glb_tessellation_max_depth: int = 9,
    required_rmin_nm: float | None = None,
    required_min_separation_nm: float | None = None,
    tube_radius_nm: float = 0.5,
    tube_radial_segments: int = 12,
    output_dir: str | Path | None = None,
) -> FitResult:
    if nm_per_unit <= 0.0:
        raise ValueError("nm_per_unit must be positive")
    if handle_scale <= 0.0:
        raise ValueError("handle_scale must be positive")
    if validation_tol_nm is not None and validation_tol_nm <= 0.0:
        raise ValueError("validation_tol_nm must be positive")
    if glb_tessellation_tol_nm <= 0.0:
        raise ValueError("glb_tessellation_tol_nm must be positive")
    if glb_tessellation_max_depth < 1:
        raise ValueError("glb_tessellation_max_depth must be at least 1")

    curve_path = Path(curve_path).expanduser().resolve()
    vertices, edges = read_curve_obj(curve_path)
    output_json, validation_json, sampled_obj, usda_curves = _default_output_paths(
        curve_path,
        Path(output_dir).expanduser().resolve() if output_dir is not None else None,
    )
    glb_mesh = _default_glb_path(
        curve_path,
        Path(output_dir).expanduser().resolve() if output_dir is not None else None,
    )

    component_inputs: list[_ComponentFitInput] = []
    for component_id, component_edges in enumerate(_connected_components(len(vertices), edges)):
        points, closed = _component_points(vertices, component_edges)
        if len(points) < 2:
            continue
        tangents = _estimate_tangents(points, smooth_tangent_iters, closed)
        component_inputs.append(
            _ComponentFitInput(
                points=points,
                tangents=tangents,
                closed=closed,
                component_id=component_id,
            )
        )

    def evaluate_fit(scale: float) -> tuple[
        list[CubicBezierCurve],
        list[ValidationSummary],
        list[tuple[list[Vec3], bool]],
        ValidationReport,
    ]:
        curves_local: list[CubicBezierCurve] = []
        validation_local: list[ValidationSummary] = []
        sampled_components_local: list[tuple[list[Vec3], bool]] = []

        for component_input in component_inputs:
            component_curves = _fit_cubic_chain(
                component_input.points,
                component_input.tangents,
                handle_scale * scale,
                component_input.component_id,
                component_input.closed,
            )
            curves_local.extend(component_curves)
            if validation_tol_nm is None:
                sampled_points = _sample_component_curves(component_curves, 0.002)
            else:
                component_validation, sampled_points = _validate_component(
                    component_curves,
                    component_input.points,
                    component_input.closed,
                    nm_per_unit,
                    validation_tol_nm,
                )
                validation_local.append(component_validation)
            sampled_components_local.append((sampled_points, component_input.closed))

        min_separation_nm_local = _min_inter_component_separation_nm(
            sampled_components_local, nm_per_unit
        )
        passes_rmin_local = (
            True
            if required_rmin_nm is None
            else all(
                summary.min_radius_nm is None or summary.min_radius_nm >= required_rmin_nm
                for summary in validation_local
            )
        )
        passes_min_separation_local = (
            True
            if required_min_separation_nm is None
            else (
                min_separation_nm_local is None
                or min_separation_nm_local >= required_min_separation_nm
            )
        )
        report_local = ValidationReport(
            components=validation_local,
            min_separation_nm=min_separation_nm_local,
            required_rmin_nm=required_rmin_nm,
            required_min_separation_nm=required_min_separation_nm,
            passes_rmin=passes_rmin_local,
            passes_min_separation=passes_min_separation_local,
            passes_all=passes_rmin_local and passes_min_separation_local,
        )
        return curves_local, validation_local, sampled_components_local, report_local

    curves, validation, sampled_components, report = evaluate_fit(1.0)
    if validation_tol_nm is not None and not report.passes_all:
        curves, validation, sampled_components, report = evaluate_fit(0.0)

    glb_sampled_components = []
    glb_tol_units = glb_tessellation_tol_nm / nm_per_unit
    component_offset = 0
    for component_input in component_inputs:
        segment_count = len(component_input.points) if component_input.closed else len(component_input.points) - 1
        component_curves = curves[component_offset : component_offset + segment_count]
        component_offset += segment_count
        glb_sampled_components.append(
            (
                _sample_component_curves_with_max_depth(
                    component_curves, glb_tol_units, glb_tessellation_max_depth
                ),
                component_input.closed,
            )
        )

    output_json.write_text(
        json.dumps({"curves": [asdict(curve) for curve in curves]}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_sampled_obj(sampled_obj, sampled_components)
    _write_usda_curves(usda_curves, curves, nm_per_unit, tube_radius_nm)
    _write_glb_tube(
        glb_mesh,
        glb_sampled_components,
        nm_per_unit,
        tube_radius_nm,
        tube_radial_segments,
    )
    validation_json.write_text(
        json.dumps(
            {"validation": [asdict(summary) for summary in validation], "report": asdict(report)},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return FitResult(
        curves=curves,
        validation=validation,
        report=report,
        output_json=output_json,
        validation_json=validation_json,
        sampled_obj=sampled_obj,
        usda_curves=usda_curves,
        glb_mesh=glb_mesh,
    )


def _write_sampled_obj(path: Path, sampled_components: list[tuple[list[Vec3], bool]]) -> None:
    lines: list[str] = []
    vertex_offset = 1
    for points, closed in sampled_components:
        if not points:
            continue
        for point in points:
            lines.append(f"v {point[0]} {point[1]} {point[2]}")
        index_values = [str(vertex_offset + idx) for idx in range(len(points))]
        if closed:
            index_values.append(str(vertex_offset))
        indices = " ".join(index_values)
        lines.append(f"l {indices}")
        vertex_offset += len(points)

    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
