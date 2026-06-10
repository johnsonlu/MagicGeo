import math
import re
import time

import numpy as np
from scipy.optimize import minimize

from geo.Auxiliary_function import check_condition_break, eval_geometry_condition
from geo.coordinate_engine_config import (
    BOOLEAN_PENALTY,
    CONTINUOUS_CONSTRAINTS,
    COORDINATE_ENGINE_TIMEOUT,
    COORDINATE_HIGH,
    COORDINATE_LOW,
    EARLY_EXIT_PENALTY,
    NELDER_MEAD_MAXITER,
    effective_nelder_mead_restarts,
)
from geo.Geometric_function import (
    POINT_CLOSE_TOLERANCE,
    check_is_calculate,
    point_to_line_distance,
)

_EXECUTE_CODE_RE = re.compile(r"^(\w+)\((?:variables,)?(.+),coordinates\)$")
_COORD_REF_RE = re.compile(r"coordinates\['(\w+)'\]")
_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def _parse_execute_code(execute_code):
    match = _EXECUTE_CODE_RE.match(execute_code.strip())
    if not match:
        return None, []
    func_name = match.group(1)
    params = []
    for token in match.group(2).split(","):
        token = token.strip()
        coord_match = _COORD_REF_RE.fullmatch(token)
        if coord_match:
            params.append(coord_match.group(1))
        elif _NUMERIC_RE.match(token):
            params.append(float(token))
        elif token == "variables":
            continue  # skip namespace parameter
        else:
            return None, []
    return func_name, params


def _resolve_points(variables, coordinates, point_names):
    refs = [coordinates[name] for name in point_names]
    is_calculate, point_list = check_is_calculate(variables, refs, coordinates)
    if not is_calculate:
        return None
    return point_list


def _calculate_angle_deg(v1, v2):
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    length_v1 = math.hypot(v1[0], v1[1])
    length_v2 = math.hypot(v2[0], v2[1])
    if length_v1 * length_v2 == 0:
        return 0.0
    cos_angle = max(min(dot_product / (length_v1 * length_v2), 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))


def _cross_2d(origin, x, y):
    return (x[0] - origin[0]) * (y[1] - origin[1]) - (x[1] - origin[1]) * (
        y[0] - origin[0]
    )


def _squared_segment_distances_sq(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _triangle_edge_distances(triangle, point):
    a, b, c = triangle
    orient = _cross_2d(a, b, c)
    if abs(orient) < 1e-10:
        return orient, ()
    sign = 1.0 if orient > 0 else -1.0
    return orient, (
        sign * _cross_2d(a, b, point) / math.hypot(b[0] - a[0], b[1] - a[1]),
        sign * _cross_2d(b, c, point) / math.hypot(c[0] - b[0], c[1] - b[1]),
        sign * _cross_2d(c, a, point) / math.hypot(a[0] - c[0], a[1] - c[1]),
    )


def _dist_residual(params, variables, coordinates):
    if len(params) != 3:
        return None
    points = _resolve_points(variables, coordinates, params[:2])
    if points is None:
        return None
    a, b = points
    target = params[2]
    actual = math.hypot(a[0] - b[0], a[1] - b[1])
    normalizer = max(target, 0.1)
    return ((actual - target) / normalizer) ** 2


def _angle_residual(params, variables, coordinates):
    if len(params) != 4:
        return None
    points = _resolve_points(variables, coordinates, params[:3])
    if points is None:
        return None
    b, a, c = points
    expected_angle = params[3]
    ab = (b[0] - a[0], b[1] - a[1])
    ac = (c[0] - a[0], c[1] - a[1])
    angle_deg = _calculate_angle_deg(ab, ac)
    return ((angle_deg - expected_angle) / 180) ** 2


def _equal_line_residual(params, variables, coordinates):
    if len(params) != 4:
        return None
    points = _resolve_points(variables, coordinates, params)
    if points is None:
        return None
    a, b, c, d = points
    dist_ab_sq = _squared_segment_distances_sq(a, b)
    dist_cd_sq = _squared_segment_distances_sq(c, d)
    normalizer = max(dist_ab_sq, dist_cd_sq, 0.1)
    return ((dist_ab_sq - dist_cd_sq) / normalizer) ** 2


def _line_ratio_residual(params, variables, coordinates):
    if len(params) != 5:
        return None
    points = _resolve_points(variables, coordinates, params[:4])
    if points is None:
        return None
    a, b, f, g = points
    k = params[4]
    dist_ab_sq = _squared_segment_distances_sq(a, b)
    dist_fg_sq = _squared_segment_distances_sq(f, g)
    expected = dist_ab_sq * (k**2)
    normalizer = max(expected, 0.1)
    return ((dist_fg_sq - expected) / normalizer) ** 2


def _angle_relation_residual(params, variables, coordinates):
    if len(params) != 7:
        return None
    points = _resolve_points(variables, coordinates, params[:6])
    if points is None:
        return None
    a, b, c, d, e, f = points
    ratio = params[6]
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    ed = (d[0] - e[0], d[1] - e[1])
    ef = (f[0] - e[0], f[1] - e[1])
    angle_abc = _calculate_angle_deg(ba, bc)
    angle_def = _calculate_angle_deg(ed, ef)
    return ((angle_abc - ratio * angle_def) / 180) ** 2


def _ortho_residual(params, variables, coordinates):
    if len(params) != 4:
        return None
    points = _resolve_points(variables, coordinates, params)
    if points is None:
        return None
    a, b, e, f = points
    ab = (b[0] - a[0], b[1] - a[1])
    ef = (f[0] - e[0], f[1] - e[1])
    dot_product = ab[0] * ef[0] + ab[1] * ef[1]
    norm_ab = math.hypot(ab[0], ab[1])
    norm_ef = math.hypot(ef[0], ef[1])
    normalizer = max(norm_ab * norm_ef, 0.1)
    return (dot_product / normalizer) ** 2


def _segment_parameter_k(point, seg_start, seg_end):
    if x_f := seg_end[0] - seg_start[0]:
        return (point[0] - seg_start[0]) / x_f
    if y_f := seg_end[1] - seg_start[1]:
        return (point[1] - seg_start[1]) / y_f
    return None


def _online_residual(mode, params, variables, coordinates):
    if len(params) != 3:
        return None
    points = _resolve_points(variables, coordinates, params)
    if points is None:
        return None
    b, e, f = points
    line_len = math.hypot(f[0] - e[0], f[1] - e[1])
    normalizer = max(line_len, 0.1)
    residual = (point_to_line_distance(b, e, f) / normalizer) ** 2
    if mode == "online":
        return residual

    k = _segment_parameter_k(b, e, f)
    if k is None:
        return residual + 1.0

    if mode == "online_inside":
        if k <= 0:
            residual += k**2
        elif k >= 1:
            residual += (k - 1) ** 2
    elif mode == "online_extension" and 0 <= k <= 1:
        residual += min(k, 1 - k) ** 2
    return residual


def _parallel_residual(params, variables, coordinates):
    if len(params) != 4:
        return None
    points = _resolve_points(variables, coordinates, params)
    if points is None:
        return None
    a, b, c, d = points
    cross = (b[0] - a[0]) * (d[1] - c[1]) - (b[1] - a[1]) * (d[0] - c[0])
    norm_ab = math.hypot(b[0] - a[0], b[1] - a[1])
    norm_cd = math.hypot(d[0] - c[0], d[1] - c[1])
    normalizer = max(norm_ab * norm_cd, 0.1)
    return (cross / normalizer) ** 2


def _midpoint_residual(params, variables, coordinates):
    if len(params) != 3:
        return None
    points = _resolve_points(variables, coordinates, params)
    if points is None:
        return None
    a, b, c = points
    mid_x = (b[0] + c[0]) / 2
    mid_y = (b[1] + c[1]) / 2
    span = math.hypot(c[0] - b[0], c[1] - b[1])
    normalizer = max(span, 0.1)
    dx = a[0] - mid_x
    dy = a[1] - mid_y
    return (dx / normalizer) ** 2 + (dy / normalizer) ** 2


def _is_point_in_triangle_residual(params, variables, coordinates):
    if len(params) != 4:
        return None
    points = _resolve_points(variables, coordinates, params)
    if points is None:
        return None
    a, b, c, p = points
    _, edge_distances = _triangle_edge_distances((a, b, c), p)
    if not edge_distances:
        return 0.0
    d1, d2, d3 = edge_distances
    if d1 >= 0 and d2 >= 0 and d3 >= 0:
        return 0.0
    return sum(d**2 for d in (d1, d2, d3) if d < 0)


def _is_point_out_triangle_residual(params, variables, coordinates):
    if len(params) != 4:
        return None
    points = _resolve_points(variables, coordinates, params)
    if points is None:
        return None
    a, b, c, p = points
    _, edge_distances = _triangle_edge_distances((a, b, c), p)
    if not edge_distances:
        return 0.0
    d1, d2, d3 = edge_distances
    if d1 >= 0 and d2 >= 0 and d3 >= 0:
        return min(d1, d2, d3) ** 2
    return 0.0


def _is_acute_triangle_residual(params, variables, coordinates):
    if len(params) != 3:
        return None
    points = _resolve_points(variables, coordinates, params)
    if points is None:
        return None
    a, b, c = points
    ab = (b[0] - a[0], b[1] - a[1])
    ac = (c[0] - a[0], c[1] - a[1])
    ba = (-ab[0], -ab[1])
    bc = (c[0] - b[0], c[1] - b[1])
    ca = (-ac[0], -ac[1])
    cb = (-bc[0], -bc[1])
    dot_a = ab[0] * ac[0] + ab[1] * ac[1]
    dot_b = ba[0] * bc[0] + ba[1] * bc[1]
    dot_c = ca[0] * cb[0] + ca[1] * cb[1]
    norm_ab = math.hypot(*ab)
    norm_ac = math.hypot(*ac)
    norm_bc = math.hypot(*bc)
    n_a = max(norm_ab * norm_ac, 0.1)
    n_b = max(norm_ab * norm_bc, 0.1)
    n_c = max(norm_ac * norm_bc, 0.1)
    if dot_a > 0 and dot_b > 0 and dot_c > 0:
        return 0.0
    return sum((dot / n) ** 2 for dot, n in ((dot_a, n_a), (dot_b, n_b), (dot_c, n_c)) if dot <= 0)


def _angle_bisector_residual(params, variables, coordinates):
    if len(params) != 5:
        return None
    points = _resolve_points(variables, coordinates, params)
    if points is None:
        return None
    a, d, c, _a_dup, b = points
    ab = (b[0] - a[0], b[1] - a[1])
    ad = (d[0] - a[0], d[1] - a[1])
    ac = (c[0] - a[0], c[1] - a[1])
    angle_bad = _calculate_angle_deg(ab, ad)
    angle_cad = _calculate_angle_deg(ac, ad)
    if angle_bad < 10 or angle_cad < 10:
        return 0.1
    return ((angle_bad - angle_cad) / 180) ** 2


def _arc_midpoint_residual(params, variables, coordinates):
    if len(params) != 3:
        return None
    points = _resolve_points(variables, coordinates, params)
    if points is None:
        return None
    a, b, c = points
    dist_ab_sq = _squared_segment_distances_sq(a, b)
    dist_ac_sq = _squared_segment_distances_sq(a, c)
    normalizer = max(dist_ab_sq, dist_ac_sq, 0.1)
    return ((dist_ab_sq - dist_ac_sq) / normalizer) ** 2


_CONTINUOUS_RESIDUAL_HANDLERS = {
    "dist": _dist_residual,
    "angle": _angle_residual,
    "equal_line": _equal_line_residual,
    "line_ratio": _line_ratio_residual,
    "angle_relation": _angle_relation_residual,
    "ortho": _ortho_residual,
    "online": lambda params, variables, coordinates: _online_residual(
        "online", params, variables, coordinates
    ),
    "online_inside": lambda params, variables, coordinates: _online_residual(
        "online_inside", params, variables, coordinates
    ),
    "online_extension": lambda params, variables, coordinates: _online_residual(
        "online_extension", params, variables, coordinates
    ),
    "parallel": _parallel_residual,
    "midpoint": _midpoint_residual,
    "is_point_in_triangle": _is_point_in_triangle_residual,
    "is_point_out_triangle": _is_point_out_triangle_residual,
    "is_acute_triangle": _is_acute_triangle_residual,
    "angle_bisector": _angle_bisector_residual,
    "arc_midpoint": _arc_midpoint_residual,
}


def _constraint_residual(func_name, params, variables, coordinates):
    handler = _CONTINUOUS_RESIDUAL_HANDLERS.get(func_name)
    if handler is None:
        return None
    return handler(params, variables, coordinates)


def _constraint_contribution(execute_code, variables, coordinates, boolean_penalty):
    func_name, params = _parse_execute_code(execute_code)
    if func_name is None:
        return boolean_penalty

    if func_name in CONTINUOUS_CONSTRAINTS:
        residual = _constraint_residual(func_name, params, variables, coordinates)
        if residual is None:
            return boolean_penalty
        return residual

    try:
        is_satisfied, is_calculate = eval_geometry_condition(
            execute_code, variables, coordinates
        )
        if is_calculate and not is_satisfied:
            return boolean_penalty
    except Exception:
        return boolean_penalty
    return 0.0


def _coordinates_distinct_penalty(coordinates, variables, boolean_penalty):
    resolved = []
    for ref in coordinates.values():
        is_calculate, point_list = check_is_calculate(variables, [ref], coordinates)
        if not is_calculate:
            return 0.0
        resolved.append(point_list[0])

    penalty = 0.0
    for i in range(len(resolved)):
        for j in range(i + 1, len(resolved)):
            gap = math.hypot(
                resolved[i][0] - resolved[j][0],
                resolved[i][1] - resolved[j][1],
            )
            if gap < POINT_CLOSE_TOLERANCE:
                penalty += ((POINT_CLOSE_TOLERANCE - gap) / POINT_CLOSE_TOLERANCE) ** 2
    return penalty * boolean_penalty


def _compute_penalty(condition_code, coordinates, variables, boolean_penalty):
    constraint_penalty = sum(
        _constraint_contribution(code, variables, coordinates, boolean_penalty)
        for code in condition_code
    )
    return constraint_penalty + _coordinates_distinct_penalty(
        coordinates, variables, boolean_penalty
    )


def _apply_values(variables, key_list, values):
    state = {key: [variables[key][0], variables[key][1]] for key in variables}
    for index, key in enumerate(key_list):
        state[key] = [float(values[index]), True]
    return state


def _is_feasible(condition_code, coordinates, variables):
    return not check_condition_break(condition_code, coordinates, variables)


def _resolve_numeric_coord(ref, variables, coordinates):
    """Return (x, y) as floats when fully numeric, else None."""
    is_calculate, point_list = check_is_calculate(variables, [ref], coordinates)
    if not is_calculate:
        return None
    x, y = point_list[0]
    if isinstance(x, str) or isinstance(y, str):
        return None
    return float(x), float(y)


def _resolve_known_points(coordinates, variables):
    """Return list of (x, y) for coordinates with fully known numeric values."""
    resolved = []
    for ref in coordinates.values():
        coord = _resolve_numeric_coord(ref, variables, coordinates)
        if coord is not None:
            resolved.append(coord)
    return resolved


_CIRCLE_CONSTRAINT_RE = re.compile(
    r"^dist\(variables,coordinates\['(\w+)'\],coordinates\['(\w+)'\],([\d.]+),coordinates\)$"
)


def _parse_dist_circle_constraint(code):
    match = _CIRCLE_CONSTRAINT_RE.match(code.strip())
    if not match:
        return None
    return match.group(1), match.group(2), float(match.group(3))


def _point_has_unknown_variables(point_ref):
    if len(point_ref) != 2:
        return False
    px, py = point_ref
    return isinstance(px, str) or isinstance(py, str)


def _circle_constraint_entry(code, coordinates, variables):
    parsed = _parse_dist_circle_constraint(code)
    if parsed is None:
        return None
    center_name, point_name, radius = parsed
    if center_name not in coordinates or point_name not in coordinates:
        return None
    if _resolve_numeric_coord(coordinates[center_name], variables, coordinates) is None:
        return None
    if not _point_has_unknown_variables(coordinates[point_name]):
        return None
    return center_name, radius, point_name


def _build_circle_constraint_results(circles, coordinates, variables):
    result = []
    for (center_name, radius), points in circles.items():
        center_xy = _resolve_numeric_coord(
            coordinates[center_name], variables, coordinates
        )
        if center_xy is None:
            continue
        result.append((center_xy, radius, points))
    return result


def _detect_circle_constraints(condition_code, coordinates, variables):
    """Find groups of unknown points pinned to the same circle.

    Returns a list of (center_xy, radius, [var_x_key, var_y_key, ...]) tuples.
    Each entry in the var list is a (x_key, y_key) pair for an unknown point
    that must lie on that circle.
    """
    circles = {}
    for code in condition_code:
        entry = _circle_constraint_entry(code, coordinates, variables)
        if entry is None:
            continue
        center_name, radius, point_name = entry
        circles.setdefault((center_name, radius), []).append((point_name, radius))
    return _build_circle_constraint_results(circles, coordinates, variables)


def extract_and_modify_nelder_mead(coordinates, condition_code, variables):
    key_list = list(variables.keys())
    if not key_list:
        return coordinates, variables, True

    n_restarts = effective_nelder_mead_restarts(len(key_list))
    print(f"Nelder-Mead: {len(key_list)} variables, {n_restarts} restarts")
    print(f"Keys: {key_list}")

    rng = np.random.default_rng()
    deadline = time.monotonic() + COORDINATE_ENGINE_TIMEOUT
    best_variables = None
    best_penalty = float("inf")
    best_x = None
    known_points = _resolve_known_points(coordinates, variables)
    centroid_x = (
        sum(p[0] for p in known_points) / len(known_points) if known_points else 0.0
    )
    centroid_y = (
        sum(p[1] for p in known_points) / len(known_points) if known_points else 0.0
    )
    n_vars = len(key_list)

    # Detect circle constraints for smart seeding
    circle_constraints = _detect_circle_constraints(
        condition_code, coordinates, variables
    )
    # Build circle-aware seeds: place unknown points on their circles
    # at evenly-spaced base angles with random jitter for diversity
    circle_seeds = []
    if circle_constraints:
        n_circle_seeds = max(0, min(6, n_restarts - 2))
        for seed_idx in range(n_circle_seeds):
            base_angle = (2 * math.pi * seed_idx) / n_circle_seeds
            # Add jitter so seeds explore a range of angles, not just fixed slots
            jitter = (
                rng.uniform(-math.pi / n_circle_seeds, math.pi / n_circle_seeds)
                if n_circle_seeds > 1
                else rng.uniform(-math.pi, math.pi)
            )
            seed_values = {}
            for (cx, cy), radius, points in circle_constraints:
                for i, (point_name, _r) in enumerate(points):
                    spread = (2 * math.pi * i) / max(len(points), 1)
                    angle_offset = base_angle + spread + jitter
                    # Find variable keys for this point's coordinates
                    px_ref, py_ref = coordinates[point_name]
                    if isinstance(px_ref, str) and px_ref in key_list:
                        seed_values[px_ref] = cx + radius * math.cos(angle_offset)
                    if isinstance(py_ref, str) and py_ref in key_list:
                        seed_values[py_ref] = cy + radius * math.sin(angle_offset)
            # For non-circle variables, use centroid or origin
            seed = np.empty(n_vars)
            for i, key in enumerate(key_list):
                if key in seed_values:
                    seed[i] = seed_values[key]
                elif i % 2 == 0:
                    seed[i] = centroid_x + rng.normal(0, 0.2)
                else:
                    seed[i] = centroid_y + rng.normal(0, 0.2)
            seed = np.clip(seed, COORDINATE_LOW, COORDINATE_HIGH)
            circle_seeds.append(seed)

    def objective(values):
        values = np.clip(values, COORDINATE_LOW, COORDINATE_HIGH)
        state = _apply_values(variables, key_list, values)
        return _compute_penalty(condition_code, coordinates, state, BOOLEAN_PENALTY)

    for restart in range(n_restarts):
        if time.monotonic() >= deadline:
            break

        # Seed strategy:
        # 1. Circle-aware seeds (if available)
        # 2. Geometric seed near centroid (if centroid away from origin)
        # 3. Adaptive perturbation of best_x / random
        if restart < len(circle_seeds):
            x0 = circle_seeds[restart]
        elif (
            known_points
            and math.hypot(centroid_x, centroid_y) > 0.3
            and restart < len(circle_seeds) + 2
        ):
            x0 = np.empty(n_vars)
            for i in range(n_vars):
                if i % 2 == 0:
                    x0[i] = centroid_x + rng.normal(0, 0.3)
                else:
                    x0[i] = centroid_y + rng.normal(0, 0.3)
            x0 = np.clip(x0, COORDINATE_LOW, COORDINATE_HIGH)
        elif best_x is not None and rng.random() < 0.7:
            x0 = best_x + rng.normal(0, 0.3, size=n_vars)
            x0 = np.clip(x0, COORDINATE_LOW, COORDINATE_HIGH)
        else:
            x0 = rng.uniform(COORDINATE_LOW, COORDINATE_HIGH, size=n_vars)

        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": NELDER_MEAD_MAXITER, "disp": False},
        )

        candidate = _apply_values(variables, key_list, result.x)
        raw_penalty = result.fun
        improved = raw_penalty < best_penalty
        feasible = _is_feasible(condition_code, coordinates, candidate)

        if improved:
            best_penalty = raw_penalty
            best_x = result.x.copy()

        if feasible and improved:
            best_variables = candidate
            print(
                f"Restart {restart + 1}: feasible assignment (penalty={raw_penalty:.6f})"
            )
            if raw_penalty < EARLY_EXIT_PENALTY:
                break

    # L-BFGS-B polish: refine the best Nelder-Mead result with a gradient-based pass
    if best_x is not None and time.monotonic() < deadline:
        bounds = [(COORDINATE_LOW, COORDINATE_HIGH)] * n_vars
        polish_result = minimize(
            objective,
            best_x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": NELDER_MEAD_MAXITER, "disp": False},
        )
        polish_candidate = _apply_values(variables, key_list, polish_result.x)
        polish_penalty = polish_result.fun
        polish_feasible = _is_feasible(condition_code, coordinates, polish_candidate)

        if polish_feasible and polish_penalty < best_penalty:
            best_penalty = polish_penalty
            best_variables = polish_candidate
            print(
                f"L-BFGS-B polish: feasible assignment (penalty={polish_penalty:.6f})"
            )
        elif best_variables is None and polish_feasible:
            best_variables = polish_candidate
            print(
                f"L-BFGS-B polish: rescued feasible assignment (penalty={polish_penalty:.6f})"
            )

    if best_variables is not None:
        for key in variables:
            variables[key] = best_variables[key]
        print("\nfound a feasible solution!\n")
        print(variables)
        print(coordinates)
        return coordinates, variables, True

    print("\ndidn't find a feasible solution!\n")
    return coordinates, variables, False
