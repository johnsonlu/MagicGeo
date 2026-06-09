import math
import re
import time

import numpy as np
from scipy.optimize import minimize

from geo.Auxiliary_function import check_condition_break, eval_geometry_condition
from geo.Geometric_function import (
    POINT_CLOSE_TOLERANCE,
    check_is_calculate,
    point_to_line_distance,
)
from geo.coordinate_engine_config import (
    BOOLEAN_PENALTY,
    CONTINUOUS_CONSTRAINTS,
    COORDINATE_ENGINE_TIMEOUT,
    COORDINATE_HIGH,
    COORDINATE_LOW,
    EARLY_EXIT_PENALTY,
    NELDER_MEAD_MAXITER,
    NELDER_MEAD_RESTARTS,
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


def _constraint_residual(func_name, params, variables, coordinates):
    if func_name == "dist" and len(params) == 3:
        points = _resolve_points(variables, coordinates, params[:2])
        if points is None:
            return None
        a, b = points
        target = params[2]
        actual = math.hypot(a[0] - b[0], a[1] - b[1])
        normalizer = max(target, 0.1)
        return ((actual - target) / normalizer) ** 2

    if func_name == "angle" and len(params) == 4:
        points = _resolve_points(variables, coordinates, params[:3])
        if points is None:
            return None
        b, a, c = points
        expected_angle = params[3]
        ab = (b[0] - a[0], b[1] - a[1])
        ac = (c[0] - a[0], c[1] - a[1])
        angle_deg = _calculate_angle_deg(ab, ac)
        return ((angle_deg - expected_angle) / 180) ** 2

    if func_name == "equal_line" and len(params) == 4:
        points = _resolve_points(variables, coordinates, params)
        if points is None:
            return None
        a, b, c, d = points
        dist_ab_sq = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        dist_cd_sq = (c[0] - d[0]) ** 2 + (c[1] - d[1]) ** 2
        normalizer = max(dist_ab_sq, dist_cd_sq, 0.1)
        return ((dist_ab_sq - dist_cd_sq) / normalizer) ** 2

    if func_name == "line_ratio" and len(params) == 5:
        points = _resolve_points(variables, coordinates, params[:4])
        if points is None:
            return None
        a, b, f, g = points
        k = params[4]
        dist_ab_sq = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        dist_fg_sq = (f[0] - g[0]) ** 2 + (f[1] - g[1]) ** 2
        expected = dist_ab_sq * (k ** 2)
        normalizer = max(expected, 0.1)
        return ((dist_fg_sq - expected) / normalizer) ** 2

    if func_name == "angle_relation" and len(params) == 7:
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

    if func_name == "ortho" and len(params) == 4:
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

    if func_name in {"online", "online_inside", "online_extension"} and len(params) == 3:
        points = _resolve_points(variables, coordinates, params)
        if points is None:
            return None
        b, e, f = points
        line_len = math.hypot(f[0] - e[0], f[1] - e[1])
        normalizer = max(line_len, 0.1)
        residual = (point_to_line_distance(b, e, f) / normalizer) ** 2
        if func_name == "online":
            return residual

        if x_f := f[0] - e[0]:
            k = (b[0] - e[0]) / x_f
        elif y_f := f[1] - e[1]:
            k = (b[1] - e[1]) / y_f
        else:
            return residual + 1.0

        if func_name == "online_inside":
            if k <= 0:
                residual += k ** 2
            elif k >= 1:
                residual += (k - 1) ** 2
        elif func_name == "online_extension":
            if 0 <= k <= 1:
                residual += min(k, 1 - k) ** 2
        return residual

    if func_name == "parallel" and len(params) == 4:
        points = _resolve_points(variables, coordinates, params)
        if points is None:
            return None
        a, b, c, d = points
        # Cross product of direction vectors: zero when parallel
        cross = (b[0] - a[0]) * (d[1] - c[1]) - (b[1] - a[1]) * (d[0] - c[0])
        norm_ab = math.hypot(b[0] - a[0], b[1] - a[1])
        norm_cd = math.hypot(d[0] - c[0], d[1] - c[1])
        normalizer = max(norm_ab * norm_cd, 0.1)
        return (cross / normalizer) ** 2

    if func_name == "is_point_in_triangle" and len(params) == 4:
        points = _resolve_points(variables, coordinates, params)
        if points is None:
            return None
        A, B, C, P = points
        def _cross(O, X, Y):
            return (X[0] - O[0]) * (Y[1] - O[1]) - (X[1] - O[1]) * (Y[0] - O[0])
        # Triangle orientation (independent of P)
        orient = _cross(A, B, C)
        if abs(orient) < 1e-10:
            return 0.0
        # Signed distances from P to each edge, normalized so inside = positive
        sign = 1.0 if orient > 0 else -1.0
        d1 = sign * _cross(A, B, P) / math.hypot(B[0] - A[0], B[1] - A[1])
        d2 = sign * _cross(B, C, P) / math.hypot(C[0] - B[0], C[1] - B[1])
        d3 = sign * _cross(C, A, P) / math.hypot(A[0] - C[0], A[1] - C[1])
        if d1 >= 0 and d2 >= 0 and d3 >= 0:
            return 0.0
        # Outside: penalize violated edges
        penalty = 0.0
        for d in (d1, d2, d3):
            if d < 0:
                penalty += d ** 2
        return penalty

    if func_name == "is_point_out_triangle" and len(params) == 4:
        points = _resolve_points(variables, coordinates, params)
        if points is None:
            return None
        A, B, C, P = points
        def _cross(O, X, Y):
            return (X[0] - O[0]) * (Y[1] - O[1]) - (X[1] - O[1]) * (Y[0] - O[0])
        orient = _cross(A, B, C)
        if abs(orient) < 1e-10:
            return 0.0
        sign = 1.0 if orient > 0 else -1.0
        d1 = sign * _cross(A, B, P) / math.hypot(B[0] - A[0], B[1] - A[1])
        d2 = sign * _cross(B, C, P) / math.hypot(C[0] - B[0], C[1] - B[1])
        d3 = sign * _cross(C, A, P) / math.hypot(A[0] - C[0], A[1] - C[1])
        if d1 >= 0 and d2 >= 0 and d3 >= 0:
            # Inside: penalize by closest edge (must cross to get out)
            return min(d1, d2, d3) ** 2
        return 0.0

    if func_name == "is_acute_triangle" and len(params) == 3:
        points = _resolve_points(variables, coordinates, params)
        if points is None:
            return None
        A, B, C = points
        AB = (B[0] - A[0], B[1] - A[1])
        AC = (C[0] - A[0], C[1] - A[1])
        BA = (-AB[0], -AB[1])
        BC = (C[0] - B[0], C[1] - B[1])
        CA = (-AC[0], -AC[1])
        CB = (-BC[0], -BC[1])
        dot_A = AB[0] * AC[0] + AB[1] * AC[1]
        dot_B = BA[0] * BC[0] + BA[1] * BC[1]
        dot_C = CA[0] * CB[0] + CA[1] * CB[1]
        norm_AB = math.hypot(*AB)
        norm_AC = math.hypot(*AC)
        norm_BC = math.hypot(*BC)
        n_A = max(norm_AB * norm_AC, 0.1)
        n_B = max(norm_AB * norm_BC, 0.1)
        n_C = max(norm_AC * norm_BC, 0.1)
        if dot_A > 0 and dot_B > 0 and dot_C > 0:
            return 0.0
        penalty = 0.0
        for dot, n in ((dot_A, n_A), (dot_B, n_B), (dot_C, n_C)):
            if dot <= 0:
                penalty += (dot / n) ** 2
        return penalty

    if func_name == "angle_bisector" and len(params) == 5:
        points = _resolve_points(variables, coordinates, params)
        if points is None:
            return None
        A, D, C, A_dup, B = points
        AB = (B[0] - A[0], B[1] - A[1])
        AD = (D[0] - A[0], D[1] - A[1])
        AC = (C[0] - A[0], C[1] - A[1])
        angle_BAD = _calculate_angle_deg(AB, AD)
        angle_CAD = _calculate_angle_deg(AC, AD)
        if angle_BAD < 10 or angle_CAD < 10:
            return 0.1
        return ((angle_BAD - angle_CAD) / 180) ** 2

    if func_name == "arc_midpoint" and len(params) == 3:
        points = _resolve_points(variables, coordinates, params)
        if points is None:
            return None
        A, B, C = points
        dist_AB_sq = (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2
        dist_AC_sq = (A[0] - C[0]) ** 2 + (A[1] - C[1]) ** 2
        normalizer = max(dist_AB_sq, dist_AC_sq, 0.1)
        return ((dist_AB_sq - dist_AC_sq) / normalizer) ** 2

    return None


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


def _resolve_known_points(coordinates):
    """Return list of (x, y) for coordinates with fully known numeric values."""
    resolved = []
    for ref in coordinates.values():
        x, y = ref
        if not isinstance(x, str) and not isinstance(y, str):
            resolved.append((float(x), float(y)))
    return resolved


def extract_and_modify_nelder_mead(coordinates, condition_code, variables):
    key_list = list(variables.keys())
    if not key_list:
        return coordinates, variables, True

    print(f"Nelder-Mead: {len(key_list)} variables, {NELDER_MEAD_RESTARTS} restarts")
    print(f"Keys: {key_list}")

    rng = np.random.default_rng()
    deadline = time.monotonic() + COORDINATE_ENGINE_TIMEOUT
    best_variables = None
    best_penalty = float("inf")
    best_x = None
    known_points = _resolve_known_points(coordinates)
    centroid_x = sum(p[0] for p in known_points) / len(known_points) if known_points else 0.0
    centroid_y = sum(p[1] for p in known_points) / len(known_points) if known_points else 0.0
    n_vars = len(key_list)

    def objective(values):
        values = np.clip(values, COORDINATE_LOW, COORDINATE_HIGH)
        state = _apply_values(variables, key_list, values)
        return _compute_penalty(condition_code, coordinates, state, BOOLEAN_PENALTY)

    for restart in range(NELDER_MEAD_RESTARTS):
        if time.monotonic() >= deadline:
            break

        # Seed strategy: geometric for first 2 (if centroid is away from origin),
        # then adaptive/random mix
        use_geometric = (
            restart < 2
            and known_points
            and math.hypot(centroid_x, centroid_y) > 0.3
        )
        if use_geometric:
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
            print(f"Restart {restart + 1}: feasible assignment (penalty={raw_penalty:.6f})")
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
            print(f"L-BFGS-B polish: feasible assignment (penalty={polish_penalty:.6f})")
        elif best_variables is None and polish_feasible:
            best_variables = polish_candidate
            print(f"L-BFGS-B polish: rescued feasible assignment (penalty={polish_penalty:.6f})")

    if best_variables is not None:
        for key in variables:
            variables[key] = best_variables[key]
        print("\nfound a feasible solution!\n")
        print(variables)
        print(coordinates)
        return coordinates, variables, True

    print("\ndidn't find a feasible solution!\n")
    return coordinates, variables, False
