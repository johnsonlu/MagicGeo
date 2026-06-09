import math
import re
import time

import numpy as np
from scipy.optimize import minimize

from geo.Auxiliary_function import check_condition_break
from geo.Geometric_function import check_is_calculate, point_to_line_distance
from geo.coordinate_engine_config import (
    BOOLEAN_PENALTY,
    CONTINUOUS_CONSTRAINTS,
    COORDINATE_ENGINE_TIMEOUT,
    COORDINATE_HIGH,
    COORDINATE_LOW,
    NELDER_MEAD_MAXITER,
    NELDER_MEAD_RESTARTS,
)

_EXECUTE_CODE_RE = re.compile(r"^(\w+)\(variables,(.+),coordinates\)$")
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


def _slope(p1, p2):
    if p2[0] == p1[0]:
        return float("inf")
    return (p2[1] - p1[1]) / (p2[0] - p1[0])


def _constraint_residual(func_name, params, variables, coordinates):
    if func_name == "dist" and len(params) == 3:
        points = _resolve_points(variables, coordinates, params[:2])
        if points is None:
            return None
        a, b = points
        target = params[2]
        actual = math.hypot(a[0] - b[0], a[1] - b[1])
        return (actual - target) ** 2

    if func_name == "angle" and len(params) == 4:
        points = _resolve_points(variables, coordinates, params[:3])
        if points is None:
            return None
        b, a, c = points
        expected_angle = params[3]
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        angle_deg = _calculate_angle_deg(ba, bc)
        return (angle_deg - expected_angle) ** 2

    if func_name == "equal_line" and len(params) == 4:
        points = _resolve_points(variables, coordinates, params)
        if points is None:
            return None
        a, b, c, d = points
        dist_ab_sq = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        dist_cd_sq = (c[0] - d[0]) ** 2 + (c[1] - d[1]) ** 2
        return (dist_ab_sq - dist_cd_sq) ** 2

    if func_name == "line_ratio" and len(params) == 5:
        points = _resolve_points(variables, coordinates, params[:4])
        if points is None:
            return None
        a, b, f, g = points
        k = params[4]
        dist_ab_sq = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        dist_fg_sq = (f[0] - g[0]) ** 2 + (f[1] - g[1]) ** 2
        return (dist_fg_sq - dist_ab_sq * (k ** 2)) ** 2

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
        return (angle_abc - ratio * angle_def) ** 2

    if func_name == "ortho" and len(params) == 4:
        points = _resolve_points(variables, coordinates, params)
        if points is None:
            return None
        a, b, e, f = points
        ab = (b[0] - a[0], b[1] - a[1])
        ef = (f[0] - e[0], f[1] - e[1])
        dot_product = ab[0] * ef[0] + ab[1] * ef[1]
        return dot_product ** 2

    if func_name in {"online", "online_inside", "online_extension"} and len(params) == 3:
        points = _resolve_points(variables, coordinates, params)
        if points is None:
            return None
        b, e, f = points
        residual = point_to_line_distance(b, e, f) ** 2
        if func_name == "online":
            return residual

        if x_f := f[0] - e[0]:
            k = (b[0] - e[0]) / x_f
        elif y_f := f[1] - e[1]:
            k = (b[1] - e[1]) / y_f
        else:
            return residual + BOOLEAN_PENALTY

        if func_name == "online_inside":
            if k <= 0 or k >= 1:
                residual += BOOLEAN_PENALTY
        elif func_name == "online_extension" and 0 <= k <= 1:
            residual += BOOLEAN_PENALTY
        return residual

    if func_name == "parallel" and len(params) == 4:
        points = _resolve_points(variables, coordinates, params)
        if points is None:
            return None
        a, b, c, d = points
        slope_ab = _slope(a, b)
        slope_cd = _slope(c, d)
        if math.isinf(slope_ab) and math.isinf(slope_cd):
            return 0.0
        if math.isinf(slope_ab) or math.isinf(slope_cd):
            return BOOLEAN_PENALTY
        return (slope_ab - slope_cd) ** 2

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
        is_satisfied, is_calculate = eval(execute_code)
        if is_calculate and not is_satisfied:
            return boolean_penalty
    except Exception:
        return boolean_penalty
    return 0.0


def _compute_penalty(condition_code, coordinates, variables, boolean_penalty):
    return sum(
        _constraint_contribution(code, variables, coordinates, boolean_penalty)
        for code in condition_code
    )


def _apply_values(variables, key_list, values):
    state = {key: [variables[key][0], variables[key][1]] for key in variables}
    for index, key in enumerate(key_list):
        state[key] = [float(values[index]), True]
    return state


def _is_feasible(condition_code, coordinates, variables):
    return not check_condition_break(condition_code, coordinates, variables)


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

    def objective(values):
        state = _apply_values(variables, key_list, values)
        return _compute_penalty(condition_code, coordinates, state, BOOLEAN_PENALTY)

    for restart in range(NELDER_MEAD_RESTARTS):
        if time.monotonic() >= deadline:
            break

        x0 = rng.uniform(COORDINATE_LOW, COORDINATE_HIGH, size=len(key_list))
        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": NELDER_MEAD_MAXITER, "disp": False},
        )

        candidate = _apply_values(variables, key_list, result.x)
        penalty = _compute_penalty(condition_code, coordinates, candidate, BOOLEAN_PENALTY)
        if _is_feasible(condition_code, coordinates, candidate) and penalty < best_penalty:
            best_penalty = penalty
            best_variables = candidate
            print(f"Restart {restart + 1}: feasible assignment (penalty={penalty:.6f})")

    if best_variables is not None:
        for key in variables:
            variables[key] = best_variables[key]
        print("\nfound a feasible solution!\n")
        print(variables)
        print(coordinates)
        return coordinates, variables, True

    print("\ndidn't find a feasible solution!\n")
    return coordinates, variables, False
