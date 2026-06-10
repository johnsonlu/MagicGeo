import math

import pytest

from tests.conftest import assert_feasible, reload_extract_and_modify


def _dist_code(point_a, point_b, target):
    return (
        f"dist(variables,coordinates['{point_a}'],coordinates['{point_b}'],"
        f"{target},coordinates)"
    )


def _angle_code(point_b, point_a, point_c, degrees):
    return (
        f"angle(variables,coordinates['{point_b}'],coordinates['{point_a}'],"
        f"coordinates['{point_c}'],{degrees},coordinates)"
    )


def _online_code(point_b, point_e, point_f):
    return (
        f"online(variables,coordinates['{point_b}'],coordinates['{point_e}'],"
        f"coordinates['{point_f}'],coordinates)"
    )


def _resolved_distance(coordinates, variables, point_a, point_b):
    def coord(point):
        x, y = coordinates[point]
        if isinstance(x, str):
            x = variables[x][0]
        if isinstance(y, str):
            y = variables[y][0]
        return float(x), float(y)

    ax, ay = coord(point_a)
    bx, by = coord(point_b)
    return math.hypot(ax - bx, ay - by)


@pytest.mark.parametrize("engine", ["brute_force", "nelder_mead"])
def test_dist_only_finds_feasible_assignment(monkeypatch, engine, fast_nelder_mead_env):
    coordinates = {"A": (0.0, 0.0), "B": ("a", "b")}
    variables = {"a": [0.0, False], "b": [0.0, False]}
    condition_code = [_dist_code("A", "B", 1)]

    extract_and_modify = reload_extract_and_modify(
        monkeypatch,
        COORDINATE_ENGINE=engine,
        **fast_nelder_mead_env,
    )
    _, result_variables, found = extract_and_modify(
        coordinates, condition_code, variables.copy()
    )

    assert found is True
    assert_feasible(condition_code, coordinates, result_variables)
    assert abs(_resolved_distance(coordinates, result_variables, "A", "B") - 1.0) < 0.05


@pytest.mark.parametrize("engine", ["brute_force", "nelder_mead"])
def test_dist_and_angle_finds_feasible_assignment(
    monkeypatch, engine, fast_nelder_mead_env
):
    coordinates = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": ("a", "b")}
    variables = {"a": [0.0, False], "b": [0.0, False]}
    condition_code = [
        _angle_code("B", "A", "C", 90),
        _dist_code("A", "C", 1),
    ]

    extract_and_modify = reload_extract_and_modify(
        monkeypatch,
        COORDINATE_ENGINE=engine,
        **fast_nelder_mead_env,
    )
    _, result_variables, found = extract_and_modify(
        coordinates, condition_code, variables.copy()
    )

    assert found is True
    assert_feasible(condition_code, coordinates, result_variables)


@pytest.mark.parametrize("engine", ["brute_force", "nelder_mead"])
def test_online_with_dist_finds_feasible_assignment(
    monkeypatch, engine, fast_nelder_mead_env
):
    coordinates = {"E": (0.0, 0.0), "F": (2.0, 0.0), "B": ("a", "b")}
    variables = {"a": [0.0, False], "b": [0.0, False]}
    condition_code = [
        _online_code("B", "E", "F"),
        _dist_code("E", "B", 1),
    ]

    extract_and_modify = reload_extract_and_modify(
        monkeypatch,
        COORDINATE_ENGINE=engine,
        **fast_nelder_mead_env,
    )
    _, result_variables, found = extract_and_modify(
        coordinates, condition_code, variables.copy()
    )

    assert found is True
    assert_feasible(condition_code, coordinates, result_variables)


@pytest.mark.parametrize("engine", ["brute_force", "nelder_mead"])
def test_zero_free_variables_succeeds_immediately(monkeypatch, engine):
    coordinates = {"A": (0.0, 0.0), "B": (1.0, 0.0)}
    variables = {}
    condition_code = [_dist_code("A", "B", 1)]

    extract_and_modify = reload_extract_and_modify(
        monkeypatch,
        COORDINATE_ENGINE=engine,
    )
    _, result_variables, found = extract_and_modify(
        coordinates, condition_code, variables
    )

    assert found is True
    assert result_variables == {}


@pytest.mark.parametrize("engine", ["brute_force", "nelder_mead"])
def test_infeasible_constraint_returns_not_found(
    monkeypatch, engine, fast_nelder_mead_env
):
    coordinates = {"A": (0.0, 0.0), "B": ("a", "b")}
    variables = {"a": [0.0, False], "b": [0.0, False]}
    condition_code = [
        _dist_code("A", "B", 1),
        _dist_code("A", "B", 3),
    ]

    extract_and_modify = reload_extract_and_modify(
        monkeypatch,
        COORDINATE_ENGINE=engine,
        **fast_nelder_mead_env,
    )
    _, _, found = extract_and_modify(coordinates, condition_code, variables.copy())

    assert found is False


def test_coordinate_engine_routes_to_nelder_mead(monkeypatch, fast_nelder_mead_env):
    coordinates = {"A": (0.0, 0.0), "B": ("a", "b")}
    variables = {"a": [0.0, False], "b": [0.0, False]}
    condition_code = [_dist_code("A", "B", 1)]

    extract_and_modify = reload_extract_and_modify(
        monkeypatch,
        COORDINATE_ENGINE="nelder_mead",
        **fast_nelder_mead_env,
    )
    _, result_variables, found = extract_and_modify(
        coordinates, condition_code, variables.copy()
    )

    assert found is True
    assert_feasible(condition_code, coordinates, result_variables)


def _resolved_point(coordinates, variables, point):
    x, y = coordinates[point]
    if isinstance(x, str):
        x = variables[x][0]
    if isinstance(y, str):
        y = variables[y][0]
    return float(x), float(y)


@pytest.mark.parametrize("engine", ["brute_force", "nelder_mead"])
def test_angle_bisector_rejects_overlapping_points(
    monkeypatch, engine, fast_nelder_mead_env
):
    coordinates = {
        "O": (0.0, 0.0),
        "A": (1.0, 0.0),
        "B": (-1.0, 0.0),
        "C": ("cx", "cy"),
        "D": ("dx", "dy"),
    }
    variables = {
        "cx": [0.0, False],
        "cy": [0.0, False],
        "dx": [0.0, False],
        "dy": [0.0, False],
    }
    condition_code = [
        _dist_code("O", "C", 1.0),
        _dist_code("O", "D", 1.0),
        (
            "angle_relation(variables,coordinates['A'],coordinates['O'],"
            "coordinates['D'],coordinates['D'],coordinates['O'],"
            "coordinates['C'],1.0,coordinates)"
        ),
    ]

    extract_and_modify = reload_extract_and_modify(
        monkeypatch,
        COORDINATE_ENGINE=engine,
        **fast_nelder_mead_env,
    )
    _, result_variables, found = extract_and_modify(
        coordinates, condition_code, variables.copy()
    )

    assert found is True
    assert_feasible(condition_code, coordinates, result_variables)

    cx, cy = _resolved_point(coordinates, result_variables, "C")
    ax, ay = _resolved_point(coordinates, result_variables, "A")
    bx, by = _resolved_point(coordinates, result_variables, "B")
    assert math.hypot(cx - ax, cy - ay) >= 0.1
    assert math.hypot(cx - bx, cy - by) >= 0.1


def test_coordinate_engine_defaults_to_brute_force(monkeypatch):
    coordinates = {"A": (0.0, 0.0), "B": ("a", "b")}
    variables = {"a": [0.0, False], "b": [0.0, False]}
    condition_code = [_dist_code("A", "B", 1)]

    extract_and_modify = reload_extract_and_modify(monkeypatch)
    _, result_variables, found = extract_and_modify(
        coordinates, condition_code, variables.copy()
    )

    assert found is True
    assert_feasible(condition_code, coordinates, result_variables)


def _circle_problem_5_condition_code():
    return [
        "arc_midpoint(variables,coordinates['C'],coordinates['A'],coordinates['B'],coordinates)",
        _angle_code("B", "A", "C", 35),
        _angle_code("A", "O", "B", 140),
    ]


def test_convert_coordinates_collapses_radius_expressions():
    from geo.Auxiliary_function import convert_coordinates

    points = {
        "O": [0, 0],
        "A": ["R", 0],
        "B": ["-R*cos(40*pi/180)", "R*sin(40*pi/180)"],
        "C": ["R*cos(70*pi/180)", "R*sin(70*pi/180)"],
    }
    coordinates, variables = convert_coordinates(points, radius=1.0)

    assert variables == {}
    assert coordinates["A"] == (1.0, 0.0)
    assert abs(coordinates["B"][0] - (-math.cos(40 * math.pi / 180))) < 1e-9
    assert abs(coordinates["B"][1] - math.sin(40 * math.pi / 180)) < 1e-9


@pytest.mark.parametrize("engine", ["brute_force", "nelder_mead"])
def test_circle_problem_5_radius_expressions_find_solution(
    monkeypatch, engine, fast_nelder_mead_env
):
    from geo.Auxiliary_function import convert_coordinates

    points = {
        "O": [0, 0],
        "A": ["R", 0],
        "B": ["-R*cos(40*pi/180)", "R*sin(40*pi/180)"],
        "C": ["R*cos(70*pi/180)", "R*sin(70*pi/180)"],
    }
    coordinates, variables = convert_coordinates(points, radius=None)
    condition_code = _circle_problem_5_condition_code()

    assert list(variables.keys()) == ["r"]

    extract_and_modify = reload_extract_and_modify(
        monkeypatch,
        COORDINATE_ENGINE=engine,
        **fast_nelder_mead_env,
    )
    _, result_variables, found = extract_and_modify(
        coordinates, condition_code, variables.copy()
    )

    assert found is True
    assert_feasible(condition_code, coordinates, result_variables)


def test_arc_midpoint_penalty_is_zero_when_satisfied():
    import math

    from geo.Auxiliary_function import convert_coordinates
    from geo.coordinate_engine_config import BOOLEAN_PENALTY
    from geo.nelder_mead_engine import _compute_penalty

    points = {
        "O": [0, 0],
        "A": ["R", 0],
        "B": ["-R*cos(40*pi/180)", "R*sin(40*pi/180)"],
        "C": ["R*cos(70*pi/180)", "R*sin(70*pi/180)"],
    }
    coordinates, variables = convert_coordinates(points, radius=1.0)
    condition_code = _circle_problem_5_condition_code()

    penalty = _compute_penalty(condition_code, coordinates, variables, BOOLEAN_PENALTY)
    assert penalty < 1e-6


def test_extract_on_circle_points_from_problem_5_text():
    from geo.Auxiliary_function import _extract_on_circle_points

    text = "已知点A、B、C在⊙O上，C为弧AB的中点.∠BAC=35°，∠AOB=140°"
    assert _extract_on_circle_points(text) == {"A", "B", "C"}


def test_extract_on_circle_points_skips_outside_point():
    from geo.Auxiliary_function import _extract_on_circle_points

    text = "点A是⊙O外一点，AB与⊙O相切于点B"
    assert "A" not in _extract_on_circle_points(text)
    assert "B" not in _extract_on_circle_points(text)


def test_add_on_circle_dist_constraints_for_problem_5():
    from geo.Auxiliary_function import add_on_circle_dist_constraints

    text = "已知点A、B、C在⊙O上，C为弧AB的中点.∠BAC=35°，∠AOB=140°"
    coordinates = {
        "O": (0.0, 0.0),
        "A": ("r", 0.0),
        "B": ("-r*cos(40*pi/180)", "r*sin(40*pi/180)"),
        "C": ("r*cos(70*pi/180)", "r*sin(70*pi/180)"),
    }
    conditions = [
        ["arc_midpoint", ["C", "A", "B"]],
        ["angle", ["B", "A", "C", 35.0]],
    ]

    result = add_on_circle_dist_constraints(text, coordinates, conditions, radius=1.0)

    dist_points = {params[1] for name, params in result if name == "dist"}
    assert dist_points == {"A", "B", "C"}
    assert all(params[2] == 1.0 for name, params in result if name == "dist")


def test_check_coordinates_distinct_uses_euclidean_distance():
    from geo.Auxiliary_function import convert_coordinates
    from geo.Geometric_function import check_coordinates_distinct

    coordinates, variables = convert_coordinates(
        {
            "F": ["f_x", "f_y"],
            "G": ["g_x", "g_y"],
        }
    )
    variables.update(
        {
            "f_x": [0.24, True],
            "f_y": [0.32, True],
            "g_x": [0.18, True],
            "g_y": [0.24, True],
        }
    )

    assert check_coordinates_distinct(coordinates, variables) is True


def test_fix_online_inside_coordinates_swaps_wrong_axis():
    from geo.Auxiliary_function import convert_coordinates, fix_online_inside_coordinates

    coordinates, variables = convert_coordinates(
        {
            "A": [0, 0],
            "B": [0, 0.5],
            "C": [0.5, 0.5],
            "D": [0.5, 0],
            "E": [0.5, "e"],
        }
    )
    conditions = [["online_inside", ["E", "B", "C"]]]

    fix_online_inside_coordinates(coordinates, variables, conditions)

    assert coordinates["E"] == ("e", 0.5)


def test_quadrangle_problem_0_solves_after_online_inside_fix(
    monkeypatch, fast_nelder_mead_env
):
    from geo.Auxiliary_function import convert_conditions, convert_coordinates

    text = (
        "在正方形ABCD中,点E在BC上,点F在AE上，点G在AE上，"
        "BF⊥AE,DG⊥AE，BC=0.5,DG=0.4"
    )
    result = {
        "coordinates": {
            "A": [0, 0],
            "B": [0, 0.5],
            "C": [0.5, 0.5],
            "D": [0.5, 0],
            "E": [0.5, "e"],
            "F": ["f_x", "f_y"],
            "G": ["g_x", "g_y"],
        },
        "conditions": {
            "c1": ["online_inside", "E", "B", "C"],
            "c2": ["online_inside", "F", "A", "E"],
            "c3": ["online_inside", "G", "A", "E"],
            "c4": ["ortho", "B", "F", "A", "E"],
            "c5": ["ortho", "D", "G", "A", "E"],
            "c6": ["dist", "D", "G", 0.4],
        },
    }

    coordinates, variables = convert_coordinates(result["coordinates"])
    coordinates, variables, _, condition_code = convert_conditions(
        text, variables, coordinates, result["conditions"]
    )
    assert coordinates["E"] == ("e", 0.5)

    extract_and_modify = reload_extract_and_modify(
        monkeypatch,
        COORDINATE_ENGINE="nelder_mead",
        **fast_nelder_mead_env,
    )
    _, result_variables, found = extract_and_modify(
        coordinates, condition_code, variables.copy()
    )

    assert found is True
    assert_feasible(condition_code, coordinates, result_variables)
    assert abs(_resolved_distance(coordinates, result_variables, "D", "G") - 0.4) < 0.05


def test_add_on_circle_dist_constraints_does_not_duplicate():
    from geo.Auxiliary_function import add_on_circle_dist_constraints

    text = "点A，B，C在⊙O上"
    coordinates = {"O": (0.0, 0.0), "A": ("r", 0.0), "B": (0.0, "r"), "C": ("a", "b")}
    conditions = [["dist", ["O", "A", 1.0]]]

    result = add_on_circle_dist_constraints(text, coordinates, conditions, radius=1.0)
    dist_for_a = [params for name, params in result if name == "dist" and params[1] == "A"]

    assert len(dist_for_a) == 1


# ── Soft boolean penalty tests ──────────────────────────────────────────────


def _make_code(func_name, *point_names):
    coords = ",".join(f"coordinates['{p}']" for p in point_names)
    return f"{func_name}(variables,{coords},coordinates)"


def _make_coords(**pts):
    return {name: (float(x), float(y)) for name, (x, y) in pts.items()}


class TestIsPointInTriangleSoftResidual:
    """Soft residual for is_point_in_triangle should be 0 inside, smooth outside."""

    def test_zero_when_inside(self):
        from geo.nelder_mead_engine import _constraint_contribution

        coords = _make_coords(A=(0, 0), B=(2, 0), C=(1, 2), P=(1, 0.5))
        code = _make_code("is_point_in_triangle", "A", "B", "C", "P")
        assert _constraint_contribution(code, {}, coords, 1000) == 0.0

    def test_positive_when_outside(self):
        from geo.nelder_mead_engine import _constraint_contribution

        coords = _make_coords(A=(0, 0), B=(2, 0), C=(1, 2), P=(5, 5))
        code = _make_code("is_point_in_triangle", "A", "B", "C", "P")
        residual = _constraint_contribution(code, {}, coords, 1000)
        assert 0 < residual < 1000  # smooth, not BOOLEAN_PENALTY

    def test_smooth_gradient_outside(self):
        """Penalty should increase as P moves further outside."""
        from geo.nelder_mead_engine import _constraint_contribution

        code = _make_code("is_point_in_triangle", "A", "B", "C", "P")
        coords_near = _make_coords(A=(0, 0), B=(2, 0), C=(1, 2), P=(2.1, 0))
        coords_far = _make_coords(A=(0, 0), B=(2, 0), C=(1, 2), P=(4.0, 0))
        r_near = _constraint_contribution(code, {}, coords_near, 1000)
        r_far = _constraint_contribution(code, {}, coords_far, 1000)
        assert r_far > r_near


class TestIsPointOutTriangleSoftResidual:
    """Soft residual for is_point_out_triangle should be 0 outside, positive inside."""

    def test_zero_when_outside(self):
        from geo.nelder_mead_engine import _constraint_contribution

        coords = _make_coords(A=(0, 0), B=(2, 0), C=(1, 2), P=(5, 5))
        code = _make_code("is_point_out_triangle", "A", "B", "C", "P")
        assert _constraint_contribution(code, {}, coords, 1000) == 0.0

    def test_positive_when_inside(self):
        from geo.nelder_mead_engine import _constraint_contribution

        coords = _make_coords(A=(0, 0), B=(2, 0), C=(1, 2), P=(1, 0.5))
        code = _make_code("is_point_out_triangle", "A", "B", "C", "P")
        residual = _constraint_contribution(code, {}, coords, 1000)
        assert 0 < residual < 1000  # smooth, not BOOLEAN_PENALTY


class TestIsAcuteTriangleSoftResidual:
    """Soft residual for is_acute_triangle should be 0 when acute, positive when obtuse."""

    def test_zero_for_acute(self):
        from geo.nelder_mead_engine import _constraint_contribution

        # Equilateral triangle: all angles 60°, definitely acute
        coords = _make_coords(A=(0, 0), B=(1, 0), C=(0.5, 0.866))
        code = _make_code("is_acute_triangle", "A", "B", "C")
        assert _constraint_contribution(code, {}, coords, 1000) == 0.0

    def test_positive_for_obtuse(self):
        from geo.nelder_mead_engine import _constraint_contribution

        # Very obtuse triangle: angle at B > 90°
        coords = _make_coords(A=(0, 0), B=(0.1, 0), C=(1, 0.01))
        code = _make_code("is_acute_triangle", "A", "B", "C")
        residual = _constraint_contribution(code, {}, coords, 1000)
        assert 0 < residual < 1000  # smooth, not BOOLEAN_PENALTY


class TestAngleBisectorSoftResidual:
    """Soft residual for angle_bisector should be 0 when angles are equal."""

    def test_zero_when_bisecting(self):
        from geo.nelder_mead_engine import _constraint_contribution

        # AD bisects angle BAC: AB and AC are symmetric about AD
        coords = _make_coords(
            A=(0, 0), D=(1, 0), C=(1, 1), A_dup=(0, 0), B=(1, -1),
        )
        code = _make_code("angle_bisector", "A", "D", "C", "A_dup", "B")
        residual = _constraint_contribution(code, {}, coords, 1000)
        assert residual < 1e-6

    def test_positive_when_not_bisecting(self):
        from geo.nelder_mead_engine import _constraint_contribution

        coords = _make_coords(
            A=(0, 0), D=(1, 0.5), C=(1, 1), A_dup=(0, 0), B=(1, -1),
        )
        code = _make_code("angle_bisector", "A", "D", "C", "A_dup", "B")
        residual = _constraint_contribution(code, {}, coords, 1000)
        assert 0 < residual < 1000  # smooth, not BOOLEAN_PENALTY


class TestOnlineInsideSoftResidual:
    """online_inside should use smooth penalty instead of BOOLEAN_PENALTY."""

    def test_small_penalty_when_slightly_outside_segment(self):
        from geo.nelder_mead_engine import _constraint_contribution

        # B is slightly past F on the line EF
        coords = _make_coords(E=(0, 0), F=(1, 0), B=(1.1, 0))
        code = _make_code("online_inside", "B", "E", "F")
        residual = _constraint_contribution(code, {}, coords, 1000)
        assert 0 < residual < 10  # Smooth, not BOOLEAN_PENALTY (1000)


class TestArcMidpointSoftResidual:
    """arc_midpoint soft residual should be 0 when AB == AC."""

    def test_zero_when_equidistant(self):
        from geo.nelder_mead_engine import _constraint_contribution

        coords = _make_coords(A=(1, 0), B=(0, 1), C=(0, -1))
        code = _make_code("arc_midpoint", "A", "B", "C")
        residual = _constraint_contribution(code, {}, coords, 1000)
        assert residual < 1e-6
