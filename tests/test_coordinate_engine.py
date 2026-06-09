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
