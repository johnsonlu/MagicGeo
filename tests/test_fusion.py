from geo.Auxiliary_function import convert_conditions, convert_coordinates
from geo.text_to_geometric import _build_fusion


def test_convert_conditions_routes_midpoint_to_derived_point():
    text = "在矩形ABCD中，O为AC的中点"
    result = {
        "coordinates": {
            "A": [0, 0],
            "B": ["a", 0],
            "C": ["a", "b"],
            "D": [0, "b"],
            "O": ["m", "n"],
        },
        "conditions": {
            "c4": ["midpoint", "O", "A", "C"],
        },
    }

    coordinates, variables = convert_coordinates(result["coordinates"])
    coordinates, variables, calc_conds, condition_code = convert_conditions(
        text, variables, coordinates, result["conditions"]
    )

    assert calc_conds == [["midpoint", ["O", "A", "C"]]]
    assert "O" in coordinates
    assert coordinates["O"][0].startswith("calc_midpoint(")
    assert "m" not in variables
    assert "n" not in variables
    assert not any("midpoint" in code for code in condition_code)


def test_build_fusion_resolves_calc_midpoint():
    coordinates = {
        "A": (0.0, 0.0),
        "B": (4.0, 0.0),
        "C": (4.0, 3.0),
        "O": ["calc_midpoint(variables,coordinates['A'],coordinates['C'],coordinates)"],
    }
    variables = {}

    fusion = _build_fusion(coordinates, variables, scale=1.0)

    assert fusion["A"] == (0.0, 0.0)
    assert fusion["C"] == (4.0, 3.0)
    assert fusion["O"] == (2.0, 1.5)


def test_build_fusion_resolves_calc_midpoint_with_variables():
    coordinates = {
        "A": (0.0, 0.0),
        "B": ("a", 0.0),
        "C": ("a", "b"),
        "O": ["calc_midpoint(variables,coordinates['A'],coordinates['C'],coordinates)"],
    }
    variables = {"a": [4.0, True], "b": [3.0, True]}

    fusion = _build_fusion(coordinates, variables, scale=2.0)

    assert fusion["O"] == (4.0, 3.0)
