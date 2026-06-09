import numpy as np

from geo.Auxiliary_function import check_condition_break
from geo.coordinate_engine_config import COORDINATE_ENGINE
from geo.nelder_mead_engine import extract_and_modify_nelder_mead


def _brute_force(coordinates, condition_code, variables):
    key_list = list(variables.keys())
    flag = len(key_list)

    print(f"Brute force: {flag} variables")
    print(f"Keys: {key_list}")

    if flag == 0:
        return coordinates, variables, True

    low, up, step = -2, 2, 0.01
    search_values = np.around(np.arange(low, up + step, step), decimals=2)

    def backtrack(index):
        if index == flag:
            cond_break = check_condition_break(condition_code, coordinates, variables)
            return not cond_break

        current_key = key_list[index]

        for val in search_values:
            variables[current_key] = [val, True]

            if not check_condition_break(condition_code, coordinates, variables):
                if backtrack(index + 1):
                    return True

            variables[current_key][1] = False

        return False

    if backtrack(0):
        print("\nfound a feasible solution!\n")
        print(variables)
        print(coordinates)
        return coordinates, variables, True

    print("\ndidn't find a feasible solution!\n")
    return coordinates, variables, False


def extract_and_modify(coordinates, condition_code, variables):
    if COORDINATE_ENGINE == "nelder_mead":
        return extract_and_modify_nelder_mead(coordinates, condition_code, variables)
    return _brute_force(coordinates, condition_code, variables)
