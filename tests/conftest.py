import importlib

import pytest

from geo.Auxiliary_function import check_condition_break


def assert_feasible(condition_code, coordinates, variables):
    assert not check_condition_break(condition_code, coordinates, variables)


def reload_extract_and_modify(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))

    import geo.coordinate_engine_config as config
    import geo.nelder_mead_engine as nelder_mead_engine
    import geo.Kernel_function as kernel

    importlib.reload(config)
    importlib.reload(nelder_mead_engine)
    importlib.reload(kernel)
    return kernel.extract_and_modify


@pytest.fixture
def fast_nelder_mead_env():
    return {
        "NELDER_MEAD_RESTARTS": "6",
        "NELDER_MEAD_MAXITER": "800",
    }
