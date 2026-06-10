import os

COORDINATE_ENGINE = os.getenv("COORDINATE_ENGINE", "brute_force").lower()
COORDINATE_ENGINE_TIMEOUT = int(os.getenv("COORDINATE_ENGINE_TIMEOUT", "600"))
COORDINATE_LOW = float(os.getenv("COORDINATE_LOW", "-2"))
COORDINATE_HIGH = float(os.getenv("COORDINATE_HIGH", "2"))
NELDER_MEAD_RESTARTS = int(os.getenv("NELDER_MEAD_RESTARTS", "8"))
NELDER_MEAD_MAXITER = int(os.getenv("NELDER_MEAD_MAXITER", "2000"))
NELDER_MEAD_RESTARTS_PER_VAR = int(os.getenv("NELDER_MEAD_RESTARTS_PER_VAR", "4"))
NELDER_MEAD_RESTARTS_CAP = int(os.getenv("NELDER_MEAD_RESTARTS_CAP", "64"))


def effective_nelder_mead_restarts(n_vars):
    """Scale restarts up for high-dimensional constraint systems."""
    if n_vars <= 4:
        return NELDER_MEAD_RESTARTS
    scaled = max(NELDER_MEAD_RESTARTS, n_vars * NELDER_MEAD_RESTARTS_PER_VAR)
    return min(scaled, NELDER_MEAD_RESTARTS_CAP)


BOOLEAN_PENALTY = float(os.getenv("BOOLEAN_PENALTY", "1000"))
EARLY_EXIT_PENALTY = float(os.getenv("EARLY_EXIT_PENALTY", "1e-6"))

CONTINUOUS_CONSTRAINTS = frozenset(
    {
        "dist",
        "angle",
        "equal_line",
        "angle_relation",
        "line_ratio",
        "ortho",
        "online",
        "online_inside",
        "online_extension",
        "parallel",
        "is_point_in_triangle",
        "is_point_out_triangle",
        "is_acute_triangle",
        "angle_bisector",
        "arc_midpoint",
        "midpoint",
    }
)
