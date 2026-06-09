# MagicGeo

Training-free text-guided geometric diagram generation. An LLM translates problem text into symbolic coordinates and constraints; a coordinate engine then assigns numeric values to free variables so the diagram can be rendered.

## Language

**Coordinate assignment**:
The process of finding concrete numeric values for free coordinate variables so that all geometric constraints hold.
_Avoid_: Solving, optimization (when referring to this phase specifically)

**Constraint**:
A geometric relation that must hold between points (e.g. distance, angle, collinearity), expressed as an executable check against the current coordinate state.
_Avoid_: Equation, rule

**Penalty function**:
The scalar objective minimized by Nelder-Mead: sum of squared residuals for numeric constraints, plus a large penalty for each violated boolean constraint.
_Avoid_: Loss function, cost function

**Coordinate engine**:
The algorithm that performs coordinate assignment. Selected at runtime via `COORDINATE_ENGINE`; one engine per run, no automatic fallback.
_Avoid_: Solver (ambiguous with the LLM constraint formalism)

**Search region**:
The bounding box `[-2, 2]` per free variable within which coordinate assignment is attempted. Shared by both coordinate engines.
_Avoid_: Domain, bounds (unless referring to this specific region)

**Feasible assignment**:
A variable assignment where every remaining constraint passes `check_condition_break`. The success criterion for all coordinate engines.
_Avoid_: Optimal solution, converged

**Engine timeout**:
Wall-clock seconds (`COORDINATE_ENGINE_TIMEOUT`) after which coordinate assignment is aborted. Applies to both coordinate engines.
_Avoid_: Solver timeout

**Boolean penalty**:
Fixed penalty (`BOOLEAN_PENALTY`, default `1000`) added to the objective for each violated boolean constraint during Nelder-Mead search.
_Avoid_: Constraint weight

**Constraint residual**:
Squared error for a numeric constraint, used only during Nelder-Mead search. Computed by constraint-type dispatch separate from the boolean constraint checks in `Geometric_function.py`.
_Avoid_: Error term, slack

**Nelder-Mead engine**:
Coordinate engine implementation in `geo/nelder_mead_engine.py`. Multi-restart Nelder-Mead within the search region, guided by the penalty function.
_Avoid_: Optimizer, scipy solver

**Restart selection**:
Among all Nelder-Mead restarts within the engine timeout, keep the feasible assignment with the lowest penalty. Continue searching after the first feasible hit to allow better layouts.
_Avoid_: Early stopping

**Continuous constraint**:
A constraint whose violation is measured as a squared error during Nelder-Mead search (`dist`, `angle`, `equal_line`, `angle_relation`, `line_ratio`, `ortho`, `online`, `online_inside`, `online_extension`, `parallel`).

**Discrete constraint**:
A constraint evaluated only as pass/fail, contributing `BOOLEAN_PENALTY` when violated (`is_point_in_triangle`, `is_point_out_triangle`, `is_acute_triangle`, `arc_midpoint`, and others not in the continuous set).
