Status: ready-for-agent

# PRD: Configurable Nelder-Mead Coordinate Engine

## Problem Statement

MagicGeo performs coordinate assignment by exhaustively searching a fixed grid (`[-2, 2]` in steps of `0.01`) over all free variables. This brute-force approach is predictable but scales poorly as the number of unknowns grows — runtime explodes combinatorially and problems with solutions outside the first feasible hit may produce degenerate layouts. Users have no way to choose a faster continuous search strategy, and the README describes a "solver" while the implementation is grid backtracking.

## Solution

Add a configurable **coordinate engine** selectable at runtime via environment variables. The default remains brute force. An optional **Nelder-Mead engine** performs multi-restart continuous optimization within the same **search region**, guided by a **penalty function** (squared **constraint residuals** for **continuous constraints** plus a **boolean penalty** for **discrete constraints**). Both engines share the same **feasible assignment** criterion: success requires every constraint to pass the existing boolean check layer. No automatic fallback between engines — one engine per run.

## User Stories

1. As a MagicGeo operator, I want to select the coordinate engine via configuration, so that I can experiment with Nelder-Mead without changing code.
2. As a MagicGeo operator, I want brute force to remain the default coordinate engine, so that existing behaviour is preserved.
3. As a MagicGeo operator, I want to set `COORDINATE_ENGINE=nelder_mead` to use the Nelder-Mead engine, so that I can attempt faster coordinate assignment on suitable problems.
4. As a MagicGeo operator, I want both engines to use the same search region (`[-2, 2]` per free variable by default), so that results are comparable.
5. As a MagicGeo operator, I want to configure the search region bounds via environment variables, so that I can widen or narrow the assignment space without code changes.
6. As a MagicGeo operator, I want a shared engine timeout for both coordinate engines, so that runaway assignment does not block the pipeline indefinitely.
7. As a MagicGeo operator, I want Nelder-Mead to run multiple random restarts within the search region, so that local minima do not prevent finding a feasible assignment.
8. As a MagicGeo operator, I want to configure the number of Nelder-Mead restarts, so that I can trade speed against robustness.
9. As a MagicGeo operator, I want to configure the maximum iterations per Nelder-Mead restart, so that a single bad start cannot consume the entire timeout budget unchecked.
10. As a MagicGeo operator, I want Nelder-Mead to keep searching after the first feasible assignment, so that better layouts can be found among multiple feasible hits.
11. As a MagicGeo operator, I want the Nelder-Mead engine to return the feasible assignment with the lowest penalty among all restarts, so that diagram quality is not left to whichever restart finishes first.
12. As a MagicGeo operator, I want success to mean the same thing for both engines (all constraints pass the boolean check layer), so that I can compare engines fairly.
13. As a MagicGeo operator, I want discrete constraints to contribute a large configurable penalty during Nelder-Mead search, so that the optimizer does not sacrifice boolean correctness for small numeric gains.
14. As a MagicGeo operator, I want continuous constraints (`dist`, `angle`, `equal_line`, `angle_relation`, `line_ratio`, `ortho`, `online`, `online_inside`, `online_extension`, `parallel`) to guide Nelder-Mead via squared-error residuals, so that the optimizer has gradient-friendly signal on common problem types.
15. As a MagicGeo operator, I want discrete constraints (`is_point_in_triangle`, `is_point_out_triangle`, `is_acute_triangle`, `arc_midpoint`, and others outside the continuous set) to use boolean penalty only, so that v1 does not require residual reformulation of every predicate.
16. As a MagicGeo operator, I want the boolean constraint check layer in the geometric module to remain unchanged, so that both engines share one source of truth for constraint satisfaction.
17. As a MagicGeo operator, I want residual computation to live with the Nelder-Mead engine, so that optimization-specific logic does not pollute shared constraint primitives.
18. As a MagicGeo operator, I want the coordinate assignment entry point to dispatch to the configured engine transparently, so that the text-to-diagram pipeline requires no changes beyond timeout configuration.
19. As a MagicGeo operator, I want `scipy` added as a project dependency, so that Nelder-Mead is available in the standard install path.
20. As a MagicGeo operator, I want new configuration variables documented in the example environment file and README, so that I can discover and tune the feature.
21. As a MagicGeo operator, I want an ADR recording why two coordinate engines coexist and how Nelder-Mead success is defined, so that future contributors understand the design without re-litigating it.
22. As a MagicGeo operator, I want domain terms for coordinate assignment, coordinate engine, penalty function, and feasible assignment captured in the project glossary, so that documentation and issues use consistent vocabulary.
23. As a developer maintaining MagicGeo, I want synthetic constraint fixtures testable without calling the LLM, so that coordinate engine behaviour can be verified in CI.
24. As a developer maintaining MagicGeo, I want tests to assert feasible assignment outcomes through the public coordinate assignment entry point, so that engine internals can evolve without brittle tests.
25. As a developer maintaining MagicGeo, I want tests covering both coordinate engines on the same fixtures where feasible, so that behavioural parity on the success criterion is enforced.
26. As a developer maintaining MagicGeo, I want tests for the zero-variable edge case (no free variables), so that both engines short-circuit correctly.
27. As a developer maintaining MagicGeo, I want tests for single-constraint problems (`dist`, `angle`, `online`), so that continuous residual dispatch is validated end-to-end.
28. As a developer maintaining MagicGeo, I want tests for multi-constraint combinations, so that penalty function composition does not mask individual constraint failures.
29. As a developer maintaining MagicGeo, I want tests verifying that an infeasible constraint set returns `found=False`, so that failure modes are explicit.
30. As a developer maintaining MagicGeo, I want engine selection tests that confirm `COORDINATE_ENGINE` routing, so that misconfiguration does not silently fall through to the wrong engine.

## Implementation Decisions

### Coordinate engine architecture

- A thin **dispatcher** at the existing coordinate assignment entry point reads `COORDINATE_ENGINE` and delegates to either the existing brute-force backtracker or the new Nelder-Mead engine. The entry point signature and return shape are unchanged: `(coordinates, variables, found)`.
- Brute-force logic is renamed internally but behaviour-preserving: same grid `[-2, 2]` step `0.01`, same pruning via the boolean break check, same success/failure logging.
- Nelder-Mead logic lives in a dedicated engine module separate from the dispatcher and separate from the shared geometric constraint primitives.

### Configuration module

- Centralise all coordinate-engine environment variables in a single configuration module loaded at import time:
  - `COORDINATE_ENGINE` — `brute_force` (default) or `nelder_mead`
  - `COORDINATE_ENGINE_TIMEOUT` — wall-clock seconds (default `600`)
  - `COORDINATE_LOW` / `COORDINATE_HIGH` — search region per variable (default `-2` / `2`)
  - `NELDER_MEAD_RESTARTS` — random restart count (default `8`)
  - `NELDER_MEAD_MAXITER` — scipy max iterations per restart (default `2000`)
  - `BOOLEAN_PENALTY` — penalty per violated discrete constraint (default `1000`)
- The text-to-diagram pipeline reads `COORDINATE_ENGINE_TIMEOUT` for its multiprocessing worker join, replacing the hardcoded `600`.

### Nelder-Mead engine behaviour

- If there are zero free variables, return success immediately (same as brute force).
- For each restart: draw a uniform random initial assignment within `[COORDINATE_LOW, COORDINATE_HIGH]` per variable; run `scipy.optimize.minimize` with method `Nelder-Mead` and `maxiter=NELDER_MEAD_MAXITER`.
- Objective = sum of per-constraint contributions:
  - **Continuous constraints**: squared residual via constraint-type dispatch (parses the existing executable constraint code strings to extract function name and point references).
  - **Discrete constraints**: evaluate via existing `eval` path; add `BOOLEAN_PENALTY` when `is_calculate` and not satisfied.
  - Unparseable or unevaluable constraints: treat as violated (`BOOLEAN_PENALTY`).
- After each restart: evaluate penalty; if assignment is **feasible** (boolean break check passes) and penalty is lower than the best so far, record it. Continue all restarts until restart count exhausted or engine timeout reached.
- On success: write winning variable values back into the `variables` dict (same mutation pattern as brute force) and return `found=True`.

### Continuous residual dispatch (v1 scope)

| Constraint type | Residual |
|---|---|
| `dist` | `(actual_distance − target)²` |
| `angle` | `(computed_angle − expected)²` |
| `equal_line` | `(dist_AB² − dist_CD²)²` |
| `line_ratio` | `(dist_FG² − k² · dist_AB²)²` |
| `angle_relation` | `(∠ABC − ratio · ∠DEF)²` |
| `ortho` | `dot_product²` |
| `online` | `point_to_line_distance²` |
| `online_inside` | line distance² + boolean penalty if parametric position outside `(0, 1)` |
| `online_extension` | line distance² + boolean penalty if parametric position inside `[0, 1]` |
| `parallel` | `(slope_AB − slope_CD)²` with infinite-slope handling |

All other constraint types: boolean penalty only.

### Shared success criterion

- Neither engine uses penalty threshold alone for success. After optimization (or grid search), success is determined solely by whether `check_condition_break` returns false — identical to pre-change behaviour.

### Dependencies and documentation

- Add `scipy` to project dependencies.
- Update example environment file and README configuration table with all new variables.
- Record decision in ADR `0001` (Nelder-Mead coordinate engine alongside brute force).
- Update `CONTEXT.md` glossary with coordinate-assignment domain terms.

### Explicit non-decisions

- No automatic fallback from Nelder-Mead to brute force on failure.
- No change to LLM prompt flow, condition conversion, or PDF rendering.
- No change to boolean constraint primitives or their tolerances.

## Testing Decisions

### Test seams (confirmed)

Tests should hit the **highest seam that exercises real behaviour without the LLM or multiprocessing pipeline**:

**Primary seam — coordinate assignment entry point (`extract_and_modify`)**

- Input: hand-crafted `coordinates`, `variables`, and `condition_code` lists (the same executable constraint strings produced by condition conversion).
- Output: `(coordinates, variables, found)` tuple; verify `found` and, when `found=True`, that `check_condition_break` returns false on the returned state.
- Engine selection: set `COORDINATE_ENGINE` via environment variable before importing the dispatcher module (or use `monkeypatch` in pytest).
- This seam validates routing, feasible assignment, penalty-guided search, and variable mutation in one call.

**Not proposed as primary seams**

- Individual residual helper functions (implementation detail; covered indirectly through entry-point fixtures).
- scipy `minimize` internals.
- Full `process_geometry_task` end-to-end (requires API key; belongs in manual/benchmark runs, not CI unit tests).
- `convert_conditions` / LLM path (orthogonal to coordinate engine selection).

**Fixture strategy**

- Minimal fixtures with known feasible solutions in the search region:
  - `dist` only (e.g. `A=(0,0)`, `B=(a,b)`, `dist=1`)
  - `dist` + `angle`
  - `online` or `ortho` combined with a numeric constraint
  - Zero free variables (immediate success)
  - Deliberately infeasible constraint (expect `found=False`)
- Parametrize the same fixtures across `brute_force` and `nelder_mead` where both can reasonably succeed within timeout.

**What makes a good test**

- Assert external behaviour only: `found` flag, feasible assignment invariant, and optionally that assigned numeric values satisfy constraints within existing geometric tolerances.
- Do not assert specific optimizer iteration counts, internal penalty values, or which restart succeeded.
- Keep tests fast: low restart count and low maxiter via env overrides in test setup.

**Prior art**

- No existing test suite in the repo. This feature establishes the first coordinate-assignment tests. Follow standard pytest conventions; add `pytest` as a dev dependency if not present.

## Out of Scope

- Replacing brute force as the default engine.
- Automatic fallback from Nelder-Mead to brute force when Nelder-Mead fails.
- Penalty-threshold-only success criterion (decoupled from boolean break check).
- Continuous residuals for all discrete constraint types in v1 (`arc_midpoint`, triangle inside/outside predicates, etc.).
- Extending brute-force search bounds configuration (brute force keeps hardcoded `[-2, 2]` step `0.01` in v1).
- Performance benchmarking across full JSON datasets (circle, triangle, quadrangle) as a CI gate.
- Refactoring condition conversion or geometric constraint primitives.
- Seed-controlled Nelder-Mead for fully deterministic CI (acceptable to test feasible existence, not exact coordinates).

## Further Notes

- Implementation from the design session is largely complete: dispatcher, Nelder-Mead engine, configuration module, scipy dependency, ADR, glossary, and documentation updates.
- Test seams confirmed by maintainer. Remaining optional work: manual benchmark comparing engines on dataset JSON files.
- A smoke test (`dist=1`, two free variables) already passes on the Nelder-Mead engine outside pytest.
- Brute force may fail on some feasible problems that Nelder-Mead finds (e.g. target distance requiring coordinates near the search region boundary with coarse grid quantisation) — parametrized cross-engine tests should use fixtures both engines can satisfy, or mark engine-specific fixtures explicitly.
