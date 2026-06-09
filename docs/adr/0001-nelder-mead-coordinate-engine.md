# Nelder-Mead coordinate engine alongside brute-force grid search

MagicGeo assigns numeric values to free coordinate variables so geometric constraints hold. The original implementation exhaustively searches a `[-2, 2]` grid in steps of `0.01` via backtracking (`Kernel_function.py`). We added an optional Nelder-Mead coordinate engine (`geo/nelder_mead_engine.py`) selected by `COORDINATE_ENGINE`, with no automatic fallback between engines.

Nelder-Mead minimizes a penalty function: squared residuals for continuous constraints (`dist`, `angle`, `ortho`, `online`, etc.) plus a configurable fixed penalty per violated discrete constraint. Success is still defined by `check_condition_break` — the same feasible-assignment criterion as brute force — so both engines agree on what "solved" means. Multi-restart random initialisation within the shared search region mitigates Nelder-Mead's local-minima sensitivity; all restarts run within `COORDINATE_ENGINE_TIMEOUT`, keeping the best feasible assignment with the lowest penalty.

**Considered options:** Replace brute force entirely (rejected — grid search is the proven default across datasets); Nelder-Mead with automatic fallback (rejected — obscures which engine succeeded and doubles runtime); penalty-threshold-only success (rejected — would diverge from existing per-constraint tolerances in `Geometric_function.py`).

**Consequences:** Adds `scipy` as a dependency. Residual computation lives in the Nelder-Mead module, not in `Geometric_function.py`, so boolean constraint checks remain shared unchanged.
