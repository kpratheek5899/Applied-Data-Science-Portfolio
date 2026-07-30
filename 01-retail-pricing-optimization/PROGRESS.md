# Nova Retail — Progress Tracker

Living checklist for `01-retail-pricing-optimization/`. Updated and committed after each completed step.

## Phase 1 — Data Simulator ✅ DONE

- [x] Build simulator v3 (price, promotion, marketing, digital funnel, inventory, ops capacity)
- [x] Validation notebook 01 (elasticity recovery proof: naive → FE → FE+controls)
- [x] Generate & validate full dataset (2.92M rows / 56 cols, 2024-01-01 to 2025-12-31)
- [x] Commit checkpoint (`a6442f6`)

## Phase 2 — Econometric Elasticity Recovery ✅ DONE

- [x] Naive pooled OLS baseline (log units ~ log price)
- [x] SKU fixed-effects model
- [x] Fixed effects + full controls model (promo depth, marketing spend, event/channel/day/region)
- [x] Diagnostics: standard errors, confidence intervals, residual analysis
- [x] Implement in `src/modeling.py`
- [x] Rewrite `notebooks/03_elasticity_modeling.ipynb` with real outputs

Result: full model (FE + controls) recovers true elasticity within 0.2-2.8% for all four product types (R² 0.67-0.89). See notebook 03 for the full naive → FE → FE+controls comparison, diagnostics, and the small-cluster CI caveat.

## Phase 3 — Bayesian Demand Modeling ✅ DONE

- [x] PyMC Bayesian hierarchical linear regression (NUTS, non-centered parameterization)
- [x] Posterior distributions and credible intervals (SKU- and product-type-level)
- [x] Hierarchical pooling (SKU-level elasticity toward product-type-level)
- [x] Implement in `src/modeling.py` (`prepare_bayesian_data`, `build_hierarchical_model`, `fit_bayesian_model`, `summarize_bayesian_elasticity`)
- [x] `notebooks/04_bayesian_elasticity_modeling.ipynb` with real executed outputs

Result: recovers true elasticity within ~2-11% per product type (Commodity 1.9%, Promo Sensitive 1.1%, Seasonal 4.1%, Premium 11.0%), clean convergence (r-hat 1.00, ess_bulk 3000+, zero divergences). Two real bugs found and fixed along the way (both documented in the notebook): (1) date-SKU aggregation destroyed the row-level price variation needed for identification, fixed via row-level random subsampling; (2) a promotion-depth coefficient shared across product types misattributed type-specific promo lift into price elasticity, fixed by letting it vary by product type. Fitting takes ~15 min (no C compiler on this machine for PyTensor) — the notebook loads a cached posterior from `data/processed/phase3_idata.pkl` rather than refitting live.

## Phase 4 — Price Optimization ✅ DONE

- [x] `src/optimization.py` (grid search over the constant-elasticity demand curve + a cvxpy geometric-programming solver for revenue, used to cross-validate the grid search)
- [x] Objectives: max profit, max revenue, protect inventory
- [x] Constraints: minimum margin, inventory, price bounds, max %-change
- [x] `tests/test_optimizer.py` (13 tests: bounds, margin, inventory feasibility, internal consistency, objective divergence, elasticity sensitivity, inventory-protection vs profit ordering, overstock/shortage scenarios, grid-vs-GP cross-check) — all passing

Result: revenue(price) is a pure monomial under constant elasticity (clean GP problem), but profit(price) is a monomial minus a monomial (a signomial, not DCP/DGP-representable) — so `optimize_price` uses grid search as one general mechanism for all three objectives (which also produces the exact data Phase 5's price-response chart needs), and `optimize_price_gp` demonstrates/validates the true GP solution for revenue specifically. Validated on both synthetic elastic/inelastic cases and a real SKU from the dataset.

## Phase 5 — Streamlit Decision-Support App ✅ DONE

Revised scope (see plan `crystalline-splashing-lecun.md`): presets are prefills only, closed-loop Decision Replay, Bayesian risk-aware optimization, custom date-range selection. Must run as a self-contained, deployable simulation (Streamlit Community Cloud) for portfolio/LinkedIn demo purposes — no live dependency on the 970MB dataset, Bayesian refitting, or `pymc`/`cvxpy` at runtime.

### 5a — Data layer ✅ DONE

- [x] `scripts/build_app_data.py` (offline precompute, not part of the live app)
- [x] `data/app/sku_master.csv` (50 rows, no ground-truth columns)
- [x] `data/app/daily_sku_timeseries.csv` (36,550 rows, full 2024-01-01 to 2025-12-31 coverage; carries `true_price_elasticity` as the one deliberate, clearly-scoped ground-truth column for Decision Replay's outcome realization only)
- [x] `data/app/posterior_samples.csv` (15,000 rows, 300 thinned draws/SKU)
- [x] `src/data_loader.py`
- [x] `tests/test_data_loader.py` (12 tests: no nulls, full date coverage, no ground-truth leakage into sku_master, inventory identity, elasticity fallback logic, posterior draw counts) — all passing

### 5b — Scenario Explorer (point-estimate) ✅ DONE

- [x] `src/demand_model.py`
- [x] `src/scenario_engine.py` (presets as prefills only, single date + custom date range)
- [x] `src/explanations.py`
- [x] `src/metrics.py`
- [x] `app/streamlit_app.py` landing page + `app/pages/1_Scenario_Explorer.py`
- [x] `optimize_price_multi_day` / `multi_day_price_response_curve` added to `src/optimization.py` (existing single-day functions/tests untouched)
- [x] `tests/test_scenarios.py` (11 tests) + `tests/test_app_pages.py` (5 tests, via Streamlit's own `AppTest` headless runner)

Result: all 6 presets and manual/date-range mode run end-to-end through one shared code path (`Scenario` → `build_demand_context` → `recommend_price`), verified with `AppTest` (not just unit tests of internal functions) — this caught and fixed a real pandas dtype bug (assigning formatted currency strings into a float64 column) that unit tests alone wouldn't have surfaced. 41/41 tests passing across the full suite.

### 5c — Bayesian risk-aware optimization ✅ DONE

- [x] `optimize_price_bayesian` in `src/optimization.py` (posterior draws + risk-aversion slider, day-array signature covers single-day and multi-day uniformly)
- [x] `demand_model.recommend_price_bayesian` bridge
- [x] Fan/percentile chart (p10/p50/p90) of outcomes at the recommended price
- [x] Inventory-override control added to the UI (was a real gap — without it there was no way to construct a genuinely inventory-constrained `maximize_profit` scenario through the app, since real historical inventory is usually loose relative to profit-optimal prices)
- [x] `tests/test_optimizer.py` +6 tests (zero-risk-aversion ≈ point estimate, higher risk-aversion lowers stockout probability, protect_inventory ≥ profit, percentiles ordered, rejects non-negative elasticity draws, multi-day sums correctly)
- [x] `tests/test_app_pages.py` +2 tests (risk-aversion slider changes the recommendation, disabling Bayesian mode falls back cleanly)

Result: all three objectives now use the mean across ~300 posterior draws instead of a point estimate; `protect_inventory` keeps 5b's deterministic hard floor (computed from the posterior mean), while `maximize_profit`/`maximize_revenue` subtract `risk_aversion × P(stockout) × (target's own range on the grid)` — the range normalization keeps the 0–1 slider meaningfully comparable across differently-scaled SKUs. 49/49 tests passing across the full suite. One real finding while testing: risk_aversion only has room to act when the *unconstrained* profit-optimal price both sits inside the allowed price-change band and carries genuine stockout risk there — a real economic property (elastic goods pushed to a binding price-change ceiling already get whatever inventory protection higher pricing provides "for free"), not a bug, but it made the naive test scenario a poor showcase, which is what surfaced the missing inventory-override control.

### 5d — Closed-loop Decision Replay ✅ DONE

- [x] `src/replay_engine.py` (`realize_true_outcome` — the only function in the codebase allowed to read `true_price_elasticity` — and `run_closed_loop_replay`)
- [x] `app/pages/2_Decision_Replay.py` (SKU/date/window picker, step-through-days control, parallel cumulative profit + inventory trajectories, explicit UI note on why the closed loop only works because Nova Retail is synthetic)
- [x] Landing page nav link added
- [x] `tests/test_replay_engine.py` (11 tests) + `tests/test_app_pages.py` +3 tests

Result: two trajectories per SKU/window, both starting from the same actual Day-1 inventory — "actual" is a pure passthrough of history, "optimizer" recommends each day's price from the estimated model (same decision path as Scenario Explorer) and realizes the outcome via the true simulator model, with Day-(N+1) starting inventory coming from Day-N's own optimizer decision, not the next fixed historical row. 63/63 tests passing across the full suite. Two real bugs found and fixed while building, not just tuned away: (1) an initial replenishment rule added back the *absolute amount* the real network consumed each day, which let the optimizer trajectory's inventory drift upward without bound whenever its policy sold less than history (e.g. under `protect_inventory`) — fixed to replenish toward the same bounded *target level* the real network was refilled to, matching how `src/data_generation.py`'s simulator actually replenishes (target-based, not consumption-based); (2) the window-length and start-date controls could combine to overflow past the dataset's last date — an initial fix just caught this after the fact with `st.error`/`st.stop()`, but the real fix (per explicit user feedback) makes it structurally impossible: the start-date picker's own `max_value` is derived from whichever window length is currently selected (read before the date widget in script order, independent of visual column layout), with the persisted value proactively clamped in `st.session_state` when the window grows — confirmed via `AppTest` that widening the window mid-session clamps the date silently with zero errors and zero exceptions, rather than requiring the user to notice and fix an invalid combination themselves.

### 5e — Polish + deployment ✅ DONE

- [x] Landing page copy (retrospective/counterfactual framing, synthetic-data disclosure) — written in 5b, confirmed still accurate
- [x] `01-retail-pricing-optimization/requirements.txt` — `streamlit`, `pandas`, `numpy`, `matplotlib` only (exact pins matching what's tested); confirmed by grepping every top-level import on the live app's code path, then proving it for real by running all 3 pages via `AppTest` inside a **fresh, isolated venv containing only those 4 packages** — zero exceptions, no `pymc`/`cvxpy`/`statsmodels`/`arviz` needed
- [x] Sidebar branding + GitHub source link added to all 3 pages (portfolio/demo polish)
- [x] `DEPLOYMENT.md` — step-by-step Streamlit Community Cloud instructions, including an explicit warning about the repo-root `requirements.txt` (a separate messy UTF-16 pip-freeze dump covering all 3 portfolio projects) so the deploy doesn't accidentally pick that up instead

Result: 63/63 tests passing. The app is genuinely self-contained — confirmed empirically, not just by inspection, that it runs with only 4 lightweight packages and the small committed `data/app/` CSVs (~4.7MB), no dependency on the 970MB simulated dataset or any model refitting. Deployment itself (connecting the repo on Streamlit Community Cloud) is a handoff — needs the user's account — documented in `DEPLOYMENT.md`.

### 5f — Adaptive Learning demo (Thompson Sampling) ✅ DONE

Supersedes the simpler Modifications 2/3 from the original follow-up spec with a full closed-form Bayesian bandit: three pricing policies (frozen "static" baseline, "Thompson Sampling", and a labeled-upper-bound "Oracle") run side by side from the same deliberately weak starting belief, learning purely from observing the outcomes of their own pricing decisions.

- [x] `src/bayesian_learning.py` — Normal-Inverse-Gamma conjugate posterior over elasticity, closed-form sequential update, Thompson sample draw, Student-t credible interval. No PyMC/MCMC (must run interactively).
- [x] `src/adaptive_simulation.py` — `run_adaptive_simulation` (all 3 variants, day-by-day, regret vs. Oracle's own profit).
- [x] `apply_daily_replenishment` extracted from `replay_engine.py` into a shared helper (pure refactor — existing tests confirmed unchanged behavior) so Decision Replay and the 3 Adaptive Learning trajectories all reuse the same already-debugged bounded target-level replenishment logic.
- [x] `app/pages/3_Adaptive_Learning.py` — SKU/window/objective/"starting confidence" controls, cached simulation run, day-scrub slider, all 4 required visualizations (cumulative regret race, posterior-narrowing band, explore/exploit-annotated price trajectory, running CI-width readout).
- [x] `tests/test_adaptive_learning.py` (16 tests) + 4 new `AppTest` cases in `tests/test_app_pages.py`. 83/83 tests passing across the full suite.
- [x] `requirements.txt` +`scipy` (needed for the Student-t credible interval) — re-verified the whole app still runs with zero exceptions in a fresh isolated venv containing only the 5 listed packages.

**Three real bugs found and fixed while building this, not just tuned away** — each one changed the actual design, not just a parameter:
1. **Exponential price runaway.** The price ceiling was computed relative to each day's own rolling (price, units) anchor; once a day's price hit that ceiling, the next day's ceiling was computed *from that price*, compounding into a runaway that doubled the price every day. Fixed by anchoring price bounds to the fixed window-*starting* price instead — the demand-curve anchor still rolls day to day (correct), only the bound doesn't.
2. **Statistically unstable belief.** The first version modeled `log_units ~ alpha + beta*log_price` (2 free parameters). Once the pricing policy converges toward a narrow price range, that regression can't separate alpha from beta when its one predictor barely varies — a single noisy update pushed the fitted elasticity to a nonsensical *positive* value within days. Fixed by regressing log-*differences from the anchor* instead (`dlog_units ~ beta*dlog_price`, one parameter, no intercept needed) — matches how every other demand curve in this app already works, and is far better identified. Verified against synthetic data: matches closed-form OLS almost exactly with a weak prior.
3. **No visible learning.** Two compounding calibration issues: (a) seeding the prior mean from Phase 2's `elasticity_phase2` gave "static" an already-accurate starting belief, leaving Thompson Sampling nothing to visibly out-learn; (b) the initial generic guess (-1.0, "unit elastic") sits almost exactly at the singularity of the closed-form profit-max formula (`cost*e/(e+1)` diverges as `e -> -1`), so every belief near it saturates the price ceiling regardless of the true value. Fixed by using a fixed, genuinely uninformed prior mean away from that singularity (-1.5) with a wider default price band (needed anyway: this dataset's actual historical prices run 40-300%+ below their theoretical profit-max, verified across the full SKU set) — confirmed across multiple SKUs and 20 seeds that Thompson Sampling now cuts regret roughly in half vs. static and reaches 95-98% of Oracle's profit, improving as the window grows.

**Post-implementation audit against the memo** caught three small but real gaps, all fixed and re-verified (83/83 tests): (1) chart 1 was regret-only despite being titled a "profit / regret race chart" — restructured into a 2-panel figure (cumulative profit above cumulative regret, sharing an x-axis, no dual-axis); (2) Oracle's framing was a loose paraphrase — tightened to the memo's own phrase, "if we had known the truth all along," in the intro, the chart legend, and a new caption near the summary metrics; (3) the running confidence readout showed only an abstract CI-width number — changed to show the actual interval bounds (e.g. `[-2.8, -0.3] → [-1.9, -1.5]`) as the headline value, matching the memo's own example format.

**Phase 5 (all sub-phases 5a-5f) is now complete.**

## Outstanding / Low Priority

- [ ] Clean up root `requirements.txt` (currently a raw UTF-16 pip-freeze dump)
- [ ] Decide fate of `notebooks/02_simulator_validation.ipynb` (old scratch notebook)
- [ ] Delete or keep `notebooks/_build_notebook_01.py` (one-off notebook-builder helper, safe to delete)
