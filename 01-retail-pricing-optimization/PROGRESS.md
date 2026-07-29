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

## Phase 5 — Streamlit Decision-Support App 🔄 IN PROGRESS

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

### 5c — Bayesian risk-aware optimization ⏸️ NOT STARTED

- [ ] `optimize_price_bayesian` in `src/optimization.py` (posterior draws + risk-aversion slider)
- [ ] Fan/percentile chart of outcomes at recommended price

### 5d — Closed-loop Decision Replay ⏸️ NOT STARTED

- [ ] `src/replay_engine.py` (`realize_true_outcome`, `run_closed_loop_replay`)
- [ ] `app/pages/2_Decision_Replay.py`
- [ ] `tests/test_replay_engine.py`

### 5e — Polish + deployment ⏸️ NOT STARTED

- [ ] Landing page copy (retrospective/counterfactual framing, synthetic-data disclosure)
- [ ] `01-retail-pricing-optimization/requirements.txt` (clean, deployment-scoped)
- [ ] Streamlit Community Cloud deployment instructions

## Outstanding / Low Priority

- [ ] Clean up root `requirements.txt` (currently a raw UTF-16 pip-freeze dump)
- [ ] Decide fate of `notebooks/02_simulator_validation.ipynb` (old scratch notebook)
- [ ] Delete or keep `notebooks/_build_notebook_01.py` (one-off notebook-builder helper, safe to delete)
