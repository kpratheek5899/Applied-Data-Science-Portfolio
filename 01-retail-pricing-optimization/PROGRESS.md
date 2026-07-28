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

## Phase 4 — Price Optimization ⏸️ NOT STARTED

- [ ] `src/optimization.py` with cvxpy
- [ ] Objectives: max profit, max revenue, protect inventory
- [ ] Constraints: minimum margin, inventory, price bounds

## Phase 5 — Streamlit Decision-Support App ⏸️ NOT STARTED

- [ ] `src/data_loader.py`
- [ ] `src/demand_model.py`
- [ ] `src/optimizer.py`
- [ ] `src/scenario_engine.py` (predefined scenarios: overstock, holiday surge, shortage, slow-mover, high/low elasticity)
- [ ] `src/explanations.py` (plain-English recommendation rationale)
- [ ] `src/metrics.py`
- [ ] `app.py` — user-defined scenario mode, objective selection, before/after comparison
- [ ] Price-response visualization
- [ ] Decision Replay mode
- [ ] `tests/test_optimizer.py`
- [ ] `tests/test_scenarios.py`
- [ ] Landing page framing (synthetic data disclosure, methodology summary)

## Outstanding / Low Priority

- [ ] Clean up root `requirements.txt` (currently a raw UTF-16 pip-freeze dump)
- [ ] Decide fate of `notebooks/02_simulator_validation.ipynb` (old scratch notebook)
- [ ] Delete or keep `notebooks/_build_notebook_01.py` (one-off notebook-builder helper, safe to delete)
