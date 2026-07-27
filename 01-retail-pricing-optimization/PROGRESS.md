# Nova Retail — Progress Tracker

Living checklist for `01-retail-pricing-optimization/`. Updated and committed after each completed step.

## Phase 1 — Data Simulator ✅ DONE

- [x] Build simulator v3 (price, promotion, marketing, digital funnel, inventory, ops capacity)
- [x] Validation notebook 01 (elasticity recovery proof: naive → FE → FE+controls)
- [x] Generate & validate full dataset (2.92M rows / 56 cols, 2024-01-01 to 2025-12-31)
- [x] Commit checkpoint (`a6442f6`)

## Phase 2 — Econometric Elasticity Recovery 🔄 IN PROGRESS

- [ ] Naive pooled OLS baseline (log units ~ log price)
- [ ] SKU fixed-effects model
- [ ] Fixed effects + full controls model (promo depth, marketing spend, event/channel/day/region)
- [ ] Diagnostics: standard errors, confidence intervals, residual analysis
- [ ] Implement in `src/modeling.py`
- [ ] Rewrite `notebooks/03_elasticity_modeling.ipynb` with real outputs

## Phase 3 — Bayesian Demand Modeling ⏸️ NOT STARTED

- [ ] PyMC Bayesian linear regression
- [ ] Posterior distributions and credible intervals
- [ ] Hierarchical pooling (SKU-level elasticity toward category-level)

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
