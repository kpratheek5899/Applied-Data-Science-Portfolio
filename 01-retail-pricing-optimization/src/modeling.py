"""
Elasticity modeling module for the Retail Pricing & Capacity Optimization Engine.

Implements the naive -> SKU fixed effects -> fixed effects + full controls
progression validated in notebooks/01_data_simulation_design.ipynb, formalized
with statsmodels so every estimate carries real standard errors, confidence
intervals, and fit diagnostics instead of a bare coefficient.

Because true_price_elasticity is a known ground-truth column baked into the
simulator, every model here can be graded directly against the truth rather
than an unknown black box -- that comparison is the point of Phase 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

FULL_MODEL_FORMULA = (
    "log_units ~ log_price + C(sku) + promotion_depth "
    "+ log_search + log_social + log_display "
    "+ C(event_combo) + C(channel) + C(day_of_week) + C(region)"
)


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to rows with positive realized units and add the log-transformed
    columns the elasticity models need.

    Rows with zero units (stockouts / fully lost demand) are dropped because
    log(0) is undefined -- this mirrors the approach used in notebook 01.
    """
    work = df[df["units"] > 0].copy()

    work["log_price"] = np.log(work["price"])
    work["log_units"] = np.log(work["units"])
    work["log_search"] = np.log(work["search_spend"])
    work["log_social"] = np.log(work["social_spend"])
    work["log_display"] = np.log(work["display_spend"])
    work["event_combo"] = (
        work["event_name"].astype(str) + "_" + work["event_phase"].astype(str)
    )

    return work


def _drop_unused_categories(g: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unused levels from categorical columns before fitting.

    `sku` (and the other categoricals) carry the *global* set of category
    levels from `utils.optimize_dtypes` even after filtering down to a single
    product_type. Left alone, patsy's `C(...)` builds a dummy column for
    every level in `.cat.categories`, not just the ones present in `g` --
    padding the design matrix with all-zero columns for absent SKUs. That
    makes the matrix rank-deficient, and statsmodels silently resolves it via
    pseudo-inverse instead of raising, which can quietly perturb the
    coefficient on log_price. Dropping unused levels first keeps the design
    matrix exactly as small as it should be.
    """
    g = g.copy()
    for col in ["sku", "channel", "day_of_week", "region"]:
        if isinstance(g[col].dtype, pd.CategoricalDtype):
            g[col] = g[col].cat.remove_unused_categories()
    return g


def fit_naive_ols(work: pd.DataFrame) -> RegressionResultsWrapper:
    """
    Attempt A: naive pooled OLS, log(units) ~ log(price).

    No SKU controls at all, so SKUs at very different price points (and with
    very different baseline demand) get pooled together. This is the
    regression a naive first pass at elasticity estimation would run, and it
    badly fails to recover the true elasticity (see notebook 01 / Phase 2
    results).
    """
    model = smf.ols("log_units ~ log_price", data=work).fit(
        cov_type="cluster", cov_kwds={"groups": work["sku"]}
    )
    return model


def fit_fixed_effects_ols(work: pd.DataFrame) -> RegressionResultsWrapper:
    """
    Attempt B: SKU fixed effects via least-squares dummy variables (LSDV),
    log(units) ~ log(price) + C(sku).

    Absorbs each SKU's own price level and baseline demand, which fixes the
    pooling problem from Attempt A. It still overshoots the true elasticity,
    though, because it has no way to separate the pure price effect from
    demand lift driven by promotions and marketing that happen to coincide
    with price changes (omitted variable bias).
    """
    g = _drop_unused_categories(work)
    model = smf.ols("log_units ~ log_price + C(sku)", data=g).fit(
        cov_type="cluster", cov_kwds={"groups": g["sku"]}
    )
    return model


def fit_full_model(work: pd.DataFrame) -> RegressionResultsWrapper:
    """
    Attempt C: SKU fixed effects + promotion / marketing / event / channel /
    day-of-week / region controls.

    This is the specification validated in notebook 01: once promotion depth
    and marketing spend are controlled for alongside the fixed effects, the
    coefficient on log_price recovers the true simulated elasticity closely.
    """
    g = _drop_unused_categories(work)
    model = smf.ols(FULL_MODEL_FORMULA, data=g).fit(
        cov_type="cluster", cov_kwds={"groups": g["sku"]}
    )
    return model


def run_elasticity_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the naive -> fixed effects -> fixed effects + controls progression
    separately for each product type and summarize recovered elasticity
    (the coefficient on log_price in each spec) against the true simulated
    elasticity.

    Returns one row per product_type with coefficients, standard errors, a
    95% CI and R^2 for the full model, and the full model's percent error
    against ground truth.
    """
    work = prepare_model_data(df)

    rows = []
    for product_type, g in work.groupby("product_type", observed=True):
        naive = fit_naive_ols(g)
        fe = fit_fixed_effects_ols(g)
        full = fit_full_model(g)

        true_elasticity = g["true_price_elasticity"].iloc[0]
        full_coef = full.params["log_price"]
        ci_low, ci_high = full.conf_int().loc["log_price"]

        rows.append(
            {
                "product_type": product_type,
                "n_obs": len(g),
                "n_skus": g["sku"].nunique(),
                "true_elasticity": true_elasticity,
                "naive_coef": naive.params["log_price"],
                "naive_se": naive.bse["log_price"],
                "fe_coef": fe.params["log_price"],
                "fe_se": fe.bse["log_price"],
                "full_coef": full_coef,
                "full_se": full.bse["log_price"],
                "full_ci_low": ci_low,
                "full_ci_high": ci_high,
                "full_r2": full.rsquared,
                "full_pct_error": abs(full_coef - true_elasticity) / abs(true_elasticity) * 100,
            }
        )

    return pd.DataFrame(rows).set_index("product_type").sort_values("true_elasticity")


def fit_bayesian_model():
    """Fit Bayesian elasticity model. Implemented in Phase 3 (PyMC hierarchical model)."""
    pass
