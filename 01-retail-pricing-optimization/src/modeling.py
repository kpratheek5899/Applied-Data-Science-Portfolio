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

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
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


BAYESIAN_CONTROL_COLUMNS = ["log_search", "log_social", "log_display"]


def prepare_bayesian_data(
    df: pd.DataFrame, sample_frac: float = 0.02, random_seed: int = 42
) -> pd.DataFrame:
    """
    Build the panel the Bayesian hierarchical model is fit on: a random
    row-level subsample of the full (date x store x sku x channel) data,
    NOT an aggregation.

    Phase 2's fixed-effects models estimate one elasticity per product type
    at full store/channel resolution. Phase 3's model instead estimates one
    elasticity *per SKU*, partially pooled toward its product type -- that
    needs a SKU-level random effect sampled with NUTS, and MCMC over the
    full ~2.9M-row panel is not tractable on a laptop without a C compiler
    for PyTensor (confirmed by timing a representative model on this
    machine).

    The first version of this function aggregated up to (date, sku) instead
    of subsampling, averaging price across each day's ~80 store/channel
    rows. That turned out to be a real bug, not just a speed shortcut: most
    of a SKU's *row-level* price movement comes from independent per-row
    noise (`price_index`), which averages toward zero across 80 rows,
    leaving mostly promotion-driven price changes -- which are collinear
    with the `promotion_depth` control that's also in the model. Recovered
    elasticity degraded sharply for the two highest-promo-sensitivity
    product types (Promo Sensitive, Seasonal), and one run even flipped the
    sign for Premium. Randomly subsampling rows instead of averaging them
    preserves the real per-row price variation that identifies elasticity
    (verified: a 3% row-level subsample recovers all four product types
    within ~4% of true elasticity via OLS, matching Phase 2's fixed-effects
    result), while still cutting the ~2.4M-row (units > 0) panel down to a
    size NUTS can sample in minutes rather than hours.
    """
    work = prepare_model_data(df)
    sample = work.sample(frac=sample_frac, random_state=random_seed).copy()

    sample = _drop_unused_categories(sample)

    for col in ["sku", "product_type", "day_of_week", "channel", "region"]:
        sample[col] = sample[col].astype(str)

    return sample.reset_index(drop=True)


def _build_hierarchy_indices(agg: pd.DataFrame):
    """
    Build the integer index arrays the hierarchical model needs: which SKU
    each row belongs to, which product type each SKU belongs to, and the
    ordered label lists both index into.
    """
    sku_codes, sku_levels = pd.factorize(agg["sku"], sort=True)
    sku_to_type = agg.groupby("sku")["product_type"].first().loc[sku_levels]
    type_codes, type_levels = pd.factorize(sku_to_type.to_numpy(), sort=True)
    return sku_codes, sku_levels, type_codes, type_levels


def build_hierarchical_model(agg: pd.DataFrame) -> tuple[pm.Model, dict]:
    """
    Build (without sampling) the hierarchical elasticity model.

    Structure: log(units) = alpha_sku + beta_sku * log(price) + controls.
    `beta_sku` (the price elasticity) is partially pooled: each SKU's
    elasticity is drawn from a Normal centered on its product type's mean
    elasticity, which is itself drawn from a global mean across product
    types. Non-centered parameterization (`z_type`, `z_sku`) avoids the
    funnel geometry that causes divergences in centered hierarchical models.
    Promotion depth, marketing spend, event window, and day-of-week enter as
    ordinary (non-hierarchical) fixed effects -- the thing Phase 3 adds over
    Phase 2 is pooling on the price coefficient specifically, not on every
    control.

    Three conditioning/specification fixes, the first two standard practice
    for NUTS and both needed in practice here (an unconditioned version of
    this model hit max tree depth and took >10x longer to sample on the real
    data than an unconditioned synthetic proxy of the same size); the third
    fixes a real bias, not a sampling problem:
    - `log_price` is demeaned *within each SKU* before entering the model
      (identical to the within-transformation `fit_fixed_effects_ols` uses
      in Phase 2), which makes it near-orthogonal to `alpha_sku` and removes
      most of the alpha/beta correlation that was driving deep trees.
    - The continuous controls are standardized (zero mean, unit variance) so
      `gamma`'s posterior geometry isn't stretched by log_search living on a
      completely different scale than promotion_depth.
    - `promotion_depth`'s coefficient varies *by product type*
      (`gamma_promo`) instead of being shared across all four like the other
      controls. An earlier version of this model shared it, and recovered
      elasticity was off by 9-18% for three of four product types even after
      NUTS converged cleanly (good r-hat, no divergences) -- i.e. a real,
      stable bias, not a sampling artifact. Forcing a plain OLS on the same
      data to share one promotion_depth coefficient across types reproduced
      those exact numbers, confirming the cause: the simulator's promotion
      sensitivity genuinely varies by product type (Low/Medium/High/Very
      High per the spec), so a shared coefficient misattributes each type's
      own promotion-driven demand lift into its price coefficient by a
      different amount. Marketing spend (search/social/display) keeps a
      single shared coefficient because channel elasticities are specified
      uniformly across the business, not by product type.

    Returned separately from `fit_bayesian_model` so the model structure can
    be unit-tested (e.g. checking shapes/coords) without paying the cost of
    running NUTS.
    """
    sku_codes, sku_levels, type_codes, type_levels = _build_hierarchy_indices(agg)
    type_of_sku = np.asarray(type_codes)
    type_of_row = type_of_sku[sku_codes]

    # Event window is controlled at the coarser event_phase level (normal /
    # pre_event / event / post_event) rather than the full event_combo cross
    # used in Phase 2 -- far fewer dummy columns, which meaningfully shrinks
    # the parameter-correlation space NUTS has to explore, at the cost of not
    # distinguishing *which* holiday is driving an event-window lift.
    event_dummies = pd.get_dummies(agg["event_phase"], prefix="ph", drop_first=True)
    dow_dummies = pd.get_dummies(agg["day_of_week"], prefix="dow", drop_first=True)
    channel_dummies = pd.get_dummies(agg["channel"], prefix="ch", drop_first=True)
    region_dummies = pd.get_dummies(agg["region"], prefix="reg", drop_first=True)

    continuous = agg[BAYESIAN_CONTROL_COLUMNS].astype(float)
    continuous_std = (continuous - continuous.mean()) / continuous.std()
    promo_std = (
        (agg["promotion_depth"] - agg["promotion_depth"].mean()) / agg["promotion_depth"].std()
    ).to_numpy()

    controls = pd.concat(
        [continuous_std, event_dummies, dow_dummies, channel_dummies, region_dummies], axis=1
    ).astype(float)

    X = controls.to_numpy()
    log_price_dm = (
        agg["log_price"] - agg.groupby("sku")["log_price"].transform("mean")
    ).to_numpy()
    log_units = agg["log_units"].to_numpy()

    coords = {
        "sku": sku_levels,
        "product_type": type_levels,
        "control": controls.columns.to_numpy(),
        "obs_id": agg.index.to_numpy(),
    }

    with pm.Model(coords=coords) as model:
        sku_idx = pm.Data("sku_idx", sku_codes, dims="obs_id")

        mu_global = pm.Normal("mu_global", mu=-1.5, sigma=1.0)
        sigma_type = pm.HalfNormal("sigma_type", sigma=0.5)
        z_type = pm.Normal("z_type", 0.0, 1.0, dims="product_type")
        mu_type = pm.Deterministic(
            "mu_type", mu_global + z_type * sigma_type, dims="product_type"
        )

        sigma_sku = pm.HalfNormal("sigma_sku", sigma=0.5)
        z_sku = pm.Normal("z_sku", 0.0, 1.0, dims="sku")
        beta_sku = pm.Deterministic(
            "beta_sku", mu_type[type_of_sku] + z_sku * sigma_sku, dims="sku"
        )

        alpha_sku = pm.Normal("alpha_sku", mu=0.0, sigma=5.0, dims="sku")
        gamma_promo = pm.Normal("gamma_promo", mu=0.0, sigma=1.0, dims="product_type")
        gamma = pm.Normal("gamma", mu=0.0, sigma=1.0, dims="control")

        mu = (
            alpha_sku[sku_idx]
            + beta_sku[sku_idx] * log_price_dm
            + gamma_promo[type_of_row] * promo_std
            + pm.math.dot(X, gamma)
        )
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)
        pm.Normal("log_units_obs", mu=mu, sigma=sigma_obs, observed=log_units)

    meta = {
        "sku_levels": sku_levels,
        "type_levels": type_levels,
        "type_codes": type_codes,
    }
    return model, meta


def fit_bayesian_model(
    df: pd.DataFrame,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 42,
):
    """
    Fit the hierarchical Bayesian elasticity model end to end: aggregate the
    raw data, build the model, and sample it with NUTS.

    Returns (idata, agg, meta):
    - idata: arviz InferenceData with the full posterior
    - agg: the (date x sku) aggregated dataframe the model was fit on
    - meta: index metadata from `build_hierarchical_model`, needed to map
      `beta_sku` / `mu_type` posterior draws back to SKU and product-type
      labels in `summarize_bayesian_elasticity`
    """
    agg = prepare_bayesian_data(df)
    model, meta = build_hierarchical_model(agg)

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=False,
        )

    return idata, agg, meta


def summarize_bayesian_elasticity(idata, agg: pd.DataFrame, meta: dict):
    """
    Summarize posterior elasticity estimates at the SKU level (`beta_sku`)
    and product-type level (`mu_type`), each compared against the known true
    elasticity.

    Returns (sku_summary, type_summary) dataframes with posterior mean, SD,
    a 95% highest-density interval, and percent error vs. ground truth.
    """
    sku_levels = meta["sku_levels"]
    type_levels = meta["type_levels"]
    type_codes = meta["type_codes"]

    truth_by_sku = agg.groupby("sku")["true_price_elasticity"].first().astype(float)
    truth_by_type = agg.groupby("product_type")["true_price_elasticity"].first().astype(float)

    sku_summary = az.summary(idata, var_names=["beta_sku"], ci_prob=0.95).copy()
    numeric_cols = sku_summary.columns.drop("product_type", errors="ignore")
    sku_summary[numeric_cols] = sku_summary[numeric_cols].astype(float)
    sku_summary.index = sku_levels
    sku_summary["product_type"] = [type_levels[c] for c in type_codes]
    sku_summary["true_elasticity"] = (
        truth_by_sku.reindex(sku_levels).astype(float).to_numpy()
    )
    sku_summary["pct_error"] = (
        (sku_summary["mean"] - sku_summary["true_elasticity"]).abs()
        / sku_summary["true_elasticity"].abs()
        * 100
    )

    type_summary = az.summary(idata, var_names=["mu_type"], ci_prob=0.95).copy()
    type_summary = type_summary.astype(float)
    type_summary.index = type_levels
    type_summary["true_elasticity"] = (
        truth_by_type.reindex(type_levels).astype(float).to_numpy()
    )
    type_summary["pct_error"] = (
        (type_summary["mean"] - type_summary["true_elasticity"]).abs()
        / type_summary["true_elasticity"].abs()
        * 100
    )

    return (
        sku_summary.sort_values("true_elasticity"),
        type_summary.sort_values("true_elasticity"),
    )
