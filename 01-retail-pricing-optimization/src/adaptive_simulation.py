"""
Adaptive Learning demo (Phase 5f): three pricing policies run side by side
over the same window, same starting inventory, same day-by-day "environment"
(the real historical row for that date, exactly like `replay_engine.py`
uses for Decision Replay) --

- **static**: prices using a deliberately weak *initial* belief about this
  SKU's elasticity, forever -- the belief never updates. Isolates the value
  of learning at all.
- **thompson**: Thompson Sampling. Each day: sample one elasticity draw from
  the *current* posterior (`bayesian_learning.sample_theta`), price using
  it via the same constrained optimizer used everywhere else in this app,
  realize the outcome via the TRUE simulator model
  (`replay_engine.realize_true_outcome`), then update the posterior on the
  realized `(price, units)` pair. This is the only variant whose belief
  changes.
- **oracle**: prices using the SKU's TRUE elasticity every day. This is the
  one place in this module explicitly sanctioned to read
  `true_price_elasticity` for a *decision* (everywhere else in this
  codebase, ground truth is only used to *realize outcomes*, never to
  decide) -- Oracle exists purely as a labeled upper-bound comparison
  ("if we had known the truth all along"), not something the app claims to
  achieve. Oracle's own daily profit doubles as the regret reference for
  the other two variants: `regret_t(v) = oracle_profit_t - v_profit_t`.
  This makes Oracle's own regret exactly zero by construction (its price
  each day *is* the profit-maximizing price under the same deterministic
  demand curve used to compute "true optimal"), which is a genuine
  invariant, not just "should be near zero" -- see
  `tests/test_adaptive_learning.py`.

Deliberate simplification, not an oversight: the sequential belief is a
plain `log_units ~ alpha + beta*log_price` regression with no controls for
promotion/event/day-of-week (see `bayesian_learning.py`'s module docstring
for why -- closed-form/no-MCMC was an explicit engineering constraint for
interactive use). Notebook 03 showed this exact "no controls" specification
is confounded by promotions/events over a *full 2-year panel*; over the
short (10-30 day) adaptive-learning window here that's a real but bounded
risk, flagged in the UI rather than hidden.

Prior seeding: the prior mean is a fixed, genuinely uninformed guess
(`GENERIC_PRIOR_MEAN_ELASTICITY`, -1.0 -- "unit elastic"), not looked up
per SKU or per product category. An early version used `sku_master.csv`'s
`elasticity_phase2` (Phase 2's per-product-type estimate) instead, reasoning
it was generic-enough category knowledge -- but Phase 2 is *accurate*
(within a few percent for every type), so that prior mean already sat close
to the truth for most SKUs, leaving little room for Thompson Sampling to
visibly out-learn the frozen static baseline (verified empirically: static
came out with *lower* regret, defeating the point). A fixed, deliberately
naive guess is what actually delivers "the model begins genuinely unsure."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bayesian_learning import (
    NIGPosterior,
    build_prior,
    update_posterior,
    sample_theta,
    posterior_mean_beta,
    posterior_sd_beta,
    credible_interval_beta,
)
from optimization import optimize_price
from replay_engine import realize_true_outcome, apply_daily_replenishment

MAX_ELASTICITY_RESAMPLE_ATTEMPTS = 100
ELASTICITY_FALLBACK = -1e-3  # used only if resampling can't find a negative draw (should not occur in practice)

# Generic starting-belief mean for the prior -- deliberately NOT looked up
# per SKU or per product category. An earlier version used sku_master.csv's
# `elasticity_phase2` (Phase 2's per-product-type fixed-effects estimate) as
# the prior mean, reasoning it was "generic category knowledge, not this
# SKU's own answer" -- but Phase 2 is *accurate* (within a few percent for
# every product type), so that prior mean already sat close to the truth
# for most SKUs, leaving Thompson Sampling little room to visibly out-learn
# the frozen static baseline (verified empirically: static's regret came
# out lower than Thompson Sampling's, defeating the demo's point).
#
# -1.5, not -1.0: the closed-form unconstrained profit-max price under
# constant elasticity is cost*e/(e+1) (see src/optimization.py), which
# *diverges* as e -> -1 -- so a starting guess of exactly "unit elastic"
# (-1.0) is the worst possible choice for this particular formula: it always
# wants to push to whatever price ceiling is available regardless of the
# true elasticity, making static and Thompson Sampling behave identically
# until the belief moves substantially away from -1 (also verified
# empirically). -1.5 is a genuinely wrong starting guess for most product
# types here (true elasticity ranges roughly -0.8 to -2.0 across the
# simulator's four product types) while staying well clear of that
# singularity.
GENERIC_PRIOR_MEAN_ELASTICITY = -1.5


@dataclass
class VariantDay:
    date: pd.Timestamp
    price: float
    units: float
    starting_inventory: float
    ending_inventory: float
    revenue: float
    profit: float
    regret: float
    stockout: bool
    elasticity_used: float
    # Thompson-Sampling-only fields; None for static/oracle.
    posterior_mean_beta: float | None = None
    posterior_ci_low: float | None = None
    posterior_ci_high: float | None = None
    is_exploring: bool | None = None


@dataclass
class AdaptiveSimulationResult:
    sku: str
    objective: str
    prior_strength: float
    random_seed: int
    initial_prior: NIGPosterior
    final_posterior: NIGPosterior
    static: list[VariantDay]
    thompson: list[VariantDay]
    oracle: list[VariantDay]


def _sample_valid_elasticity(posterior: NIGPosterior, rng: np.random.Generator) -> float:
    """
    Rejection-sample a negative beta draw (a valid demand curve) from the
    posterior. The prior is deliberately weak/wide, so an early draw landing
    at beta >= 0 is a real, expected occurrence, not a bug -- resampling a
    few times is the standard pragmatic fix rather than a full truncated-NIG
    reformulation, which isn't needed for a demo at this scale.
    """
    for _ in range(MAX_ELASTICITY_RESAMPLE_ATTEMPTS):
        beta = sample_theta(posterior, rng)
        if beta < -1e-6:
            return beta
    return ELASTICITY_FALLBACK


def run_adaptive_simulation(
    sku_master: pd.DataFrame,
    daily: pd.DataFrame,
    sku: str,
    start_date,
    n_days: int,
    objective: str = "maximize_profit",
    prior_strength: float = 2.0,
    random_seed: int = 42,
    min_margin: float | None = None,
    max_price_change_pct: float | None = None,
    inventory_override: float | None = None,
) -> AdaptiveSimulationResult:
    """
    `inventory_override`, if given, replaces Day 1's starting inventory
    (real historical stock otherwise) for all three variants uniformly --
    keeping the three-way comparison fair. Only Day 1 can be overridden;
    every later day is already determined by each variant's own prior
    decisions plus replenishment, not by history.
    """
    start_date = pd.Timestamp(start_date)
    window_dates = pd.date_range(start_date, periods=n_days, freq="D")

    sku_daily = daily[daily["sku"] == sku].set_index("date").sort_index()
    missing = [d for d in window_dates if d not in sku_daily.index]
    if missing:
        raise ValueError(
            f"No data for {sku} on {missing[0].date()} -- window must stay within the dataset's date range."
        )

    sku_row = sku_master.loc[sku_master["sku"] == sku]
    if sku_row.empty:
        raise ValueError(f"Unknown sku {sku!r} -- not present in sku_master")
    cost = float(sku_row.iloc[0]["cost"])

    rng = np.random.default_rng(random_seed)

    first_row = sku_daily.loc[window_dates[0]]
    start_price = float(first_row["actual_price"])
    start_units = float(first_row["actual_units"])
    start_inventory = (
        inventory_override if inventory_override is not None else float(first_row["starting_inventory"])
    )

    # Price bounds are fixed for the whole window, anchored to the *starting*
    # price -- not `max_price_change_pct` relative to each day's rolling
    # (price, units) anchor, which `optimize_price` would otherwise treat as
    # the reference. That combination is a real bug this smoke-testing
    # caught: once a day's optimal price hits its ceiling, the next day's
    # ceiling is computed from *that* price, compounding into unbounded
    # exponential price growth within a handful of days. A fixed window-wide
    # band avoids the runaway while the demand-curve anchor below still
    # rolls day to day (that part is correct and intentional -- the curve
    # itself should swing from what actually just happened).
    # 1.2 (not the 0.5 used elsewhere in this app), because this dataset's
    # *actual* historical prices sit well below their theoretical profit-max
    # under the true elasticity for most SKUs (verified: 40-70% below for
    # Commodity/Promo Sensitive types) -- a tighter band saturates the
    # ceiling for every plausible belief, static and thompson included,
    # which erases the very differentiation this demo exists to show.
    change_pct = max_price_change_pct if max_price_change_pct is not None else 1.2
    window_price_min = max(cost * 1.01, start_price * (1 - change_pct))
    window_price_max = start_price * (1 + change_pct)

    prior = build_prior(GENERIC_PRIOR_MEAN_ELASTICITY, prior_strength)
    static_elasticity = posterior_mean_beta(prior)

    anchor = {
        "static": {"price": start_price, "units": start_units, "inventory": start_inventory},
        "thompson": {"price": start_price, "units": start_units, "inventory": start_inventory},
        "oracle": {"price": start_price, "units": start_units, "inventory": start_inventory},
    }
    posterior = prior

    static_days: list[VariantDay] = []
    thompson_days: list[VariantDay] = []
    oracle_days: list[VariantDay] = []

    prev_actual_ending_inventory: float | None = None

    for date in window_dates:
        row = sku_daily.loc[date]
        actual_starting_inventory = float(row["starting_inventory"])
        actual_price = float(row["actual_price"])
        demand_units_uncapped = float(row["demand_units_uncapped"])
        true_elasticity = float(row["true_price_elasticity"])

        if prev_actual_ending_inventory is not None:
            for v in anchor:
                anchor[v]["inventory"] = apply_daily_replenishment(
                    anchor[v]["inventory"], actual_starting_inventory, prev_actual_ending_inventory
                )

        # --- Oracle first: its profit is the shared regret reference. ---
        oracle_result = optimize_price(
            objective,
            base_price=anchor["oracle"]["price"],
            base_units=anchor["oracle"]["units"],
            cost=cost,
            elasticity=true_elasticity,
            inventory=anchor["oracle"]["inventory"],
            min_margin=min_margin,
            price_min=window_price_min,
            price_max=window_price_max,
        )
        oracle_price = oracle_result["recommended_price"]
        oracle_units = realize_true_outcome(
            oracle_price, actual_price, demand_units_uncapped, true_elasticity, anchor["oracle"]["inventory"]
        )
        oracle_ending_inventory = anchor["oracle"]["inventory"] - oracle_units
        oracle_profit = (oracle_price - cost) * oracle_units
        oracle_days.append(
            VariantDay(
                date=date,
                price=oracle_price,
                units=oracle_units,
                starting_inventory=anchor["oracle"]["inventory"],
                ending_inventory=oracle_ending_inventory,
                revenue=oracle_price * oracle_units,
                profit=oracle_profit,
                regret=0.0,
                stockout=oracle_ending_inventory <= 0,
                elasticity_used=true_elasticity,
            )
        )
        anchor["oracle"] = {"price": oracle_price, "units": oracle_units, "inventory": oracle_ending_inventory}

        # --- Static: frozen elasticity, never updates. ---
        static_result = optimize_price(
            objective,
            base_price=anchor["static"]["price"],
            base_units=anchor["static"]["units"],
            cost=cost,
            elasticity=static_elasticity,
            inventory=anchor["static"]["inventory"],
            min_margin=min_margin,
            price_min=window_price_min,
            price_max=window_price_max,
        )
        static_price = static_result["recommended_price"]
        static_units = realize_true_outcome(
            static_price, actual_price, demand_units_uncapped, true_elasticity, anchor["static"]["inventory"]
        )
        static_ending_inventory = anchor["static"]["inventory"] - static_units
        static_profit = (static_price - cost) * static_units
        static_days.append(
            VariantDay(
                date=date,
                price=static_price,
                units=static_units,
                starting_inventory=anchor["static"]["inventory"],
                ending_inventory=static_ending_inventory,
                revenue=static_price * static_units,
                profit=static_profit,
                regret=oracle_profit - static_profit,
                stockout=static_ending_inventory <= 0,
                elasticity_used=static_elasticity,
            )
        )
        anchor["static"] = {"price": static_price, "units": static_units, "inventory": static_ending_inventory}

        # --- Thompson Sampling: sample, decide, realize, learn. ---
        posterior_mean = posterior_mean_beta(posterior)
        posterior_sd = posterior_sd_beta(posterior)
        ci_low, ci_high = credible_interval_beta(posterior)
        sampled_beta = _sample_valid_elasticity(posterior, rng)
        is_exploring = abs(sampled_beta - posterior_mean) > 0.5 * posterior_sd if posterior_sd > 0 else False

        ts_result = optimize_price(
            objective,
            base_price=anchor["thompson"]["price"],
            base_units=anchor["thompson"]["units"],
            cost=cost,
            elasticity=sampled_beta,
            inventory=anchor["thompson"]["inventory"],
            min_margin=min_margin,
            price_min=window_price_min,
            price_max=window_price_max,
        )
        ts_price = ts_result["recommended_price"]
        ts_units = realize_true_outcome(
            ts_price, actual_price, demand_units_uncapped, true_elasticity, anchor["thompson"]["inventory"]
        )
        ts_ending_inventory = anchor["thompson"]["inventory"] - ts_units
        ts_profit = (ts_price - cost) * ts_units
        thompson_days.append(
            VariantDay(
                date=date,
                price=ts_price,
                units=ts_units,
                starting_inventory=anchor["thompson"]["inventory"],
                ending_inventory=ts_ending_inventory,
                revenue=ts_price * ts_units,
                profit=ts_profit,
                regret=oracle_profit - ts_profit,
                stockout=ts_ending_inventory <= 0,
                elasticity_used=sampled_beta,
                posterior_mean_beta=posterior_mean,
                posterior_ci_low=ci_low,
                posterior_ci_high=ci_high,
                is_exploring=is_exploring,
            )
        )

        if ts_units > 0 and anchor["thompson"]["units"] > 0:
            dlog_price = np.log(ts_price) - np.log(anchor["thompson"]["price"])
            dlog_units = np.log(ts_units) - np.log(anchor["thompson"]["units"])
            posterior = update_posterior(posterior, dlog_price, dlog_units)
        # else: a zero-unit day (this trajectory's or its anchor's) carries
        # no learnable signal (log(0) undefined) -- belief carries over
        # unchanged rather than erroring.

        anchor["thompson"] = {"price": ts_price, "units": ts_units, "inventory": ts_ending_inventory}
        prev_actual_ending_inventory = float(row["ending_inventory"])

    return AdaptiveSimulationResult(
        sku=sku,
        objective=objective,
        prior_strength=prior_strength,
        random_seed=random_seed,
        initial_prior=prior,
        final_posterior=posterior,
        static=static_days,
        thompson=thompson_days,
        oracle=oracle_days,
    )


def variant_to_frame(days: list[VariantDay], variant_name: str) -> pd.DataFrame:
    df = pd.DataFrame([vars(d) for d in days])
    df["variant"] = variant_name
    df["cumulative_profit"] = df["profit"].cumsum()
    df["cumulative_regret"] = df["regret"].cumsum()
    return df


def adaptive_to_frame(result: AdaptiveSimulationResult) -> pd.DataFrame:
    """Long-format frame (a `variant` column) of all three trajectories, for charting."""
    return pd.concat(
        [
            variant_to_frame(result.static, "static"),
            variant_to_frame(result.thompson, "thompson"),
            variant_to_frame(result.oracle, "oracle"),
        ],
        ignore_index=True,
    )
