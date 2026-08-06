"""
One-time offline precompute for the Phase 5 Streamlit app.

Reads the full simulated dataset (~2.9M rows, gitignored, regenerate with
`python src/data_generation.py`) and Phase 3's cached Bayesian posterior
(`data/processed/phase3_idata.pkl`, gitignored, regenerate with
`fit_bayesian_model` in `src/modeling.py`), and writes three small,
committed CSVs to `data/app/` that the live app reads instead:

- sku_master.csv        -- 50 rows, static per-SKU attributes + estimated
                            elasticity. No ground-truth columns.
- daily_sku_timeseries.csv -- ~36.5k rows (50 skus x 731 days), aggregated
                            to daily company-wide grain. Powers scenario
                            presets, custom date/range selection, and
                            Decision Replay.
- posterior_samples.csv -- ~15k rows, thinned posterior draws of per-SKU
                            elasticity for the risk-aware optimizer.

Ground-truth boundary: `true_price_elasticity` in daily_sku_timeseries.csv
is the ONLY ground-truth column shipped to the app, and it exists solely so
Decision Replay's closed loop can realize "what actually would have
happened" at a candidate price (src/replay_engine.py::realize_true_outcome
is the only function allowed to read it). Nothing that recommends a price
may read it -- see PROGRESS.md / the Phase 5 plan for the full rationale.

Run from the project root (or notebooks/scripts dir with sys.path adjusted):
    python scripts/build_app_data.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import load_retail_data
from modeling import (
    prepare_bayesian_data,
    build_hierarchical_model,
    summarize_bayesian_elasticity,
    run_elasticity_recovery,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "simulated" / "nova_retail_simulated_data_v3_full.csv"
IDATA_PATH = PROJECT_ROOT / "data" / "processed" / "phase3_idata.pkl"
OUT_DIR = PROJECT_ROOT / "data" / "app"

N_POSTERIOR_DRAWS_PER_SKU = 300
RANDOM_SEED = 42


def _stock_status_from_pct(pct_remaining: pd.Series, ending_inventory: pd.Series) -> pd.Series:
    """Mirror data_generation.py's stock-status thresholds on aggregated inventory."""
    conditions = [
        ending_inventory <= 0,
        pct_remaining <= 0.15,
        pct_remaining <= 0.40,
        pct_remaining > 0.40,
    ]
    choices = ["Out Of Stock", "Limited Availability", "Low Stock", "In Stock"]
    return pd.Series(np.select(conditions, choices, default="Unknown"), index=pct_remaining.index)


def build_sku_master(df: pd.DataFrame, phase3_sku_summary: pd.DataFrame, phase2_type_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Static per-SKU attributes + estimated elasticity (Phase 3 primary,
    Phase 2 per-type fallback). Deliberately excludes true_price_elasticity,
    promo_sensitivity, and seasonality_strength (simulator DGP internals not
    needed by anything in the app, decision path or display).
    """
    master = (
        df.groupby("sku", observed=True)
        .agg(category=("category", "first"), product_type=("product_type", "first"), cost=("cost", "first"))
        .reset_index()
    )

    master = master.merge(
        phase3_sku_summary[["mean", "sd"]].rename(
            columns={"mean": "elasticity_phase3_mean", "sd": "elasticity_phase3_sd"}
        ),
        left_on="sku",
        right_index=True,
        how="left",
    )

    phase2_lookup = phase2_type_summary["full_coef"].rename("elasticity_phase2")
    master = master.merge(phase2_lookup, left_on="product_type", right_index=True, how="left")

    return master.sort_values("sku").reset_index(drop=True)


def build_daily_sku_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate row-level (date x store x sku x channel) data to daily
    company-wide (date x sku) grain. Aggregation here is for display and
    Decision Replay only -- nothing is being re-estimated from it, so
    Phase 3's aggregation-bias lesson doesn't apply.
    """

    work = df.copy()
    work["_price_x_units"] = work["price"] * work["units"]

    grouped = work.groupby(["date", "sku"], observed=True)

    daily = grouped.agg(
        actual_units=("units", "sum"),
        demand_units_uncapped=("demand_units", "sum"),
        starting_inventory=("starting_inventory", "sum"),
        inventory_capacity=("inventory_capacity", "sum"),
        event_name=("event_name", "first"),
        event_phase=("event_phase", "first"),
        promotion_depth=("promotion_depth", "first"),
        true_price_elasticity=("true_price_elasticity", "first"),
        _price_x_units=("_price_x_units", "sum"),
        _mean_price=("price", "mean"),
    ).reset_index()

    # src/data_generation.py's event windows overlap (Black Friday's event
    # window runs through Dec 1, Cyber Monday's starts Nov 28), and because
    # events are applied in a fixed dict order with Cyber Monday last, its
    # label silently overwrites Black Friday's on the shared days. Rather
    # than touch the committed simulator (which would require regenerating
    # the full dataset and invalidating every already-fitted Phase 2-4
    # model), relabel it here for app-facing display -- "Black Friday Cyber
    # Monday" is standard retail terminology for the combined weekend
    # anyway, so this is a legitimate label, not just a patch.
    daily["event_name"] = daily["event_name"].astype(str).replace("Cyber Monday", "Black Friday Cyber Monday")

    # Units-weighted average price, falling back to a simple mean on days
    # with zero network-wide units (avoids a division by zero).
    daily["actual_price"] = np.where(
        daily["actual_units"] > 0,
        daily["_price_x_units"] / daily["actual_units"],
        daily["_mean_price"],
    )
    daily = daily.drop(columns=["_price_x_units", "_mean_price"])

    daily["ending_inventory"] = daily["starting_inventory"] - daily["actual_units"]
    daily["inventory_pct_remaining"] = (
        daily["ending_inventory"] / daily["inventory_capacity"]
    ).replace([np.inf, -np.inf], 0).fillna(0)
    daily["stock_status"] = _stock_status_from_pct(daily["inventory_pct_remaining"], daily["ending_inventory"])

    return daily.sort_values(["sku", "date"]).reset_index(drop=True)


def build_control_effects(idata, bayes_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Point estimates (posterior means) of Phase 3's fitted promotion/event/
    day-of-week effects, exported so Adaptive Learning can residualize its
    1-parameter online update against them (see src/bayesian_learning.py).
    Channel, region, and marketing-spend (search/social/display)
    coefficients also exist in the posterior but aren't exported here:
    daily_sku_timeseries.csv doesn't carry channel/region/marketing-spend at
    its aggregated (date, sku) grain, so there's nothing at runtime to
    multiply them against.
    """
    gamma_promo = idata.posterior["gamma_promo"].mean(dim=("chain", "draw"))
    gamma = idata.posterior["gamma"].mean(dim=("chain", "draw"))

    rows = []
    for level, val in zip(gamma_promo["product_type"].to_numpy(), gamma_promo.to_numpy()):
        rows.append({"kind": "promo_type", "key": str(level), "coefficient": float(val)})

    for name, val in zip(gamma["control"].to_numpy(), gamma.to_numpy()):
        name = str(name)
        if name.startswith("ph_"):
            rows.append({"kind": "event_phase", "key": name[len("ph_"):], "coefficient": float(val)})
        elif name.startswith("dow_"):
            rows.append({"kind": "day_of_week", "key": name[len("dow_"):], "coefficient": float(val)})
        # log_search/log_social/log_display/ch_*/reg_* intentionally not exported (see docstring above)

    # pd.get_dummies(..., drop_first=True) drops the alphabetically-first level as the
    # reference category -- its implied coefficient is 0, but it never appears in `gamma`,
    # so it has to be added back explicitly here or it would silently be treated as missing
    # (control_log_effect's dict.get(..., 0.0) fallback) rather than as a real, known zero.
    exported_dow = {r["key"] for r in rows if r["kind"] == "day_of_week"}
    for day in sorted(bayes_agg["day_of_week"].astype(str).unique()):
        if day not in exported_dow:
            rows.append({"kind": "day_of_week", "key": day, "coefficient": 0.0})

    exported_phases = {r["key"] for r in rows if r["kind"] == "event_phase"}
    for phase in sorted(bayes_agg["event_phase"].astype(str).unique()):
        if phase not in exported_phases:
            rows.append({"kind": "event_phase", "key": phase, "coefficient": 0.0})

    promo = bayes_agg["promotion_depth"].astype(float)
    rows.append({"kind": "promo_standardization", "key": "mean", "coefficient": float(promo.mean())})
    rows.append({"kind": "promo_standardization", "key": "std", "coefficient": float(promo.std())})

    return pd.DataFrame(rows)


def build_posterior_samples(idata, meta, n_per_sku: int = N_POSTERIOR_DRAWS_PER_SKU) -> pd.DataFrame:
    """Thin posterior draws of beta_sku (per-SKU elasticity) for the risk-aware optimizer."""
    sku_levels = meta["sku_levels"]
    beta_sku = idata.posterior["beta_sku"].to_numpy()  # (chain, draw, sku)
    n_chain, n_draw, n_sku = beta_sku.shape
    flat = beta_sku.reshape(n_chain * n_draw, n_sku)  # (total_draws, sku)

    rng = np.random.default_rng(RANDOM_SEED)
    total_draws = flat.shape[0]
    idx = rng.choice(total_draws, size=min(n_per_sku, total_draws), replace=False)
    thinned = flat[idx, :]  # (n_per_sku, sku)

    rows = []
    for sku_pos, sku in enumerate(sku_levels):
        for draw_id, val in enumerate(thinned[:, sku_pos]):
            rows.append({"sku": sku, "draw_id": draw_id, "elasticity_draw": float(val)})

    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading full simulated dataset...", flush=True)
    df = load_retail_data(str(DATA_PATH))
    print(f"  {len(df):,} rows loaded ({time.time() - t0:.1f}s)")

    print("Recomputing Phase 2 per-type elasticity (closed-form OLS)...", flush=True)
    phase2_summary = run_elasticity_recovery(df)
    print(f"  done ({time.time() - t0:.1f}s)")

    print("Rebuilding Phase 3 model structure (fast, no sampling) and loading cached posterior...", flush=True)
    import pickle

    bayes_agg = prepare_bayesian_data(df, sample_frac=0.02, random_seed=42)
    _, meta = build_hierarchical_model(bayes_agg)
    with open(IDATA_PATH, "rb") as f:
        idata = pickle.load(f)
    phase3_sku_summary, _ = summarize_bayesian_elasticity(idata, bayes_agg, meta)
    print(f"  done ({time.time() - t0:.1f}s)")

    print("Building sku_master.csv...", flush=True)
    sku_master = build_sku_master(df, phase3_sku_summary, phase2_summary)
    sku_master.to_csv(OUT_DIR / "sku_master.csv", index=False)
    print(f"  {len(sku_master)} rows -> {OUT_DIR / 'sku_master.csv'}")

    print("Building daily_sku_timeseries.csv (this is the slow one)...", flush=True)
    daily = build_daily_sku_timeseries(df)
    daily.to_csv(OUT_DIR / "daily_sku_timeseries.csv", index=False)
    print(f"  {len(daily):,} rows -> {OUT_DIR / 'daily_sku_timeseries.csv'} ({time.time() - t0:.1f}s)")

    print("Building posterior_samples.csv...", flush=True)
    posterior_samples = build_posterior_samples(idata, meta)
    posterior_samples.to_csv(OUT_DIR / "posterior_samples.csv", index=False)
    print(f"  {len(posterior_samples):,} rows -> {OUT_DIR / 'posterior_samples.csv'}")

    print("Building control_effects.csv...", flush=True)
    control_effects = build_control_effects(idata, bayes_agg)
    control_effects.to_csv(OUT_DIR / "control_effects.csv", index=False)
    print(f"  {len(control_effects)} rows -> {OUT_DIR / 'control_effects.csv'}")

    print(f"\nDone. Total elapsed: {time.time() - t0:.1f}s")
    for f in ["sku_master.csv", "daily_sku_timeseries.csv", "posterior_samples.csv", "control_effects.csv"]:
        size_kb = (OUT_DIR / f).stat().st_size / 1024
        print(f"  {f}: {size_kb:,.0f} KB")


if __name__ == "__main__":
    main()
