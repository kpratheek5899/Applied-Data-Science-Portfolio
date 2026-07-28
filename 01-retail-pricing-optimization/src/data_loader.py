"""
Data loading for the Phase 5 Streamlit app.

Reads only the small, committed files under `data/app/` (built offline by
`scripts/build_app_data.py`) -- never the full ~970MB simulated CSV or the
Phase 3 posterior pickle. This is what keeps the live app deployable: no
regenerating data, no refitting models, just three small CSVs.

Framework-agnostic on purpose (no Streamlit import here) so it can be unit
tested directly; app pages wrap these with `st.cache_data`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_APP_DIR = Path(__file__).resolve().parents[1] / "data" / "app"

# Ground-truth columns that exist in daily_sku_timeseries.csv solely for
# src/replay_engine.py's outcome-realization step. Nothing that recommends
# a price may read these -- see scripts/build_app_data.py's module
# docstring for the full rationale.
OUTCOME_ONLY_COLUMNS = ("true_price_elasticity",)


def load_sku_master(data_dir: Path = DATA_APP_DIR) -> pd.DataFrame:
    """50 rows: static per-SKU attributes + estimated elasticity (Phase 3 primary, Phase 2 fallback)."""
    return pd.read_csv(data_dir / "sku_master.csv")


def load_daily_timeseries(data_dir: Path = DATA_APP_DIR) -> pd.DataFrame:
    """~36.5k rows: (date, sku) daily company-wide aggregates."""
    return pd.read_csv(data_dir / "daily_sku_timeseries.csv", parse_dates=["date"])


def load_posterior_samples(data_dir: Path = DATA_APP_DIR) -> pd.DataFrame:
    """~15k rows, long format: thinned posterior draws of elasticity per SKU."""
    return pd.read_csv(data_dir / "posterior_samples.csv")


def get_date_bounds(daily_df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Real min/max date present in the data -- for the date-range picker, never hardcoded."""
    return daily_df["date"].min(), daily_df["date"].max()


def get_elasticity_samples(posterior_df: pd.DataFrame, sku: str) -> "pd.Series[float]":
    """All thinned posterior draws for one SKU, for the risk-aware optimizer."""
    return posterior_df.loc[posterior_df["sku"] == sku, "elasticity_draw"]


def resolve_elasticity(sku_master_row: pd.Series) -> float:
    """
    Point-estimate elasticity for a SKU: Phase 3's per-SKU posterior mean if
    present, else Phase 2's per-product-type fixed-effects estimate.
    """
    phase3 = sku_master_row.get("elasticity_phase3_mean")
    if pd.notna(phase3):
        return float(phase3)
    return float(sku_master_row["elasticity_phase2"])
