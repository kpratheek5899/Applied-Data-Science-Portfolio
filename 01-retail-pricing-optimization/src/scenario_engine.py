"""
Predefined and user-defined pricing scenarios for the Scenario Explorer.

MODIFICATION 1 (from the Phase 5 follow-up spec): presets are prefills, not
separate answers. Every function here returns a `Scenario` -- pure input
state (SKU, date/date-range, suggested objective, constraint defaults) --
never a precomputed recommendation. From a `Scenario`, there is exactly one
downstream code path regardless of whether it came from a preset or manual
entry: `demand_model.build_demand_context` -> `demand_model.recommend_price`.

Predefined scenarios are query filters over `data/app/daily_sku_timeseries.csv`
+ `sku_master.csv` (real rows matching each archetype's actual business
condition), not hand-picked/hardcoded values.

Scope note: the app operates at daily, company-wide (all stores/channels
combined) granularity -- `daily_sku_timeseries.csv` was aggregated that way
so a full 2-year date range could stay small enough to ship (see
scripts/build_app_data.py). There's no per-channel selection in this data
model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEFAULT_MAX_PRICE_CHANGE_PCT = 0.5
DEFAULT_MIN_MARGIN = 0.10


@dataclass
class Scenario:
    name: str
    description: str
    sku: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    objective: str
    price_min: float | None = None
    price_max: float | None = None
    min_margin: float | None = DEFAULT_MIN_MARGIN
    max_price_change_pct: float | None = DEFAULT_MAX_PRICE_CHANGE_PCT
    inventory_override: float | None = None


def build_manual_scenario(
    sku: str,
    start_date,
    end_date,
    objective: str,
    price_min: float | None = None,
    price_max: float | None = None,
    min_margin: float | None = DEFAULT_MIN_MARGIN,
    max_price_change_pct: float | None = DEFAULT_MAX_PRICE_CHANGE_PCT,
    inventory_override: float | None = None,
) -> Scenario:
    """User-defined mode: build the same input-state shape a preset would produce."""
    return Scenario(
        name="Custom",
        description="User-defined scenario.",
        sku=sku,
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date),
        objective=objective,
        price_min=price_min,
        price_max=price_max,
        min_margin=min_margin,
        max_price_change_pct=max_price_change_pct,
        inventory_override=inventory_override,
    )


def _row_to_scenario(row: pd.Series, name: str, description: str, objective: str) -> Scenario:
    return Scenario(
        name=name,
        description=description,
        sku=row["sku"],
        start_date=row["date"],
        end_date=row["date"],
        objective=objective,
    )


def _pick_overstock(daily: pd.DataFrame) -> pd.Series:
    normal = daily[daily["event_phase"] == "normal"]
    return normal.sort_values("inventory_pct_remaining", ascending=False).iloc[0]


def _pick_holiday_surge(daily: pd.DataFrame) -> pd.Series:
    # "Cyber Monday" is relabeled "Black Friday Cyber Monday" in
    # scripts/build_app_data.py (the two events' windows overlap in the
    # underlying simulator, and BFCM is standard retail terminology for the
    # combined weekend) -- no standalone "Cyber Monday" label remains.
    candidates = daily[daily["event_name"].isin(["Black Friday", "Black Friday Cyber Monday"])]
    return candidates.sort_values("actual_units", ascending=False).iloc[0]


def _pick_shortage(daily: pd.DataFrame) -> pd.Series:
    candidates = daily[daily["stock_status"].isin(["Limited Availability", "Out Of Stock"]) & (daily["event_phase"] == "event")]
    if candidates.empty:
        candidates = daily[daily["stock_status"].isin(["Limited Availability", "Out Of Stock"])]
    return candidates.sort_values("inventory_pct_remaining").iloc[0]


def _pick_slow_mover(daily: pd.DataFrame) -> pd.Series:
    work = daily.copy()
    work["sku_median_units"] = work.groupby("sku")["actual_units"].transform("median")
    work["units_ratio"] = work["actual_units"] / work["sku_median_units"]
    normal = work[work["event_phase"] == "normal"]
    return normal.sort_values("units_ratio").iloc[0]


def _pick_by_elasticity(sku_master: pd.DataFrame, daily: pd.DataFrame, product_type: str, most_elastic: bool) -> pd.Series:
    of_type = sku_master[sku_master["product_type"] == product_type].sort_values(
        "elasticity_phase3_mean", ascending=most_elastic
    )
    sku = of_type.iloc[0]["sku"]
    sku_rows = daily[(daily["sku"] == sku) & (daily["event_phase"] == "normal") & (daily["stock_status"] == "In Stock")]
    if sku_rows.empty:
        sku_rows = daily[daily["sku"] == sku]
    return sku_rows.iloc[len(sku_rows) // 2]


def build_predefined_scenarios(sku_master: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Scenario]:
    """The 6 preset scenarios, each a real (sku, date) row matching its archetype's actual business condition."""
    return {
        "Inventory Overstock": _row_to_scenario(
            _pick_overstock(daily),
            "Inventory Overstock",
            "Excess inventory on hand relative to normal sell-through, no active event.",
            "maximize_revenue",
        ),
        "Black Friday / Holiday Demand Surge": _row_to_scenario(
            _pick_holiday_surge(daily),
            "Black Friday / Holiday Demand Surge",
            "Peak event-window demand -- the highest single-day unit volume in the dataset.",
            "protect_inventory",
        ),
        "Inventory Shortage": _row_to_scenario(
            _pick_shortage(daily),
            "Inventory Shortage",
            "Limited or out-of-stock during an active demand event.",
            "protect_inventory",
        ),
        "Slow-Moving Product": _row_to_scenario(
            _pick_slow_mover(daily),
            "Slow-Moving Product",
            "Demand well below this SKU's own typical (non-event) volume.",
            "maximize_revenue",
        ),
        "High-Elasticity Commodity Product": _row_to_scenario(
            _pick_by_elasticity(sku_master, daily, "Commodity", most_elastic=True),
            "High-Elasticity Commodity Product",
            "The most price-sensitive Commodity SKU by estimated elasticity.",
            "maximize_profit",
        ),
        "Low-Elasticity Premium Product": _row_to_scenario(
            _pick_by_elasticity(sku_master, daily, "Premium", most_elastic=False),
            "Low-Elasticity Premium Product",
            "The least price-sensitive Premium SKU by estimated elasticity.",
            "maximize_profit",
        ),
    }
