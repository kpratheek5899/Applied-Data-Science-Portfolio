"""
Bridges the precomputed data/app tables to src/optimization.py's demand-
curve functions for the app's decision path.

Every price-response curve is anchored at what a real analyst could
actually observe: a day's *realized* price and units (`actual_price`,
`actual_units`), never uncapped/true demand or any ground-truth simulator
parameter. That column-level boundary is what keeps this module's
recommendation path honestly "estimated-only" -- see
`data_loader.py::resolve_elasticity` (Phase 3 posterior mean, else Phase 2)
and `replay_engine.py` (the one place ground truth is allowed to appear,
for outcome realization in Decision Replay).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data_loader import resolve_elasticity
from optimization import optimize_price, optimize_price_multi_day, optimize_price_bayesian


@dataclass
class DemandContext:
    """Everything the optimizer's decision path needs for one SKU over one or more days."""

    sku: str
    cost: float
    elasticity: float
    elasticity_source: str  # "phase3" or "phase2_fallback"
    day_dates: list
    day_prices: list
    day_units: list
    starting_inventory: float
    inventory_capacity: float
    stock_status: str
    event_name: str
    event_phase: str
    promotion_depth: float

    @property
    def is_multi_day(self) -> bool:
        return len(self.day_dates) > 1

    @property
    def current_price(self) -> float:
        """Reference price for display: the window's own most recent actual price."""
        return float(self.day_prices[-1])


def build_demand_context(
    sku_master: pd.DataFrame,
    daily: pd.DataFrame,
    sku: str,
    start_date,
    end_date=None,
) -> DemandContext:
    """
    Resolve everything needed to recommend a price for `sku` over
    [start_date, end_date] (a single day if end_date is None), using only
    realized/estimated fields from data/app/.
    """
    end_date = end_date if end_date is not None else start_date
    start_date, end_date = pd.Timestamp(start_date), pd.Timestamp(end_date)

    window = daily[(daily["sku"] == sku) & (daily["date"] >= start_date) & (daily["date"] <= end_date)].sort_values(
        "date"
    )
    if window.empty:
        raise ValueError(f"No data for {sku} between {start_date.date()} and {end_date.date()}")

    sku_rows = sku_master.loc[sku_master["sku"] == sku]
    if sku_rows.empty:
        raise ValueError(f"Unknown sku {sku!r} -- not present in sku_master")
    sku_row = sku_rows.iloc[0]

    elasticity = resolve_elasticity(sku_row)
    elasticity_source = "phase3" if pd.notna(sku_row.get("elasticity_phase3_mean")) else "phase2_fallback"

    first_day = window.iloc[0]

    return DemandContext(
        sku=sku,
        cost=float(sku_row["cost"]),
        elasticity=elasticity,
        elasticity_source=elasticity_source,
        day_dates=window["date"].tolist(),
        day_prices=window["actual_price"].tolist(),
        day_units=window["actual_units"].tolist(),
        starting_inventory=float(first_day["starting_inventory"]),
        inventory_capacity=float(window["inventory_capacity"].sum()),
        stock_status=str(first_day["stock_status"]),
        event_name=str(first_day["event_name"]),
        event_phase=str(first_day["event_phase"]),
        promotion_depth=float(first_day["promotion_depth"]),
    )


def recommend_price(
    context: DemandContext,
    objective: str,
    inventory: float | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
    min_margin: float | None = None,
    max_price_change_pct: float | None = None,
    n_points: int = 400,
) -> dict:
    """
    Run the optimizer for `context`, dispatching to the single-day or
    multi-day mechanism in `optimization.py` depending on the window size.
    `inventory` defaults to the context's own starting inventory if not
    overridden by the caller (e.g. a user-defined scenario).
    """
    inventory = inventory if inventory is not None else context.starting_inventory

    if context.is_multi_day:
        return optimize_price_multi_day(
            objective,
            day_base_prices=context.day_prices,
            day_base_units=context.day_units,
            cost=context.cost,
            elasticity=context.elasticity,
            inventory=inventory,
            price_min=price_min,
            price_max=price_max,
            min_margin=min_margin,
            max_price_change_pct=max_price_change_pct,
            n_points=n_points,
        )

    return optimize_price(
        objective,
        base_price=context.day_prices[0],
        base_units=context.day_units[0],
        cost=context.cost,
        elasticity=context.elasticity,
        inventory=inventory,
        price_min=price_min,
        price_max=price_max,
        min_margin=min_margin,
        max_price_change_pct=max_price_change_pct,
        n_points=n_points,
    )


def recommend_price_bayesian(
    context: DemandContext,
    objective: str,
    elasticity_samples,
    risk_aversion: float = 0.0,
    inventory: float | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
    min_margin: float | None = None,
    max_price_change_pct: float | None = None,
    n_points: int = 400,
) -> dict:
    """
    Same as `recommend_price`, but using the full posterior over elasticity
    (`elasticity_samples`, from `data_loader.get_elasticity_samples`) instead
    of the point estimate in `context.elasticity` -- Phase 5c. Works
    uniformly for single-day and multi-day contexts (unlike `recommend_price`,
    which dispatches to two different `optimization.py` functions).
    """
    inventory = inventory if inventory is not None else context.starting_inventory

    return optimize_price_bayesian(
        objective,
        day_base_prices=context.day_prices,
        day_base_units=context.day_units,
        cost=context.cost,
        elasticity_samples=elasticity_samples,
        inventory=inventory,
        risk_aversion=risk_aversion,
        price_min=price_min,
        price_max=price_max,
        min_margin=min_margin,
        max_price_change_pct=max_price_change_pct,
        n_points=n_points,
    )
