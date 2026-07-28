"""
Optimization module for pricing recommendations.

Given a SKU's current price, cost, expected baseline demand, and price
elasticity (from `src/modeling.py` -- Phase 2's per-product-type estimate or
Phase 3's per-SKU Bayesian posterior mean), recommends a price under one of
three business objectives, subject to margin, price-bound, and inventory
constraints.

Demand follows the same constant-elasticity (log-log) curve the simulator
was built on and Phase 2/3 estimated:

    units(price) = base_units * (price / base_price) ** elasticity

Why grid search instead of a single cvxpy formulation for everything: under
this demand curve, revenue(price) = price * units(price) is a pure monomial
in price -- a textbook geometric-programming (GP) objective, cleanly solved
by cvxpy in `gp=True` mode. But profit(price) = (price - cost) * units(price)
is a monomial *minus* a monomial (a signomial), which is not DCP/DGP-
representable -- cvxpy has no direct way to maximize it. Rather than solve
revenue one way and profit another, `optimize_price` uses grid search as one
general mechanism for all three objectives, which also directly produces the
price-response curve Phase 5's chart needs. `optimize_price_gp` implements
the true GP solution for revenue specifically, and is used in
`tests/test_optimizer.py` to verify grid search converges to the same
answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OBJECTIVES = ("maximize_profit", "maximize_revenue", "protect_inventory")


def constant_elasticity_units(price, base_price: float, base_units: float, elasticity: float):
    """Expected units at `price`, given a base price/demand anchor and elasticity."""
    price = np.asarray(price, dtype=float)
    return base_units * (price / base_price) ** elasticity


def price_response_curve(
    base_price: float,
    base_units: float,
    cost: float,
    elasticity: float,
    price_low: float,
    price_high: float,
    n_points: int = 400,
) -> pd.DataFrame:
    """
    Evaluate units/revenue/profit/margin across a grid of candidate prices.

    This is the same data Phase 5's price-response chart needs; `optimize_price`
    reuses it as the search space instead of a separate numerical solver.
    """
    prices = np.linspace(price_low, price_high, n_points)
    units = constant_elasticity_units(prices, base_price, base_units, elasticity)
    revenue = prices * units
    profit = (prices - cost) * units
    margin_pct = (prices - cost) / prices

    return pd.DataFrame(
        {
            "price": prices,
            "units": units,
            "revenue": revenue,
            "profit": profit,
            "margin_pct": margin_pct,
        }
    )


@dataclass
class PriceBounds:
    low: float
    high: float


def resolve_price_bounds(
    base_price: float,
    cost: float,
    price_min: float | None = None,
    price_max: float | None = None,
    min_margin: float | None = None,
    max_price_change_pct: float | None = None,
    inventory_floor_price: float | None = None,
) -> PriceBounds:
    """
    Combine every constraint source into one feasible [low, high] price
    interval: absolute price bounds, a max %-change limit around the current
    price, a minimum-margin floor, and (for inventory protection) a floor
    price derived from available inventory.

    Unset bounds fall back to sane defaults (never price below cost, never
    more than double the current price) rather than being unconstrained.

    Raises ValueError if the combined constraints leave no feasible price.
    """
    lo_candidates = [
        c
        for c in [
            price_min,
            cost / (1 - min_margin) if min_margin is not None else None,
            base_price * (1 - max_price_change_pct) if max_price_change_pct is not None else None,
            inventory_floor_price,
        ]
        if c is not None
    ]
    hi_candidates = [
        c
        for c in [
            price_max,
            base_price * (1 + max_price_change_pct) if max_price_change_pct is not None else None,
        ]
        if c is not None
    ]

    lo = max(lo_candidates) if lo_candidates else cost * 1.01
    hi = min(hi_candidates) if hi_candidates else base_price * 2.0

    if lo > hi:
        raise ValueError(
            f"Infeasible price bounds: floor {lo:.2f} exceeds ceiling {hi:.2f} "
            "-- check min_margin / price_max / max_price_change_pct together"
        )

    return PriceBounds(low=lo, high=hi)


def _inventory_floor_price(
    base_price: float, base_units: float, elasticity: float, inventory_cap: float
) -> float | None:
    """
    The price at which expected units exactly equal `inventory_cap`.

    Demand is strictly monotonic in price (elasticity < 0), so this is an
    exact closed-form inverse of `constant_elasticity_units`, not a search:
        base_units * (p / base_price) ** elasticity = inventory_cap
        => p = base_price * (inventory_cap / base_units) ** (1 / elasticity)
    Returns None if baseline demand already fits within inventory (no floor
    needed).
    """
    if inventory_cap >= base_units:
        return None
    return base_price * (inventory_cap / base_units) ** (1.0 / elasticity)


def optimize_price(
    objective: str,
    base_price: float,
    base_units: float,
    cost: float,
    elasticity: float,
    inventory: float | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
    min_margin: float | None = None,
    max_price_change_pct: float | None = None,
    n_points: int = 400,
) -> dict:
    """
    Recommend a price under `objective`, subject to margin / price-bound /
    inventory constraints, via grid search over `price_response_curve`.

    - "maximize_profit": pick the grid price with the highest profit.
    - "maximize_revenue": pick the grid price with the highest revenue.
    - "protect_inventory": maximize profit, but first raise the price floor
      to whatever price keeps expected units within `inventory` (demand is
      monotonic in price, so a higher floor can only push the recommended
      price up or leave it unchanged relative to plain profit maximization
      -- exactly the behavior the business objective describes).

    Returns a dict with the recommendation, before/after comparison versus
    the current price, and the full evaluated curve (for visualization).
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
    if elasticity >= 0:
        raise ValueError(f"elasticity must be negative (a valid demand curve), got {elasticity}")
    if base_price <= 0 or base_units <= 0 or cost < 0:
        raise ValueError("base_price and base_units must be positive, cost must be non-negative")

    inventory_floor = None
    if objective == "protect_inventory" and inventory is not None:
        inventory_floor = _inventory_floor_price(base_price, base_units, elasticity, inventory)

    bounds = resolve_price_bounds(
        base_price=base_price,
        cost=cost,
        price_min=price_min,
        price_max=price_max,
        min_margin=min_margin,
        max_price_change_pct=max_price_change_pct,
        inventory_floor_price=inventory_floor,
    )

    curve = price_response_curve(
        base_price, base_units, cost, elasticity, bounds.low, bounds.high, n_points=n_points
    )

    target_col = "revenue" if objective == "maximize_revenue" else "profit"
    best = curve.loc[curve[target_col].idxmax()]

    current_units = float(constant_elasticity_units(base_price, base_price, base_units, elasticity))
    current_revenue = base_price * current_units
    current_profit = (base_price - cost) * current_units
    current_margin_pct = (base_price - cost) / base_price

    recommended_price = float(best["price"])
    recommended_units = float(best["units"])
    recommended_revenue = float(best["revenue"])
    recommended_profit = float(best["profit"])

    ending_inventory = (inventory - recommended_units) if inventory is not None else None
    sell_through_rate = (recommended_units / inventory) if inventory else None
    stockout_risk = bool(inventory is not None and recommended_units >= inventory)

    return {
        "objective": objective,
        "recommended_price": recommended_price,
        "current_price": base_price,
        "price_change_pct": (recommended_price - base_price) / base_price * 100,
        "expected_units": recommended_units,
        "current_units": current_units,
        "units_change_pct": (recommended_units - current_units) / current_units * 100,
        "revenue": recommended_revenue,
        "current_revenue": current_revenue,
        "revenue_change_pct": (recommended_revenue - current_revenue) / current_revenue * 100,
        "profit": recommended_profit,
        "current_profit": current_profit,
        "profit_change_pct": (
            (recommended_profit - current_profit) / abs(current_profit) * 100
            if current_profit != 0
            else float("nan")
        ),
        "margin_pct": float(best["margin_pct"]),
        "current_margin_pct": current_margin_pct,
        "ending_inventory": ending_inventory,
        "sell_through_rate": sell_through_rate,
        "stockout_risk": stockout_risk,
        "price_bounds": (bounds.low, bounds.high),
        "curve": curve,
    }


def optimize_price_gp(
    base_price: float,
    base_units: float,
    elasticity: float,
    price_low: float,
    price_high: float,
) -> float:
    """
    Solve revenue maximization exactly via disciplined geometric programming
    (cvxpy, `gp=True`).

    revenue(p) = p * base_units * (p / base_price) ** elasticity
               = (base_units * base_price ** -elasticity) * p ** (1 + elasticity)

    is a monomial in `p` (a positive constant times `p` raised to a real
    power) -- a textbook GP objective. This exists to verify
    `optimize_price`'s grid search converges to the true optimum for
    "maximize_revenue" (see `tests/test_optimizer.py`); `optimize_price`
    itself does not call this, since it needs one mechanism that also
    covers profit (not GP-representable -- see module docstring).
    """
    import cvxpy as cp

    p = cp.Variable(pos=True)
    coef = base_units * base_price ** (-elasticity)
    revenue = coef * p ** (1 + elasticity)

    problem = cp.Problem(cp.Maximize(revenue), [p >= price_low, p <= price_high])
    problem.solve(gp=True)

    return float(p.value)
