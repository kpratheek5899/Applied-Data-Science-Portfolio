"""
Plain-English rationale for a price recommendation.

Rule-based, not hardcoded per scenario: every sentence is composed from the
actual `DemandContext` (elasticity, stock status, event state) and the
actual `optimize_price`/`optimize_price_multi_day` result (price direction,
magnitude, stockout risk, sell-through) for that specific run -- change the
inputs and the explanation changes with them (see tests/test_scenarios.py).
"""

from __future__ import annotations

MAINTAIN_THRESHOLD_PCT = 2.0
INVENTORY_PRESSURE_STATUSES = ("Limited Availability", "Out Of Stock")
LOW_SELL_THROUGH = 0.35


def _direction(price_change_pct: float) -> str:
    if price_change_pct > MAINTAIN_THRESHOLD_PCT:
        return "increase"
    if price_change_pct < -MAINTAIN_THRESHOLD_PCT:
        return "reduce"
    return "maintain"


def generate_explanation(context, result: dict) -> str:
    """
    `context` is a `demand_model.DemandContext`; `result` is the dict
    returned by `demand_model.recommend_price`.
    """
    direction = _direction(result["price_change_pct"])
    objective_label = result["objective"].replace("_", " ")

    if direction == "maintain":
        target_pct = result.get("profit_change_pct", 0.0)
        return (
            f"Maintain price near {result['current_price']:.2f}: the expected {objective_label} "
            f"improvement from changing price is negligible ({target_pct:+.1f}%) given this SKU's "
            f"estimated elasticity of {context.elasticity:.2f}."
        )

    elastic = abs(context.elasticity) > 1.0
    reasons = []

    inventory_pressured = context.stock_status in INVENTORY_PRESSURE_STATUSES or bool(result.get("stockout_risk"))
    event_active = context.event_phase == "event"
    sell_through = result.get("sell_through_rate")
    overstocked = sell_through is not None and sell_through < LOW_SELL_THROUGH

    if direction == "increase":
        if inventory_pressured:
            reasons.append("inventory is limited relative to expected demand")
        if event_active:
            reasons.append(f"{context.event_name} demand is elevated")
        if not elastic:
            reasons.append(
                f"demand for this product is not very price-sensitive (estimated elasticity {context.elasticity:.2f}), "
                "so a higher price increases revenue without a proportional drop in volume"
            )
        elif not reasons:
            reasons.append(
                f"the current price sits below the profit-maximizing level for this product's cost and "
                f"estimated elasticity ({context.elasticity:.2f})"
            )
    else:  # reduce
        if overstocked:
            reasons.append("inventory is high relative to expected sell-through")
        if elastic:
            reasons.append(
                f"demand is price-sensitive (estimated elasticity {context.elasticity:.2f}), so a lower price "
                "drives enough extra volume to increase overall revenue"
            )
        elif not reasons:
            reasons.append(f"the current price sits above the {objective_label}-maximizing level")

    verb = "Increase" if direction == "increase" else "Reduce"
    reason_text = " and ".join(reasons) if reasons else f"it improves expected {objective_label}"

    outcome_clause = ""
    if result["objective"] == "protect_inventory" and result.get("stockout_risk") is False and inventory_pressured:
        outcome_clause = " to reduce stockout risk while preserving revenue"

    return (
        f"{verb} price from {result['current_price']:.2f} to {result['recommended_price']:.2f} "
        f"({result['price_change_pct']:+.1f}%) because {reason_text}{outcome_clause}."
    )
