"""
Formatting helpers for the before/after comparison display, built on top of
`demand_model.recommend_price`'s output dict. Keeps app pages free of
formatting logic.
"""

from __future__ import annotations

import pandas as pd


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_pct(value: float) -> str:
    return f"{value:+.1f}%"


def format_units(value: float) -> str:
    return f"{value:,.0f}"


def risk_tier(result: dict) -> str:
    """A short label summarizing stockout risk for the recommended price."""
    if result.get("stockout_risk"):
        return "High -- expected demand meets or exceeds available inventory"
    sell_through = result.get("sell_through_rate")
    if sell_through is None:
        return "N/A -- no inventory constraint supplied"
    if sell_through >= 0.85:
        return "Elevated -- sell-through above 85%"
    if sell_through >= 0.6:
        return "Moderate"
    return "Low"


def before_after_table(result: dict) -> pd.DataFrame:
    """
    Current vs. recommended comparison across price/units/revenue/profit/
    margin, with absolute and percentage changes, ready for `st.dataframe`.
    """
    rows = [
        ("Price", result["current_price"], result["recommended_price"], result["price_change_pct"]),
        ("Units", result["current_units"], result["expected_units"], result["units_change_pct"]),
        ("Revenue", result["current_revenue"], result["revenue"], result["revenue_change_pct"]),
        ("Profit", result["current_profit"], result["profit"], result["profit_change_pct"]),
        (
            "Margin %",
            result["current_margin_pct"] * 100,
            result["margin_pct"] * 100,
            (result["margin_pct"] - result["current_margin_pct"]) * 100,
        ),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Current", "Recommended", "Change"])
    return df


def scenario_summary_line(context, result: dict) -> str:
    """One-line scenario header: SKU, window, objective, stock status."""
    if context.is_multi_day:
        window = f"{context.day_dates[0].date()} to {context.day_dates[-1].date()} ({len(context.day_dates)} days)"
    else:
        window = f"{context.day_dates[0].date()}"
    objective_label = result["objective"].replace("_", " ").title()
    return f"{context.sku} | {window} | Objective: {objective_label} | Stock status: {context.stock_status}"
