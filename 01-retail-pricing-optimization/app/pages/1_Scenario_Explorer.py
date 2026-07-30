"""
Scenario Explorer -- the core Phase 5 page.

Modification 1 (presets are prefills, not separate answers): selecting a
preset only changes the *default values* of the same input widgets manual
mode uses (via Streamlit widget keys scoped to the preset name, which reset
the displayed value on switch while still allowing free editing). From that
point there is exactly one code path: scenario_engine.Scenario ->
demand_model.build_demand_context -> demand_model.recommend_price. Nothing
downstream of that first step knows or cares whether the inputs came from a
preset or manual entry.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from data_loader import (
    load_sku_master,
    load_daily_timeseries,
    load_posterior_samples,
    get_date_bounds,
    get_elasticity_samples,
)
from scenario_engine import build_predefined_scenarios, build_manual_scenario
from demand_model import build_demand_context, recommend_price, recommend_price_bayesian
from explanations import generate_explanation
from metrics import before_after_table, risk_tier, scenario_summary_line, format_currency, format_pct

st.set_page_config(page_title="Scenario Explorer -- Nova Retail", page_icon="🎛️", layout="wide")

with st.sidebar:
    st.markdown("### Nova Retail")
    st.caption("Applied Data Science Portfolio — Project 1")
    st.link_button(
        "View source on GitHub",
        "https://github.com/kpratheek5899/Applied-Data-Science-Portfolio/tree/main/01-retail-pricing-optimization",
        width="stretch",
    )

# Palette (see .claude skill "dataviz" reference/palette.md) -- categorical
# slots assigned in fixed order, reference lines use ink/status colors, not
# additional categorical hues.
COLOR_PROFIT = "#2a78d6"  # categorical slot 1 (blue)
COLOR_REVENUE = "#eb6834"  # categorical slot 2 (orange)
COLOR_UNITS = "#2a78d6"
COLOR_CURRENT = "#52514e"  # secondary ink
COLOR_RECOMMENDED = "#0ca30c"  # status: good
COLOR_GRID = "#e1e0d9"
COLOR_SURFACE = "#fcfcfb"
COLOR_FEASIBLE_BAND = "#cde2fb"  # sequential step 100


@st.cache_data
def _load_data():
    return load_sku_master(), load_daily_timeseries(), load_posterior_samples()


sku_master, daily, posterior_samples = _load_data()
date_lo, date_hi = get_date_bounds(daily)
sku_options = sorted(sku_master["sku"].unique())
objective_options = ["maximize_profit", "maximize_revenue", "protect_inventory"]
objective_labels = {
    "maximize_profit": "Maximize Profit",
    "maximize_revenue": "Maximize Revenue",
    "protect_inventory": "Protect Inventory",
}

st.title("Scenario Explorer")
st.caption(
    "Retrospective analysis over already-elapsed simulated days -- see the landing page for what "
    "that means and what the optimizer is (and isn't) allowed to see."
)

# ---------------------------------------------------------------------------
# Step 1: choose a starting point (preset or custom) -- prefill only
# ---------------------------------------------------------------------------

presets = build_predefined_scenarios(sku_master, daily)
starting_point = st.selectbox("Starting point", ["Custom"] + list(presets.keys()))

if starting_point == "Custom":
    prefill_sku = sku_options[0]
    prefill_start = date_hi
    prefill_end = date_hi
    prefill_objective = "maximize_profit"
    st.caption("Manual mode -- set every input yourself.")
else:
    scenario = presets[starting_point]
    prefill_sku = scenario.sku
    prefill_start = scenario.start_date
    prefill_end = scenario.end_date
    prefill_objective = scenario.objective
    st.caption(scenario.description)

key_suffix = starting_point.replace(" ", "_")

# ---------------------------------------------------------------------------
# Step 2: inputs (same widgets regardless of preset vs custom)
# ---------------------------------------------------------------------------

input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    sku = st.selectbox("SKU", sku_options, index=sku_options.index(prefill_sku), key=f"sku_{key_suffix}")

with input_col2:
    use_range = st.checkbox(
        "Use a date range", value=(prefill_start != prefill_end), key=f"use_range_{key_suffix}"
    )

with input_col3:
    objective = st.selectbox(
        "Objective",
        objective_options,
        index=objective_options.index(prefill_objective),
        format_func=lambda o: objective_labels[o],
        key=f"objective_{key_suffix}",
    )

if use_range:
    date_range = st.date_input(
        "Date range",
        value=(prefill_start.date(), prefill_end.date()),
        min_value=date_lo.date(),
        max_value=date_hi.date(),
        key=f"date_range_{key_suffix}",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        start_date = end_date = pd.Timestamp(date_range if not isinstance(date_range, tuple) else date_range[0])
    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        st.stop()
else:
    single_date = st.date_input(
        "Date",
        value=prefill_start.date(),
        min_value=date_lo.date(),
        max_value=date_hi.date(),
        key=f"single_date_{key_suffix}",
    )
    start_date = end_date = pd.Timestamp(single_date)

with st.expander("Constraints (optional)"):
    c1, c2, c3, c4 = st.columns(4)
    price_min = c1.number_input("Min price ($)", min_value=0.0, value=0.0, step=1.0, key=f"pmin_{key_suffix}")
    price_max = c2.number_input("Max price ($, 0 = no cap)", min_value=0.0, value=0.0, step=1.0, key=f"pmax_{key_suffix}")
    min_margin = c3.slider("Min margin", 0.0, 0.9, 0.10, step=0.01, key=f"margin_{key_suffix}")
    max_change = c4.slider("Max price change", 0.05, 2.0, 0.50, step=0.05, key=f"maxchange_{key_suffix}")
    inventory_override = st.number_input(
        "Available inventory override (0 = use this day's actual starting inventory)",
        min_value=0.0,
        value=0.0,
        step=10.0,
        key=f"inventory_{key_suffix}",
        help="Lower this to explore a tighter stock position than what actually happened historically.",
    )

st.markdown("##### Uncertainty")
bayes_col, risk_col = st.columns([1, 2])
use_bayesian = bayes_col.checkbox(
    "Use Bayesian (posterior-based) optimization",
    value=True,
    key=f"bayes_{key_suffix}",
    help=(
        "Draws from Phase 3's fitted posterior over this SKU's elasticity instead of a single "
        "point estimate, and optimizes expected profit/revenue net of stockout risk."
    ),
)
risk_aversion = risk_col.slider(
    "Risk aversion (Aggressive ←→ Conservative)",
    0.0,
    1.0,
    0.3,
    step=0.05,
    key=f"risk_{key_suffix}",
    disabled=not use_bayesian,
    help="Higher values trade expected profit/revenue for a lower probability of stocking out.",
)

# ---------------------------------------------------------------------------
# Step 3: exactly one code path from here, regardless of input source
# ---------------------------------------------------------------------------

scenario = build_manual_scenario(
    sku=sku,
    start_date=start_date,
    end_date=end_date,
    objective=objective,
    price_min=price_min if price_min > 0 else None,
    price_max=price_max if price_max > 0 else None,
    min_margin=min_margin,
    max_price_change_pct=max_change,
    inventory_override=inventory_override if inventory_override > 0 else None,
)

try:
    context = build_demand_context(sku_master, daily, scenario.sku, scenario.start_date, scenario.end_date)
    if use_bayesian:
        elasticity_samples = get_elasticity_samples(posterior_samples, context.sku)
        if len(elasticity_samples) == 0:
            st.warning(f"No posterior draws for {context.sku} -- falling back to the point estimate.")
            result = recommend_price(
                context,
                scenario.objective,
                inventory=scenario.inventory_override,
                price_min=scenario.price_min,
                price_max=scenario.price_max,
                min_margin=scenario.min_margin,
                max_price_change_pct=scenario.max_price_change_pct,
            )
        else:
            result = recommend_price_bayesian(
                context,
                scenario.objective,
                elasticity_samples=elasticity_samples,
                risk_aversion=risk_aversion,
                inventory=scenario.inventory_override,
                price_min=scenario.price_min,
                price_max=scenario.price_max,
                min_margin=scenario.min_margin,
                max_price_change_pct=scenario.max_price_change_pct,
            )
    else:
        result = recommend_price(
            context,
            scenario.objective,
            inventory=scenario.inventory_override,
            price_min=scenario.price_min,
            price_max=scenario.price_max,
            min_margin=scenario.min_margin,
            max_price_change_pct=scenario.max_price_change_pct,
        )
except ValueError as e:
    st.error(str(e))
    st.stop()

st.divider()
st.subheader(scenario_summary_line(context, result))

# ---------------------------------------------------------------------------
# Before / after comparison
# ---------------------------------------------------------------------------

col_before, col_after = st.columns([2, 1])

with col_before:
    table = before_after_table(result)

    def _format_row(row: pd.Series) -> pd.Series:
        if row["Metric"] == "Margin %":
            return pd.Series(
                {
                    "Metric": row["Metric"],
                    "Current": f"{row['Current']:.1f}%",
                    "Recommended": f"{row['Recommended']:.1f}pp",
                    "Change": f"{row['Change']:+.1f}pp",
                }
            )
        if row["Metric"] == "Units":
            return pd.Series(
                {
                    "Metric": row["Metric"],
                    "Current": f"{row['Current']:,.0f}",
                    "Recommended": f"{row['Recommended']:,.0f}",
                    "Change": format_pct(row["Change"]),
                }
            )
        return pd.Series(
            {
                "Metric": row["Metric"],
                "Current": format_currency(row["Current"]),
                "Recommended": format_currency(row["Recommended"]),
                "Change": format_pct(row["Change"]),
            }
        )

    display_table = table.apply(_format_row, axis=1)
    st.dataframe(display_table, hide_index=True, width='stretch')

with col_after:
    st.metric("Recommended price", format_currency(result["recommended_price"]), format_pct(result["price_change_pct"]))
    st.metric("Expected profit", format_currency(result["profit"]), format_pct(result["profit_change_pct"]))
    st.caption(f"Stockout risk: {risk_tier(result)}")

st.info(generate_explanation(context, result))

# ---------------------------------------------------------------------------
# Price-response chart -- small multiples (profit/revenue share a $ axis;
# units gets its own panel, per the dataviz skill's "no dual axis" rule).
# ---------------------------------------------------------------------------

curve = result["curve"]
bounds_lo, bounds_hi = result["price_bounds"]

fig, (ax_dollars, ax_units) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, facecolor=COLOR_SURFACE)

for ax in (ax_dollars, ax_units):
    ax.set_facecolor(COLOR_SURFACE)
    ax.axvspan(bounds_lo, bounds_hi, color=COLOR_FEASIBLE_BAND, alpha=0.4, zorder=0, label="Feasible range")
    ax.axvline(result["current_price"], color=COLOR_CURRENT, linestyle="--", linewidth=1.5, zorder=3)
    ax.axvline(result["recommended_price"], color=COLOR_RECOMMENDED, linestyle="-", linewidth=1.5, zorder=3)
    ax.grid(color=COLOR_GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

ax_dollars.plot(curve["price"], curve["profit"], color=COLOR_PROFIT, linewidth=2, label="Profit")
ax_dollars.plot(curve["price"], curve["revenue"], color=COLOR_REVENUE, linewidth=2, label="Revenue")
ax_dollars.set_ylabel("$")
ax_dollars.set_title("Profit & Revenue vs. Price")
ax_dollars.legend(loc="best", frameon=False)

ax_units.plot(curve["price"], curve["units"], color=COLOR_UNITS, linewidth=2)
ax_units.set_ylabel("Units")
ax_units.set_xlabel("Price ($)")
ax_units.set_title("Expected Units vs. Price")

ax_units.text(
    result["current_price"], ax_units.get_ylim()[1] * 0.95, "Current", color=COLOR_CURRENT, ha="center", fontsize=9
)
ax_units.text(
    result["recommended_price"],
    ax_units.get_ylim()[1] * 0.85,
    "Recommended",
    color=COLOR_RECOMMENDED,
    ha="center",
    fontsize=9,
)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------------------------
# Outcome uncertainty at the recommended price (Bayesian mode only) --
# the posterior draws behind the mean curve above, condensed to a p10-p50-p90
# range per metric rather than swept across the whole price axis.
# ---------------------------------------------------------------------------

if use_bayesian and "profit_distribution" in result:
    st.markdown("##### Likely outcome range at the recommended price")
    st.caption(
        f"Across {len(elasticity_samples)} posterior draws of this SKU's elasticity "
        f"(stockout probability at the recommended price: {result['stockout_probability']:.1%})."
    )

    fig2, (ax_dollars_dist, ax_units_dist) = plt.subplots(2, 1, figsize=(9, 3), facecolor=COLOR_SURFACE)

    dollar_rows = [("Profit", result["profit_distribution"]), ("Revenue", result["revenue_distribution"])]
    for ax, rows, label_color in ((ax_dollars_dist, dollar_rows, COLOR_PROFIT), (ax_units_dist, [("Units", result["units_distribution"])], COLOR_UNITS)):
        ax.set_facecolor(COLOR_SURFACE)
        y_positions = list(range(len(rows)))
        for y, (label, dist) in zip(y_positions, rows):
            ax.plot([dist["p10"], dist["p90"]], [y, y], color=label_color, linewidth=4, solid_capstyle="round", zorder=2)
            ax.plot(dist["p50"], y, "o", color=COLOR_RECOMMENDED, markersize=7, zorder=3)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.grid(color=COLOR_GRID, linewidth=0.8, axis="x")
        ax.spines[["top", "right", "left"]].set_visible(False)

    ax_dollars_dist.set_xlabel("$")
    ax_units_dist.set_xlabel("Units")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

with st.expander("Underlying price-response data"):
    st.dataframe(curve, width='stretch')
