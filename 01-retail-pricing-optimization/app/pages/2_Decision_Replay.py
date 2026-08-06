"""
Decision Replay -- closed-loop mode (Phase 5d / Modification 2).

Two trajectories over the same window, both starting from the same actual
historical Day-1 inventory: "actual" (what really happened) and "optimizer"
(each day's price comes from the ESTIMATED demand model -- same decision
path as Scenario Explorer, never touches ground truth -- and the REALIZED
outcome comes from the TRUE simulator model via
src/replay_engine.py::realize_true_outcome, the one place in this codebase
allowed to read true_price_elasticity). The optimizer's own Day-N decision
determines Day-(N+1)'s starting inventory -- that's the closed loop.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import load_sku_master, load_daily_timeseries, load_posterior_samples, get_date_bounds, get_elasticity_samples
from replay_engine import run_closed_loop_replay, replay_to_frame
from metrics import format_currency, format_pct
from style import inject_metric_css

st.set_page_config(page_title="Decision Replay -- Nova Retail", page_icon="🔁", layout="wide")
inject_metric_css()

with st.sidebar:
    st.markdown("### Nova Retail")
    st.link_button(
        "View source on GitHub",
        "https://github.com/kpratheek5899/Applied-Data-Science-Portfolio/tree/main/01-retail-pricing-optimization",
        width="stretch",
    )

COLOR_ACTUAL = "#52514e"  # secondary ink
COLOR_OPTIMIZER = "#2a78d6"  # categorical slot 1
COLOR_GRID = "#e1e0d9"
COLOR_SURFACE = "#fcfcfb"
COLOR_STOCKOUT = "#d03b3b"  # status: critical


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

st.title("Decision Replay")
st.info(
    "This closed loop only works because Nova Retail is a synthetic economy with a known true "
    "demand model. The optimizer's day-to-day **decisions** use only the estimated model (same as "
    "Scenario Explorer) -- but the **realized outcome** of each decision is generated from the true "
    "simulator model, which a real business could never actually observe."
)

col1, col2, col3 = st.columns(3)
sku = col1.selectbox(
    "SKU",
    sku_options,
    help="Which product to replay. Both trajectories below (actual and optimizer) follow this one SKU's real history and cost.",
)

# Window length is read (and its widget created) before the start-date
# widget, even though it's laid out in the third column -- Streamlit column
# placement doesn't require statement order to match visual order, and this
# lets the date picker's own max_value depend on whatever window length is
# currently selected, so an invalid combination is structurally impossible
# to pick rather than caught after the fact with an error message.
n_days = col3.slider(
    "Window length (days)",
    3,
    21,
    10,
    key="replay_n_days",
    help="How many consecutive real days to replay. The optimizer re-decides a new price every day in this window, and each day's decision affects the next day's starting inventory (that's the closed loop).",
)

max_start_date = date_hi - pd.Timedelta(days=n_days - 1)
_start_date_key = "replay_start_date"
date_input_kwargs = dict(
    min_value=date_lo.date(),
    max_value=max_start_date.date(),
    key=_start_date_key,
    help=f"Latest pickable start date adjusts automatically so the window never runs past {date_hi.date()}.",
)
if _start_date_key not in st.session_state:
    # First run only: seed an initial default. On later runs, Streamlit
    # won't accept a `value=` outside the *current* min/max even when
    # session_state is meant to take precedence, so it must be omitted
    # once the key exists -- the pre-clamp below handles staying in range.
    date_input_kwargs["value"] = (date_hi - pd.Timedelta(days=13)).date()
elif pd.Timestamp(st.session_state[_start_date_key]) > max_start_date:
    # The window just grew past what the previously-picked start date
    # allows -- clamp before the widget reads it.
    st.session_state[_start_date_key] = max_start_date.date()

start_date = col2.date_input("Start date", **date_input_kwargs)

col4, col5 = st.columns(2)
objective = col4.selectbox(
    "Optimizer objective",
    objective_options,
    format_func=lambda o: objective_labels[o],
    help=(
        "Maximize Profit: highest (price − cost) x units, each day. Maximize Revenue: highest price x "
        "units -- can pick a LOWER price than Maximize Profit when demand is price-sensitive (see the "
        "'why is optimizer profit lower' note below the chart if it looks surprising). Protect "
        "Inventory: maximize profit, but never recommend a price expected to sell more units than the "
        "optimizer trajectory currently has in stock."
    ),
)
use_bayesian = col5.checkbox(
    "Use Bayesian (posterior-based) optimization",
    value=True,
    help=(
        "Draws from Phase 3's fitted posterior over this SKU's elasticity instead of a single point "
        "estimate, every day, so the decision reflects genuine estimation uncertainty rather than one "
        "fixed number."
    ),
)
risk_aversion = st.slider(
    "Risk aversion (Aggressive ←→ Conservative)",
    0.0,
    1.0,
    0.3,
    step=0.05,
    disabled=not use_bayesian,
    help=(
        "Higher = safer: picks prices with a LOWER chance of running out of stock, even if expected "
        "profit/revenue is a bit smaller. Lower = bolder: chases the HIGHEST expected profit/revenue, "
        "even if that means a higher chance of running out of stock. No effect when Optimizer "
        "objective is Protect Inventory -- that objective already blocks stockout risk a different "
        "way (a hard rule, not a trade-off)."
    ),
)
inventory_override = st.number_input(
    "Available inventory override (0 = use Day 1's actual starting inventory)",
    min_value=0.0,
    value=0.0,
    step=10.0,
    help=(
        "Available inventory = how many units the optimizer trajectory actually has in stock to sell. "
        "This is a closed loop -- Day 2 onward already evolves from the model's own decisions, not "
        "history, so this override can only change where Day 1 starts. Type a number here to test 'what "
        "if we started with more/less stock.' 0 = use the real historical Day-1 amount. Only affects the "
        "optimizer trajectory, never the actual (historical) one."
    ),
)

elasticity_samples = get_elasticity_samples(posterior_samples, sku) if use_bayesian else None

try:
    days = run_closed_loop_replay(
        sku_master,
        daily,
        sku,
        start_date,
        n_days,
        objective=objective,
        elasticity_samples=elasticity_samples,
        risk_aversion=risk_aversion,
        inventory_override=inventory_override if inventory_override > 0 else None,
    )
except ValueError as e:
    st.error(str(e))
    st.stop()

df = replay_to_frame(days)

st.divider()

# ---------------------------------------------------------------------------
# Step through days
# ---------------------------------------------------------------------------

day_labels = [d.date.strftime("%Y-%m-%d") for d in days]
selected_label = st.select_slider("Step through the window", options=day_labels, value=day_labels[0])
selected = days[day_labels.index(selected_label)]

st.markdown(f"##### {selected.date.date()}")
d1, d2, d3, d4 = st.columns(4)
d1.metric(
    "Actual price",
    format_currency(selected.actual_price),
    help="What was really charged on this day in the simulated history -- a plain fact, not a model output.",
)
d2.metric(
    "Optimizer price",
    format_currency(selected.optimizer_price),
    format_pct((selected.optimizer_price - selected.actual_price) / selected.actual_price * 100),
    help="What the optimizer would have charged this day, decided from the estimated model only (never the true elasticity). Delta is vs. the actual historical price.",
)
d3.metric(
    "Actual units sold",
    f"{selected.actual_units:,.0f}",
    help="What was really sold on this day in the simulated history.",
)
d4.metric(
    "Optimizer units sold",
    f"{selected.optimizer_units:,.0f}",
    help="Units the optimizer's price would have sold, realized against the true simulator model -- the one place this app is allowed to use ground truth, and only to score the outcome, never to decide the price.",
)

f1, f2, f3, f4 = st.columns(4)
f1.metric(
    "Actual revenue (day)",
    format_currency(selected.actual_revenue),
    help="Actual price x actual units for this one day.",
)
f2.metric(
    "Optimizer revenue (day)",
    format_currency(selected.optimizer_revenue),
    help="price x units for the optimizer's trajectory. Under Maximize Revenue this can beat actual revenue even while optimizer profit falls -- price and margin move in opposite directions for price-sensitive (elastic) demand.",
)
f3.metric(
    "Actual profit (day)",
    format_currency(selected.actual_profit),
    help="(Actual price − cost) x actual units for this one day.",
)
f4.metric(
    "Optimizer profit (day)",
    format_currency(selected.optimizer_profit),
    help="(Optimizer price − cost) x optimizer units for this one day -- compare against Actual profit (day) to see whether the model's decision would have beaten what really happened.",
)

e1, e2 = st.columns(2)
e1.metric(
    "Actual ending inventory",
    f"{selected.actual_ending_inventory:,.0f}",
    help="Units left in stock at the end of this day, in the real historical record.",
)
e2.metric(
    "Optimizer ending inventory",
    f"{selected.optimizer_ending_inventory:,.0f}",
    help="Inventory left over under the optimizer's own trajectory -- this is the closed loop: today's number reflects yesterday's optimizer decision, not the fixed historical inventory.",
)

if selected.optimizer_stockout:
    st.error("Optimizer trajectory stocked out on this day.")

# ---------------------------------------------------------------------------
# Parallel cumulative trajectories -- small multiples (revenue/profit/
# inventory don't share a scale or unit, per the dataviz skill's "no dual
# axis" rule -- each gets its own panel even though revenue and profit are
# both dollars, since plotting 4 lines on one axis would bury the profit
# comparison under the larger revenue numbers).
# ---------------------------------------------------------------------------

st.markdown("##### Actual vs. optimizer-driven pricing over the window")
st.caption(
    "**Cumulative revenue/profit ($):** each day's revenue or profit added to the running total from "
    "day 1 through the day currently selected (dotted vertical line) -- not a daily amount, the "
    "window-to-date sum. **Ending inventory:** units left in stock at the end of each day under that "
    "trajectory's own decisions -- the optimizer's line reflects every one of its own prior days' "
    "pricing choices, not the fixed historical record."
)

fig, (ax_revenue, ax_profit, ax_inventory) = plt.subplots(3, 1, figsize=(9, 9), sharex=True, facecolor=COLOR_SURFACE)

for ax in (ax_revenue, ax_profit, ax_inventory):
    ax.set_facecolor(COLOR_SURFACE)
    ax.grid(color=COLOR_GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(selected.date, color="#898781", linestyle=":", linewidth=1.2)

ax_revenue.plot(df["date"], df["actual_cumulative_revenue"], color=COLOR_ACTUAL, linewidth=2, label="Actual (historical)")
ax_revenue.plot(df["date"], df["optimizer_cumulative_revenue"], color=COLOR_OPTIMIZER, linewidth=2, label="Optimizer-driven")
ax_revenue.set_ylabel("Cumulative revenue ($)")
ax_revenue.legend(loc="best", frameon=False)

ax_profit.plot(df["date"], df["actual_cumulative_profit"], color=COLOR_ACTUAL, linewidth=2)
ax_profit.plot(df["date"], df["optimizer_cumulative_profit"], color=COLOR_OPTIMIZER, linewidth=2)
ax_profit.set_ylabel("Cumulative profit ($)")

ax_inventory.plot(df["date"], df["actual_ending_inventory"], color=COLOR_ACTUAL, linewidth=2)
ax_inventory.plot(df["date"], df["optimizer_ending_inventory"], color=COLOR_OPTIMIZER, linewidth=2)
stockout_days = df[df["optimizer_ending_inventory"] <= 0]
if not stockout_days.empty:
    ax_inventory.scatter(stockout_days["date"], stockout_days["optimizer_ending_inventory"], color=COLOR_STOCKOUT, zorder=5, label="Optimizer stockout")
    ax_inventory.legend(loc="best", frameon=False)
ax_inventory.set_ylabel("Ending inventory")
ax_inventory.set_xlabel("Date")

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

final_actual = df["actual_cumulative_profit"].iloc[-1]
final_optimizer = df["optimizer_cumulative_profit"].iloc[-1]
delta_pct = (final_optimizer - final_actual) / abs(final_actual) * 100 if final_actual != 0 else float("nan")
st.info(
    f"Over these {n_days} days, the optimizer-driven policy would have produced "
    f"{format_currency(final_optimizer)} in cumulative profit vs. {format_currency(final_actual)} actually "
    f"realized ({format_pct(delta_pct)})."
)

with st.expander("Full day-by-day table"):
    st.dataframe(df, hide_index=True, width="stretch")
