"""
Adaptive Learning demo -- Thompson Sampling (Phase 5f).

Three pricing policies run side by side from the same starting belief,
same starting inventory, same window: a frozen "static" baseline, "Thompson
Sampling" (samples one elasticity draw from its current belief each day,
prices with it, observes the outcome, updates), and "Oracle" (uses the true
elasticity every day -- a labeled upper bound, not something the app claims
to achieve). See src/adaptive_simulation.py's module docstring for the full
design, including two real bugs found while building this (an exponential
price runaway, and a 2D regression that could learn a nonsensical positive
"elasticity") and how they were fixed.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loader import load_sku_master, load_daily_timeseries, get_date_bounds
from adaptive_simulation import run_adaptive_simulation, adaptive_to_frame
from metrics import format_currency
from style import inject_metric_css, chart_info


def _pace_description(first_half_rate: float, second_half_rate: float) -> str:
    if second_half_rate < first_half_rate * 0.8:
        return "slowing down"
    if second_half_rate > first_half_rate * 1.2:
        return "speeding up"
    return "climbing at a fairly steady pace"


def _explain_regret_chart(objective: str, result) -> str:
    n = len(result.thompson)
    half = n // 2 or 1
    static_regret = [d.regret for d in result.static]
    thompson_regret = [d.regret for d in result.thompson]
    static_pace = _pace_description(
        sum(static_regret[:half]) / half, sum(static_regret[half:]) / max(1, n - half)
    )
    thompson_pace = _pace_description(
        sum(thompson_regret[:half]) / half, sum(thompson_regret[half:]) / max(1, n - half)
    )
    static_final = sum(static_regret)
    thompson_final = sum(thompson_regret)
    better_worse = "less" if thompson_final < static_final else "more"

    lines = [
        f"The gray **Static** regret line keeps {static_pace} for the whole window -- it repeats the same "
        "pricing decision every day, so it never corrects course, and the gap to Oracle's zero line just "
        "accumulates.",
        f"The blue **Thompson Sampling** regret line is {thompson_pace}. By the final day it has given up "
        f"{format_currency(thompson_final)} in total, {better_worse} than Static's "
        f"{format_currency(static_final)}.",
    ]

    if objective == "maximize_revenue":
        lines.append(
            "Maximize Revenue has no in-between price for this kind of demand curve -- it's always the "
            "cheapest or priciest price allowed, never something in the middle. If the profit lines above "
            "sit exactly on top of each other, or jump sharply, that's this all-or-nothing choice, not a "
            "data problem."
        )
    elif objective == "protect_inventory":
        lines.append(
            "With Protect Inventory selected, each day's price is raised (if needed) to keep that day's "
            "*expected* units within what's actually in stock. A marked stockout means the day's *realized* "
            "demand still outran available inventory despite that -- the floor uses the model's belief, not "
            "the true demand curve, so it isn't a hard guarantee against a wrong belief."
        )
    else:
        agree = sum(1 for s, o in zip(result.static, result.oracle) if abs(s.price - o.price) < 0.01) / len(
            result.static
        )
        if agree > 0.5:
            lines.append(
                "Static's and Oracle's profit lines above sit exactly on top of each other for most of the "
                "window -- both wanted to charge more than this window's price cap allows, so both simply "
                "hit the same ceiling despite having different opinions about elasticity."
            )

    start_inv = result.thompson[0].starting_inventory
    end_inv = result.thompson[-1].ending_inventory
    inv_change_pct = (end_inv - start_inv) / start_inv * 100 if start_inv else 0
    if end_inv <= 0:
        inv_desc = "runs out entirely by the end of the window"
    elif inv_change_pct < -15:
        inv_desc = f"declines {abs(inv_change_pct):.0f}% over the window"
    elif inv_change_pct > 15:
        inv_desc = f"builds up {inv_change_pct:.0f}% over the window -- selling slower than it's restocked"
    else:
        inv_desc = "holds roughly steady across the window"
    lines.append(f"Thompson Sampling's **ending inventory** {inv_desc}.")
    return "\n\n".join(lines)


def _explain_belief_chart(day1_day, selected_day, true_elasticity: float) -> str:
    day1_width = day1_day.posterior_ci_high - day1_day.posterior_ci_low
    selected_width = selected_day.posterior_ci_high - selected_day.posterior_ci_low
    day1_gap = abs(day1_day.posterior_mean_beta - true_elasticity)
    selected_gap = abs(selected_day.posterior_mean_beta - true_elasticity)

    if selected_width < day1_width * 0.9:
        width_line = (
            f"The blue bar is visibly shorter than the gray Day 1 bar -- the range narrowed from "
            f"{day1_width:.2f} wide to {selected_width:.2f}, meaning the model has ruled out a lot of "
            "elasticity values it originally thought were plausible."
        )
    elif selected_width > day1_width * 1.1:
        width_line = (
            f"The blue bar is actually wider than the gray Day 1 bar ({selected_width:.2f} vs. "
            f"{day1_width:.2f}) -- a single surprising day's outcome can temporarily widen it like this; "
            "the underlying math still gets more confident on average over many days, just not guaranteed "
            "on any one run."
        )
    else:
        width_line = (
            f"The blue bar is about the same width as the gray Day 1 bar ({selected_width:.2f} vs. "
            f"{day1_width:.2f}) -- the model hasn't meaningfully narrowed down its guess yet."
        )

    if selected_gap < day1_gap * 0.8:
        dot_line = "The black dot has also moved closer to the red dashed true-elasticity line since Day 1 -- the best guess is getting more accurate."
    elif selected_gap > day1_gap * 1.2:
        dot_line = "The black dot has actually drifted further from the red dashed true-elasticity line since Day 1 -- this run's evidence has pulled the guess the wrong way so far."
    else:
        dot_line = "The black dot sits about as far from the red dashed true-elasticity line as it did on Day 1."

    lines = [width_line, dot_line]
    if not (selected_day.posterior_ci_low <= true_elasticity <= selected_day.posterior_ci_high):
        lines.append(
            "The red dashed line falls outside the blue bar entirely right now -- the model's current "
            "confidence range doesn't yet include the true answer."
        )
    return "\n\n".join(lines)


def _explain_price_trajectory_chart(objective: str, ts_df) -> str:
    prices = ts_df["price"].to_numpy()
    price_min, price_max = float(prices.min()), float(prices.max())
    swing_pct = (price_max - price_min) / prices[0] * 100 if prices[0] else 0
    n_exploring = int(ts_df["is_exploring"].sum())
    n_total = len(ts_df)

    if prices[-1] > prices[0] * 1.05:
        trend = f"trends upward overall, from ${prices[0]:,.0f} to ${prices[-1]:,.0f}"
    elif prices[-1] < prices[0] * 0.95:
        trend = f"trends downward overall, from ${prices[0]:,.0f} to ${prices[-1]:,.0f}"
    else:
        trend = f"ends close to where it started (${prices[0]:,.0f} vs. ${prices[-1]:,.0f})"

    lines = [
        f"The gray price line {trend}, swinging as wide as ${price_min:,.0f} to ${price_max:,.0f} along the "
        f"way ({swing_pct:.0f}% range).",
        f"Blue dots (**Exploiting**, {n_total - n_exploring} of {n_total} days) mostly sit close to the "
        f"line's recent level. Orange triangles (**Exploring**, {n_exploring} of {n_total} days) are the "
        "deliberate probes that cause the sharper up-and-down jumps -- the cost of learning faster, not "
        "mistakes.",
    ]
    if objective == "maximize_revenue":
        lines.append(
            "Under Maximize Revenue specifically, there's no in-between price -- each day's price is either "
            "the cheapest or priciest one allowed, so an exploring day can mean a big jump between the two, "
            "not a small step."
        )
    return "\n\n".join(lines)

st.set_page_config(page_title="Adaptive Learning -- Nova Retail", page_icon="🧠", layout="wide")
inject_metric_css()

with st.sidebar:
    st.markdown("### Nova Retail")
    st.link_button(
        "View source on GitHub",
        "https://github.com/kpratheek5899/Applied-Data-Science-Portfolio/tree/main/01-retail-pricing-optimization",
        width="stretch",
    )

COLOR_STATIC = "#52514e"  # secondary ink
COLOR_THOMPSON = "#2a78d6"  # categorical slot 1
COLOR_ORACLE = "#0ca30c"  # status good
COLOR_EXPLOIT = "#2a78d6"
COLOR_EXPLORE = "#eb6834"  # categorical slot 2
COLOR_GRID = "#e1e0d9"
COLOR_SURFACE = "#fcfcfb"
COLOR_TRUE = "#d03b3b"  # status critical, reference-only marker
COLOR_STOCKOUT = "#d03b3b"  # status: critical

st.title("Adaptive Learning: Watch the Model Learn")
st.info(
    "This mode starts each SKU's price recommendation from a weak prior belief about its elasticity, "
    "and narrows that belief day by day using **Thompson Sampling**, purely from observing the outcomes "
    "of its own pricing decisions. The decision each day never sees the true elasticity -- only Oracle "
    "(labeled below as *\"if we had known the truth all along\"* -- a theoretical upper bound for "
    "comparison, not something this app claims to achieve) and the outcome-realization step do. "
    "Before each day's outcome updates the belief, promotion, event, and day-of-week effects are removed "
    "from it using effect sizes estimated separately ahead of time, so they don't get mistaken for price "
    "sensitivity -- elasticity itself is the only thing still being learned online."
)

sku_master = load_sku_master()
daily = load_daily_timeseries()
date_lo, date_hi = get_date_bounds(daily)
sku_options = sorted(sku_master["sku"].unique())
objective_options = ["maximize_profit", "maximize_revenue", "protect_inventory"]
objective_labels = {
    "maximize_profit": "Maximize Profit",
    "maximize_revenue": "Maximize Revenue",
    "protect_inventory": "Protect Inventory",
}

col1, col2, col3 = st.columns(3)
sku = col1.selectbox(
    "SKU",
    sku_options,
    index=sku_options.index("SKU_003") if "SKU_003" in sku_options else 0,
    help="Which product all three variants (Static, Thompson Sampling, Oracle) will price, starting from the same day and the same weak belief.",
)
n_days = col2.slider(
    "Window length (days)",
    5,
    30,
    15,
    key="adaptive_n_days",
    help="How many days the simulation runs. Longer windows give Thompson Sampling more days to learn and narrow its belief -- watch how its regret line bends more with a longer window.",
)

max_start_date = date_hi - pd.Timedelta(days=n_days - 1)
_start_key = "adaptive_start_date"
date_kwargs = dict(
    min_value=date_lo.date(),
    max_value=max_start_date.date(),
    key=_start_key,
    help="Bound to the real dataset and the current window length, so you can't pick a start date the window would run past.",
)
if _start_key not in st.session_state:
    date_kwargs["value"] = (date_hi - pd.Timedelta(days=n_days + 30)).date()
elif pd.Timestamp(st.session_state[_start_key]) > max_start_date:
    st.session_state[_start_key] = max_start_date.date()
start_date = col3.date_input("Start date", **date_kwargs)

col4, col5, col6 = st.columns(3)
objective = col4.selectbox(
    "Objective",
    objective_options,
    format_func=lambda o: objective_labels[o],
    help=(
        "Maximize Profit: highest (price − cost) x units, each day. Maximize Revenue: highest price x "
        "units -- can pick a lower price than Maximize Profit when demand is price-sensitive. Protect "
        "Inventory: maximize profit, but never recommend a price expected to sell more units than that "
        "variant currently has in stock. All three variants (Static/Thompson/Oracle) use the same "
        "objective, so the comparison isolates the effect of learning, not a difference in goals."
    ),
)
prior_strength = col5.slider(
    "Starting confidence",
    0.5,
    10.0,
    1.5,
    step=0.5,
    help="Lower = wider, weaker day-1 belief about elasticity -> more room to visibly learn. Higher = starts more confident.",
)
random_seed = col6.number_input(
    "Random seed",
    min_value=0,
    max_value=9999,
    value=7,
    step=1,
    help="Controls every random draw in the simulation (Thompson Sampling's daily elasticity draws). Same seed = same run every time; change it to see a different, equally valid random trajectory.",
)
inventory_override = st.number_input(
    "Available inventory override (0 = use Day 1's actual starting inventory)",
    min_value=0.0,
    value=0.0,
    step=10.0,
    help=(
        "Available inventory = how many units are in stock to sell. All three variants (Static/Thompson/"
        "Oracle) start Day 1 with the same amount -- this override changes that starting amount for all "
        "of them equally, so the comparison stays fair. Every day after Day 1 already evolves from each "
        "variant's own decisions, not history, so only Day 1 can be overridden. 0 = use the real "
        "historical Day-1 amount."
    ),
)

run_clicked = st.button(
    "Run simulation",
    type="primary",
    help="Recomputes and caches the full trajectory for the settings above. Scrubbing the Day slider afterward is free -- it doesn't recompute anything, it just replays the cached result.",
)

cache_key = (sku, str(start_date), n_days, objective, prior_strength, int(random_seed), inventory_override)
if "adaptive_last_key" not in st.session_state:
    st.session_state["adaptive_last_key"] = None

if run_clicked or st.session_state["adaptive_last_key"] is None:
    st.session_state["adaptive_last_key"] = cache_key


@st.cache_data
def _run_cached(sku, start_date, n_days, objective, prior_strength, random_seed, inventory_override):
    return run_adaptive_simulation(
        sku_master,
        daily,
        sku,
        start_date,
        n_days,
        objective=objective,
        prior_strength=prior_strength,
        random_seed=random_seed,
        inventory_override=inventory_override if inventory_override > 0 else None,
    )


try:
    result = _run_cached(*st.session_state["adaptive_last_key"])
except ValueError as e:
    st.error(str(e))
    st.stop()

df = adaptive_to_frame(result)
n_actual_days = len(result.thompson)

if objective == "maximize_revenue":
    st.info(
        "Maximize Revenue has no interior optimum for this kind of demand curve -- the revenue-maximizing "
        "price always sits at one edge of the allowed price range: the ceiling when a belief is between 0 "
        "and -1, the floor when it's more negative than -1. If every variant's belief lands on the same "
        "side of that -1 threshold, their prices -- and therefore profit and regret -- can come out "
        "identical, so lines overlapping below is expected, not a sign learning stopped. If a belief sits "
        "close to -1, day-to-day sampling can also flip the recommended price between the two edges -- "
        "same mechanism, not an error."
    )

st.divider()

# ---------------------------------------------------------------------------
# Step through days
# ---------------------------------------------------------------------------

day_index = (
    st.slider(
        "Day",
        1,
        n_actual_days,
        n_actual_days,
        key="adaptive_day_scrub",
        help="Scrub through the already-simulated window to see the belief and metrics as of that day. This is free -- it replays the cached simulation, it doesn't rerun anything.",
    )
    - 1
)
selected_ts_day = result.thompson[day_index]
day1_ts_day = result.thompson[0]

conf1, conf2, conf3 = st.columns(3)
day1_width = day1_ts_day.posterior_ci_high - day1_ts_day.posterior_ci_low
selected_width = selected_ts_day.posterior_ci_high - selected_ts_day.posterior_ci_low
conf1.metric(
    "Day 1: elasticity 90% CI",
    f"[{day1_ts_day.posterior_ci_low:.2f}, {day1_ts_day.posterior_ci_high:.2f}]",
    f"width {day1_width:.2f}",
    delta_color="off",
    help="The starting (Day 1) 90% credible interval for elasticity, before any learning -- still wide.",
)
conf2.metric(
    f"Day {day_index + 1}: elasticity 90% CI",
    f"[{selected_ts_day.posterior_ci_low:.2f}, {selected_ts_day.posterior_ci_high:.2f}]",
    # The sign has to be the very first character or Streamlit's delta-color logic can't
    # detect it (it checks str(delta).startswith("-"), not the numeric value) -- putting
    # "width"/"vs. Day 1" before the number silently broke this: every delta read as
    # "positive" regardless of its actual sign, so narrowing showed red instead of green.
    f"{selected_width - day1_width:+.2f} width vs. Day 1",
    delta_color="inverse",  # narrower (negative change) is the desired direction
    help="The 90% credible interval as of the currently selected day. A negative width delta means the belief has narrowed -- i.e. genuine learning.",
)
conf3.metric(
    "Day's sampled elasticity",
    f"{selected_ts_day.elasticity_used:.2f}",
    "Exploring" if selected_ts_day.is_exploring else "Exploiting",
    help="The value Thompson Sampling actually drew and priced with today (not the posterior mean). 'Exploring' = the draw deviated notably from the current belief, a deliberate probe rather than a mistake.",
)

# ---------------------------------------------------------------------------
# 1. Cumulative profit / regret / inventory -- small multiples (different
# scales, per the dataviz skill's "no dual axis" rule already used
# elsewhere in this app).
# ---------------------------------------------------------------------------

chart_info(
    "Cumulative profit, regret & inventory: static vs. Thompson Sampling vs. Oracle",
    _explain_regret_chart(objective, result),
)

# Distinct linestyles, not just color: under some objectives (Maximize Revenue especially -- see the
# note above when it's selected) two or three variants can land on the exact same price every day, so
# their lines sit exactly on top of each other. Without a linestyle to tell them apart, whichever one
# is drawn last visually hides the others, which reads as "the other lines are missing" rather than
# "these lines are identical."
variant_style = [
    ("static", COLOR_STATIC, "--", "Static (frozen belief)"),
    ("thompson", COLOR_THOMPSON, "-", "Thompson Sampling"),
    ("oracle", COLOR_ORACLE, ":", "Oracle (true elasticity -- upper bound only, not a real decision-maker)"),
]

fig1, (ax1_profit, ax1_regret, ax1_inventory) = plt.subplots(3, 1, figsize=(9, 10), sharex=True, facecolor=COLOR_SURFACE)
for ax in (ax1_profit, ax1_regret, ax1_inventory):
    ax.set_facecolor(COLOR_SURFACE)
    ax.axvline(day_index + 1, color="#898781", linestyle=":", linewidth=1.2)
    ax.grid(color=COLOR_GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

stockout_labeled = False
for variant, color, linestyle, label in variant_style:
    sub = df[df["variant"] == variant]
    days_axis = range(1, len(sub) + 1)
    ax1_profit.plot(days_axis, sub["cumulative_profit"], color=color, linestyle=linestyle, linewidth=2, label=label)
    ax1_regret.plot(days_axis, sub["cumulative_regret"], color=color, linestyle=linestyle, linewidth=2)
    ax1_inventory.plot(days_axis, sub["ending_inventory"], color=color, linestyle=linestyle, linewidth=2)

    stockout_days = sub[sub["stockout"]]
    if not stockout_days.empty:
        stockout_x = [d for d, is_out in zip(days_axis, sub["stockout"]) if is_out]
        ax1_inventory.scatter(
            stockout_x, stockout_days["ending_inventory"], color=COLOR_STOCKOUT, zorder=5,
            label=None if stockout_labeled else "Stockout",
        )
        stockout_labeled = True

ax1_profit.set_ylabel("Cumulative profit ($)")
ax1_profit.set_title("Cumulative profit")
ax1_profit.legend(loc="upper left", frameon=False, fontsize=8)
ax1_regret.set_ylabel("Cumulative regret ($)")
ax1_regret.set_title("Cumulative regret (lower is better; Oracle = 0 always)")
ax1_inventory.axhline(0, color=COLOR_GRID, linewidth=1.2)
ax1_inventory.set_ylabel("Ending inventory")
ax1_inventory.set_xlabel("Day")
ax1_inventory.set_title("Ending inventory (stockout = demand outran stock on hand)")
if stockout_labeled:
    ax1_inventory.legend(loc="best", frameon=False, fontsize=8)
plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

# ---------------------------------------------------------------------------
# 2. Posterior-narrowing chart: day 1 vs. currently scrubbed day
# ---------------------------------------------------------------------------

true_elasticity = float(
    daily.loc[(daily["sku"] == sku) & (daily["date"] == result.thompson[0].date), "true_price_elasticity"].iloc[0]
)
chart_info(
    f"Belief about elasticity: Day 1 vs. Day {day_index + 1}",
    _explain_belief_chart(day1_ts_day, selected_ts_day, true_elasticity),
)

fig2, ax2 = plt.subplots(figsize=(9, 2.2), facecolor=COLOR_SURFACE)
ax2.set_facecolor(COLOR_SURFACE)
rows = [("Day 1", day1_ts_day, COLOR_STATIC), (f"Day {day_index + 1}", selected_ts_day, COLOR_THOMPSON)]
for y, (label, d, color) in enumerate(rows):
    ax2.plot([d.posterior_ci_low, d.posterior_ci_high], [y, y], color=color, linewidth=5, solid_capstyle="round")
    ax2.plot(d.posterior_mean_beta, y, "o", color="#0b0b0b", markersize=6, zorder=3)
ax2.axvline(true_elasticity, color=COLOR_TRUE, linestyle="--", linewidth=1.2)
ax2.text(true_elasticity, 1.6, "True elasticity\n(reference only)", color=COLOR_TRUE, ha="center", fontsize=8)
ax2.set_yticks([0, 1])
ax2.set_yticklabels([r[0] for r in rows])
ax2.set_xlabel("Elasticity (90% credible interval, dot = posterior mean)")
ax2.set_ylim(-0.6, 2.0)
ax2.grid(color=COLOR_GRID, linewidth=0.8, axis="x")
ax2.spines[["top", "right", "left"]].set_visible(False)
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# ---------------------------------------------------------------------------
# 3. Annotated price trajectory: explore vs. exploit
# ---------------------------------------------------------------------------

ts_df = df[df["variant"] == "thompson"].reset_index(drop=True)
chart_info(
    "Thompson Sampling's price trajectory: exploring vs. exploiting",
    _explain_price_trajectory_chart(objective, ts_df),
)
fig3, ax3 = plt.subplots(figsize=(9, 4), facecolor=COLOR_SURFACE)
ax3.set_facecolor(COLOR_SURFACE)
days_axis = np.arange(1, len(ts_df) + 1)
prices = ts_df["price"].to_numpy()
is_exploring = ts_df["is_exploring"].astype(bool).to_numpy()
ax3.plot(days_axis, prices, color="#c3c2b7", linewidth=1, zorder=1)
ax3.scatter(days_axis[~is_exploring], prices[~is_exploring], color=COLOR_EXPLOIT, s=30, label="Exploiting", zorder=2)
ax3.scatter(days_axis[is_exploring], prices[is_exploring], color=COLOR_EXPLORE, s=30, marker="^", label="Exploring", zorder=2)
ax3.axvline(day_index + 1, color="#898781", linestyle=":", linewidth=1.2)
ax3.set_xlabel("Day")
ax3.set_ylabel("Price ($)")
ax3.grid(color=COLOR_GRID, linewidth=0.8)
ax3.spines[["top", "right"]].set_visible(False)
ax3.legend(loc="best", frameon=False)
plt.tight_layout()
st.pyplot(fig3)
plt.close(fig3)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

totals = df.groupby("variant")[["profit", "regret"]].sum()
s1, s2, s3 = st.columns(3)
s1.metric(
    "Static total profit",
    format_currency(totals.loc["static", "profit"]),
    help="Uses Day 1's frozen belief for every day in the window -- never updates, regardless of what it observes.",
)
s2.metric(
    "Thompson Sampling total profit",
    format_currency(totals.loc["thompson", "profit"]),
    help="Updates its belief every day from the outcome of its own pricing decision -- this is the policy that's actually learning.",
)
s3.metric(
    "Oracle total profit",
    format_currency(totals.loc["oracle", "profit"]),
    help="Uses the TRUE elasticity every day -- a theoretical ceiling for comparison, not a real decision-maker (a real business never gets to see this).",
)
st.caption("Oracle: \"if we had known the truth all along\" -- a theoretical ceiling for comparison, not a real decision-maker.")

with st.expander("Full day-by-day table"):
    st.dataframe(df, hide_index=True, width="stretch")
