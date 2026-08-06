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
from style import inject_metric_css

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

st.title("Adaptive Learning: Watch the Model Learn")
st.info(
    "This mode starts each SKU's price recommendation from a weak prior belief about its elasticity, "
    "and narrows that belief day by day using **Thompson Sampling**, purely from observing the outcomes "
    "of its own pricing decisions. The decision each day never sees the true elasticity -- only Oracle "
    "(labeled below as *\"if we had known the truth all along\"* -- a theoretical upper bound for "
    "comparison, not something this app claims to achieve) and the outcome-realization step do. "
    "Before each day's outcome updates the belief, promotion, event, and day-of-week effects are removed "
    "from it using effect sizes already estimated in Phases 2-3, so they don't get mistaken for price "
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
    f"width {selected_width - day1_width:+.2f} vs. Day 1",
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
# 1. Cumulative profit / regret race chart -- small multiples (different
# scales, per the dataviz skill's "no dual axis" rule already used
# elsewhere in this app).
# ---------------------------------------------------------------------------

st.markdown("##### Cumulative profit & regret: static vs. Thompson Sampling vs. Oracle")
st.caption(
    "Regret = profit under Oracle's price that day minus profit under the variant's own price. Oracle's "
    "regret is exactly zero by construction (its own price *is* the day's true-optimal price). A working "
    "bandit algorithm's regret should visibly bend (sub-linear) as it learns, while a frozen belief's "
    "regret keeps growing roughly linearly."
)

variant_style = [
    ("static", COLOR_STATIC, "Static (frozen belief)"),
    ("thompson", COLOR_THOMPSON, "Thompson Sampling"),
    ("oracle", COLOR_ORACLE, "Oracle (true elasticity -- upper bound only, not a real decision-maker)"),
]

fig1, (ax1_profit, ax1_regret) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, facecolor=COLOR_SURFACE)
for ax in (ax1_profit, ax1_regret):
    ax.set_facecolor(COLOR_SURFACE)
    ax.axvline(day_index + 1, color="#898781", linestyle=":", linewidth=1.2)
    ax.grid(color=COLOR_GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

for variant, color, label in variant_style:
    sub = df[df["variant"] == variant]
    ax1_profit.plot(range(1, len(sub) + 1), sub["cumulative_profit"], color=color, linewidth=2, label=label)
    ax1_regret.plot(range(1, len(sub) + 1), sub["cumulative_regret"], color=color, linewidth=2)

ax1_profit.set_ylabel("Cumulative profit ($)")
ax1_profit.set_title("Cumulative profit")
ax1_profit.legend(loc="upper left", frameon=False, fontsize=8)
ax1_regret.set_ylabel("Cumulative regret ($)")
ax1_regret.set_xlabel("Day")
ax1_regret.set_title("Cumulative regret (lower is better; Oracle = 0 always)")
plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

# ---------------------------------------------------------------------------
# 2. Posterior-narrowing chart: day 1 vs. currently scrubbed day
# ---------------------------------------------------------------------------

st.markdown("##### Belief about elasticity: Day 1 vs. Day " + str(day_index + 1))
true_elasticity = float(
    daily.loc[(daily["sku"] == sku) & (daily["date"] == result.thompson[0].date), "true_price_elasticity"].iloc[0]
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

st.markdown("##### Thompson Sampling's price trajectory: exploring vs. exploiting")
st.caption(
    "A day is marked \"exploring\" when the sampled elasticity that day deviated more than 0.5 posterior "
    "standard deviations from the current belief's mean -- a deliberate probe, not a mistake."
)

ts_df = df[df["variant"] == "thompson"].reset_index(drop=True)
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
