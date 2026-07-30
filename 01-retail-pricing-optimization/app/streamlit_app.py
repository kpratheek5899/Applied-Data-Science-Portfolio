"""
Nova Retail Pricing & Capacity Optimization Engine -- landing page.

Framing this app must communicate plainly (per the Phase 5 spec):
Nova Retail is a synthetic business, every recommendation is retrospective/
counterfactual analysis over already-elapsed historical days, and the app
demonstrates a decision-science workflow rather than operating a live
business. See PROGRESS.md / the Phase 5 plan for the full methodology.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_sku_master, load_daily_timeseries, get_date_bounds

st.set_page_config(page_title="Nova Retail Pricing Engine", page_icon="📊", layout="wide")


@st.cache_data
def _load_summary():
    sku_master = load_sku_master()
    daily = load_daily_timeseries()
    lo, hi = get_date_bounds(daily)
    return len(sku_master), lo, hi


n_skus, date_lo, date_hi = _load_summary()

GITHUB_URL = "https://github.com/kpratheek5899/Applied-Data-Science-Portfolio/tree/main/01-retail-pricing-optimization"

with st.sidebar:
    st.markdown("### Nova Retail")
    st.caption("Applied Data Science Portfolio — Project 1")
    st.link_button("View source on GitHub", GITHUB_URL, width="stretch")

st.title("Nova Retail — Pricing & Capacity Optimization Engine")
st.caption(f"Portfolio project by [kpratheek5899]({GITHUB_URL}) — full source on GitHub.")

st.markdown(
    """
    ##### A decision-science workflow, demonstrated end to end on a synthetic omnichannel retailer
    """
)

col1, col2, col3 = st.columns(3)
col1.metric("Simulated observations", "~2.92M")
col2.metric("SKUs covered", f"{n_skus}")
col3.metric("Date range", f"{date_lo.date()} – {date_hi.date()}")

st.divider()

st.markdown(
    """
    ### What this is

    **Nova Retail is a fictional omnichannel retailer** — every transaction
    in this app comes from a synthetic simulator, not a real business. That's
    deliberate: it means demand was generated from **known relationships**
    involving price elasticity, promotions, marketing spend, seasonality,
    retail events, and inventory constraints, so every model in this pipeline
    can be graded against ground truth instead of an unknown black box.

    1. **Simulate** ~2.92M realistic omnichannel transactions across 50 SKUs,
       20 stores, and 4 channels, over two full years.
    2. **Estimate** price elasticity from the observed data — first with
       fixed-effects econometrics, then with a Bayesian hierarchical model
       that partially pools each SKU's elasticity toward its product
       category and quantifies uncertainty with a real posterior.
    3. **Optimize** price under three business objectives (maximize profit,
       maximize revenue, protect inventory), subject to margin, price-bound,
       and inventory constraints.
    4. **Explore**, right here, how those recommendations play out across
       real scenarios pulled from the simulated history.

    ### What every recommendation actually means

    **This app is retrospective, not predictive.** Every scenario is a real
    day (or range of days) that already happened in the simulated 2024-2025
    history. A recommendation answers: *given what the estimated demand
    model would have predicted, what price would have been optimal on this
    already-elapsed day, compared to what was actually charged?* That's
    standard retrospective pricing analysis — the same kind real pricing
    teams run against their own historical data to inform future policy —
    not a live forecast of what to charge tomorrow.

    **The recommendation itself never sees the true, ground-truth demand
    model.** It's built entirely from *estimated* elasticity (the Bayesian
    posterior mean per SKU, falling back to the fixed-effects estimate where
    needed) — the same information a real pricing analyst would actually
    have. The one exception, used nowhere in the recommendation logic, is
    Decision Replay's closed loop: there, and only there, the *true*
    simulator model is used to show what would actually have happened in
    response to a decision — which is only possible because this is a
    synthetic economy. A real business doesn't get to see its own ground
    truth; that gap is exactly why the estimation work in this pipeline
    matters.

    ### Explore
    """
)

st.page_link("pages/1_Scenario_Explorer.py", label="→ Scenario Explorer", icon="🎛️")
st.page_link("pages/2_Decision_Replay.py", label="→ Decision Replay", icon="🔁")
st.page_link("pages/3_Adaptive_Learning.py", label="→ Adaptive Learning (Thompson Sampling)", icon="🧠")

st.caption(
    "Nova Retail is a portfolio project demonstrating a complete applied decision-science "
    "workflow (simulation → econometric and Bayesian estimation → constrained optimization → "
    "decision support), not a production pricing system."
)
