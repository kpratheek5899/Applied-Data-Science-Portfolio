"""
Data generation module for the Retail Pricing & Capacity Optimization Engine.

This module simulates Nova Retail's omnichannel retail economy end to end.

Simulator v3 includes everything from v1/v2 plus:
- Promotion depth (SKU-level, on top of price elasticity effects)
- Marketing spend by channel (search, social, display, email) with
  channel-specific elasticities, scaled up during retail events
- Digital funnel metrics (sessions, page views, conversion rate,
  add-to-cart rate) for digitally-influenced channels
- Store-level operational capacity (fulfillment capacity, labor hours,
  BOPIS capacity, capacity utilization)

Every relationship embedded here is a *known* ground-truth parameter so that
later phases (OLS, Bayesian elasticity recovery, optimization) can be graded
against the truth instead of an unknown black box.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RANDOM_SEED = 42


def create_sku_master(random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Create SKU master table with categories, product types, prices, costs,
    elasticities, and seasonality strength.
    """
    rng = np.random.default_rng(random_seed)

    category_counts = {
        "Electronics": 15,
        "Fitness": 10,
        "Home": 15,
        "Outdoor": 10,
    }

    product_type_elasticity = {
        "Premium": -0.8,
        "Commodity": -2.0,
        "Seasonal": -1.2,
        "Promo Sensitive": -1.8,
    }

    product_type_seasonality = {
        "Premium": 0.60,
        "Commodity": 1.30,
        "Seasonal": 1.50,
        "Promo Sensitive": 1.15,
    }

    # Promotion sensitivity multiplier applied on top of the pure price
    # elasticity effect (Low / Medium / High / Very High from the spec).
    product_type_promo_sensitivity = {
        "Premium": 0.50,
        "Commodity": 1.00,
        "Seasonal": 1.30,
        "Promo Sensitive": 1.60,
    }

    product_type_weights = {
        "Electronics": [0.45, 0.25, 0.10, 0.20],
        "Fitness": [0.20, 0.30, 0.25, 0.25],
        "Home": [0.20, 0.35, 0.25, 0.20],
        "Outdoor": [0.15, 0.25, 0.45, 0.15],
    }

    product_types = ["Premium", "Commodity", "Seasonal", "Promo Sensitive"]

    price_ranges = {
        "Electronics": (80, 1200),
        "Fitness": (25, 800),
        "Home": (20, 600),
        "Outdoor": (30, 1000),
    }

    rows = []
    sku_counter = 1

    for category, count in category_counts.items():
        low_price, high_price = price_ranges[category]

        for _ in range(count):
            product_type = rng.choice(
                product_types,
                p=product_type_weights[category],
            )

            base_price = rng.uniform(low_price, high_price)

            if product_type == "Premium":
                base_price *= rng.uniform(1.15, 1.60)
                margin_pct = rng.uniform(0.35, 0.50)
            elif product_type == "Commodity":
                base_price *= rng.uniform(0.60, 0.95)
                margin_pct = rng.uniform(0.18, 0.32)
            elif product_type == "Seasonal":
                base_price *= rng.uniform(0.80, 1.20)
                margin_pct = rng.uniform(0.25, 0.42)
            else:
                base_price *= rng.uniform(0.75, 1.10)
                margin_pct = rng.uniform(0.22, 0.38)

            cost = base_price * (1 - margin_pct)

            base_daily_demand = rng.uniform(5, 40)

            if product_type == "Commodity":
                base_daily_demand *= rng.uniform(1.2, 2.2)
            elif product_type == "Premium":
                base_daily_demand *= rng.uniform(0.4, 0.9)
            elif product_type == "Seasonal":
                base_daily_demand *= rng.uniform(0.8, 1.5)
            else:
                base_daily_demand *= rng.uniform(0.9, 1.8)

            rows.append(
                {
                    "sku": f"SKU_{sku_counter:03d}",
                    "category": category,
                    "product_type": product_type,
                    "base_price": round(base_price, 2),
                    "cost": round(cost, 2),
                    "true_price_elasticity": product_type_elasticity[product_type],
                    "seasonality_strength": product_type_seasonality[product_type],
                    "promo_sensitivity": product_type_promo_sensitivity[product_type],
                    "base_daily_demand": round(base_daily_demand, 2),
                }
            )

            sku_counter += 1

    return pd.DataFrame(rows)


def create_store_master(random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Create 20 stores across three regions, each with an operational
    capacity profile (fulfillment capacity, labor hours, BOPIS capacity).
    """
    rng = np.random.default_rng(random_seed + 10)

    regions = (
        ["West"] * 7
        + ["Central"] * 6
        + ["East"] * 7
    )

    rows = []

    for i, region in enumerate(regions, start=1):
        # Store "size" drives both inventory profile (already used elsewhere)
        # and operational capacity (units/day the store+labor can fulfill
        # across all channels combined).
        store_size_factor = rng.uniform(0.8, 1.4)

        rows.append(
            {
                "store_id": f"STORE_{i:02d}",
                "region": region,
                # Calibrated so an average store-day (~50 SKUs x 4 channels
                # at ~32 units each ~= 6,400 units/day) runs at roughly
                # 70-85% utilization in normal periods, with event spikes
                # legitimately pushing some stores over capacity -- which
                # is exactly the constraint Phase 4's optimizer needs to see.
                "store_capacity": round(7800 * store_size_factor),
                "labor_hours": round(rng.uniform(220, 420)),
                "bopis_capacity": round(1500 * store_size_factor),
            }
        )

    return pd.DataFrame(rows)


def create_date_calendar(
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """
    Create daily calendar table.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    calendar = pd.DataFrame({"date": dates})
    calendar["year"] = calendar["date"].dt.year
    calendar["month"] = calendar["date"].dt.month
    calendar["week"] = calendar["date"].dt.isocalendar().week.astype(int)
    calendar["day_of_week"] = calendar["date"].dt.day_name()
    calendar["is_weekend"] = calendar["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

    return calendar


def get_event_dates(year: int) -> dict[str, pd.Timestamp]:
    """
    Return major retail event dates for a given year.

    Some are fixed dates. Others are approximations suitable for simulation.
    """
    # Memorial Day = last Monday in May
    may_dates = pd.date_range(f"{year}-05-01", f"{year}-05-31")
    memorial_day = may_dates[(may_dates.day_name() == "Monday")][-1]

    # Labor Day = first Monday in September
    sept_dates = pd.date_range(f"{year}-09-01", f"{year}-09-07")
    labor_day = sept_dates[(sept_dates.day_name() == "Monday")][0]

    # Thanksgiving = fourth Thursday in November
    nov_dates = pd.date_range(f"{year}-11-01", f"{year}-11-30")
    thanksgiving = nov_dates[(nov_dates.day_name() == "Thursday")][3]

    black_friday = thanksgiving + pd.Timedelta(days=1)
    cyber_monday = thanksgiving + pd.Timedelta(days=4)

    return {
        "Memorial Day": memorial_day,
        "Independence Day": pd.Timestamp(f"{year}-07-04"),
        "Labor Day": labor_day,
        "Black Friday": black_friday,
        "Cyber Monday": cyber_monday,
    }


# Marketing spend multiplier applied during the *event* window only. Retail
# events are demand spikes but marketing spend spikes even harder to drive
# awareness ahead of / during the event.
EVENT_MARKETING_MULTIPLIER = {
    "Memorial Day": 1.50,
    "Independence Day": 1.40,
    "Labor Day": 1.60,
    "Black Friday": 3.00,
    "Cyber Monday": 2.50,
    "Holiday Season": 2.00,
}


def add_event_features(calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Add event, pre-event, and post-event effects to the date calendar.
    """
    calendar = calendar.copy()

    calendar["event_name"] = "No Event"
    calendar["event_phase"] = "normal"
    calendar["base_event_multiplier"] = 1.00
    calendar["marketing_event_multiplier"] = 1.00

    event_multipliers = {
        "Memorial Day": 1.25,
        "Independence Day": 1.20,
        "Labor Day": 1.30,
        "Black Friday": 2.50,
        "Cyber Monday": 2.00,
    }

    pre_event_multiplier = 0.90
    post_event_multiplier = 0.85

    years = calendar["year"].unique()

    for year in years:
        event_dates = get_event_dates(int(year))

        for event_name, event_date in event_dates.items():
            event_start = event_date - pd.Timedelta(days=3)
            event_end = event_date + pd.Timedelta(days=3)

            pre_start = event_start - pd.Timedelta(days=14)
            pre_end = event_start - pd.Timedelta(days=1)

            post_start = event_end + pd.Timedelta(days=1)
            post_end = event_end + pd.Timedelta(days=14)

            pre_mask = calendar["date"].between(pre_start, pre_end)
            event_mask = calendar["date"].between(event_start, event_end)
            post_mask = calendar["date"].between(post_start, post_end)

            calendar.loc[pre_mask, "event_name"] = event_name
            calendar.loc[pre_mask, "event_phase"] = "pre_event"
            calendar.loc[pre_mask, "base_event_multiplier"] = pre_event_multiplier

            calendar.loc[event_mask, "event_name"] = event_name
            calendar.loc[event_mask, "event_phase"] = "event"
            calendar.loc[event_mask, "base_event_multiplier"] = event_multipliers[event_name]
            calendar.loc[event_mask, "marketing_event_multiplier"] = (
                EVENT_MARKETING_MULTIPLIER[event_name]
            )

            calendar.loc[post_mask, "event_name"] = event_name
            calendar.loc[post_mask, "event_phase"] = "post_event"
            calendar.loc[post_mask, "base_event_multiplier"] = post_event_multiplier

    # Holiday season is treated separately because it is a longer season.
    holiday_mask = (
        (calendar["date"].dt.month == 12)
        & (calendar["date"].dt.day <= 24)
    )

    calendar.loc[holiday_mask, "event_name"] = "Holiday Season"
    calendar.loc[holiday_mask, "event_phase"] = "event"
    calendar.loc[holiday_mask, "base_event_multiplier"] = 1.75
    calendar.loc[holiday_mask, "marketing_event_multiplier"] = (
        EVENT_MARKETING_MULTIPLIER["Holiday Season"]
    )

    return calendar


# ---------------------------------------------------------------------------
# Marketing spend
# ---------------------------------------------------------------------------

# Channel elasticities from the Nova Retail specification:
# "10% increase in search spend -> 2% increase in units" => elasticity 0.20
MARKETING_ELASTICITY = {
    "search": 0.20,
    "social": 0.08,
    "display": 0.05,
}
EMAIL_LIFT = 0.15  # flat demand lift on days with an active email campaign

CATEGORY_BASE_SPEND = {
    # (search, social, display) daily base spend per region-category
    "Electronics": (900, 400, 300),
    "Fitness": (400, 250, 150),
    "Home": (500, 260, 180),
    "Outdoor": (350, 200, 140),
}


def create_marketing_table(
    calendar: pd.DataFrame,
    categories: list[str],
    regions: list[str],
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate daily marketing spend by (date, region, category).

    Marketing budgets are set at the region/category level (not per SKU),
    matching how retail marketing teams actually allocate spend. Spend
    scales up automatically during event windows via
    `marketing_event_multiplier`.
    """
    rng = np.random.default_rng(random_seed + 20)

    cal = calendar[
        ["date", "marketing_event_multiplier"]
    ].copy()
    cal["_key"] = 1

    grid = pd.DataFrame(
        [(r, c) for r in regions for c in categories],
        columns=["region", "category"],
    )
    grid["_key"] = 1

    table = cal.merge(grid, on="_key").drop(columns="_key")

    base_spend = table["category"].map(
        lambda c: CATEGORY_BASE_SPEND[c]
    )
    table["base_search_spend"] = base_spend.map(lambda t: t[0])
    table["base_social_spend"] = base_spend.map(lambda t: t[1])
    table["base_display_spend"] = base_spend.map(lambda t: t[2])

    n = len(table)

    for col in ["search", "social", "display"]:
        noise = rng.lognormal(mean=0, sigma=0.15, size=n)
        table[f"{col}_spend"] = (
            table[f"base_{col}_spend"]
            * table["marketing_event_multiplier"]
            * noise
        ).round(2)

    # Email campaigns: ~12% of days, more likely during event windows.
    email_prob = np.where(table["marketing_event_multiplier"] > 1.0, 0.35, 0.12)
    table["email_flag"] = (rng.random(n) < email_prob).astype(int)

    # Combined marketing demand multiplier (applied on top of price/promo
    # effects). Uses base spend as the reference point for elasticity.
    table["marketing_effect"] = (
        (table["search_spend"] / table["base_search_spend"]) ** MARKETING_ELASTICITY["search"]
        * (table["social_spend"] / table["base_social_spend"]) ** MARKETING_ELASTICITY["social"]
        * (table["display_spend"] / table["base_display_spend"]) ** MARKETING_ELASTICITY["display"]
        * (1 + table["email_flag"] * EMAIL_LIFT)
    )

    return table[
        [
            "date",
            "region",
            "category",
            "search_spend",
            "social_spend",
            "display_spend",
            "email_flag",
            "marketing_effect",
        ]
    ]


# ---------------------------------------------------------------------------
# Promotions
# ---------------------------------------------------------------------------

PROMOTION_DEPTHS = np.array([0.00, 0.05, 0.10, 0.15, 0.20])

# "10% promotion -> +18% demand" (on top of the pure price-elasticity effect
# already captured by the lower selling price) => lift factor of 1.8 per
# unit of depth, scaled by each product type's promo sensitivity.
PROMO_LIFT_FACTOR = 1.8


def create_promotion_table(
    calendar: pd.DataFrame,
    sku_master: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Assign a promotion depth to every (date, sku) combination.

    Promotions are more frequent for promo-sensitive product types and
    during event windows, matching real markdown/promo calendars.
    """
    rng = np.random.default_rng(random_seed + 30)

    cal = calendar[["date", "event_phase"]].copy()
    cal["_key"] = 1

    skus = sku_master[["sku", "promo_sensitivity"]].copy()
    skus["_key"] = 1

    table = cal.merge(skus, on="_key").drop(columns="_key")
    n = len(table)

    # Base probability of being on any promotion at all on a given day.
    base_promo_prob = 0.10 + 0.10 * (table["promo_sensitivity"] - 0.5) / 1.1
    event_boost = np.where(table["event_phase"] == "event", 1.8, 1.0)
    promo_prob = np.clip(base_promo_prob * event_boost, 0, 0.85)

    on_promo = rng.random(n) < promo_prob

    # Deeper discounts more likely during event windows.
    depth_weights_normal = np.array([0.55, 0.20, 0.15, 0.07, 0.03])
    depth_weights_event = np.array([0.15, 0.20, 0.25, 0.22, 0.18])

    depths = np.zeros(n)
    is_event = (table["event_phase"] == "event").to_numpy()

    depths[is_event] = rng.choice(
        PROMOTION_DEPTHS, size=is_event.sum(), p=depth_weights_event
    )
    depths[~is_event] = rng.choice(
        PROMOTION_DEPTHS, size=(~is_event).sum(), p=depth_weights_normal
    )

    table["promotion_depth"] = np.where(on_promo, depths, 0.0)
    table["promotion_flag"] = (table["promotion_depth"] > 0).astype(int)

    table["promo_lift"] = 1 + (
        table["promotion_depth"] * PROMO_LIFT_FACTOR * table["promo_sensitivity"]
    )

    return table[["date", "sku", "promotion_depth", "promotion_flag", "promo_lift"]]


# ---------------------------------------------------------------------------
# Digital funnel
# ---------------------------------------------------------------------------

DIGITAL_CHANNELS = ("Web", "App", "BOPIS")

CHANNEL_VIEWS_PER_SESSION = {
    "Web": 4.5,
    "App": 6.0,
    "BOPIS": 3.5,
}

CHANNEL_BASE_CONVERSION = {
    "Web": 0.028,
    "App": 0.035,
    "BOPIS": 0.05,
}

STOCK_STATUS_CONVERSION_MULTIPLIER = {
    "In Stock": 1.00,
    "Low Stock": 0.90,
    "Limited Availability": 0.70,
    "Out Of Stock": 0.15,
}


def add_digital_features(df: pd.DataFrame, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Derive session-level digital funnel metrics for digitally-influenced
    channels (Web, App, BOPIS) from realized units, so the funnel is
    internally consistent with actual sales instead of an independent,
    disconnected random variable.

    Physical Store channel rows get null digital metrics (no session
    concept for walk-in traffic).
    """
    rng = np.random.default_rng(random_seed + 40)

    df = df.copy()
    is_digital = df["channel"].isin(DIGITAL_CHANNELS)

    conversion_base = df["channel"].map(CHANNEL_BASE_CONVERSION).fillna(np.nan)
    stock_multiplier = df["stock_status"].map(STOCK_STATUS_CONVERSION_MULTIPLIER).fillna(1.0)

    conversion_noise = rng.lognormal(mean=0, sigma=0.12, size=len(df))
    conversion_rate = (
        conversion_base
        * stock_multiplier
        * df["marketing_effect"].clip(lower=0.5, upper=2.5) ** 0.3
        * conversion_noise
    ).clip(upper=0.35)

    # Sessions implied by units and conversion rate, plus a small floor of
    # "window shopping" traffic that never converts.
    browsing_floor = rng.uniform(3, 15, size=len(df))
    sessions = (df["units"] / conversion_rate.replace(0, np.nan)) + browsing_floor
    sessions = sessions.round().fillna(0)

    conversion_rate_realized = np.where(
        sessions > 0, df["units"] / sessions, 0
    )

    views_per_session = df["channel"].map(CHANNEL_VIEWS_PER_SESSION).fillna(4.0)
    page_views = (sessions * views_per_session * rng.uniform(0.85, 1.15, size=len(df))).round()

    add_to_cart_rate = np.clip(
        conversion_rate_realized * rng.uniform(1.8, 2.6, size=len(df)), 0, 0.9
    )

    df["sessions"] = np.where(is_digital, sessions, np.nan)
    df["page_views"] = np.where(is_digital, page_views, np.nan)
    df["conversion_rate"] = np.where(is_digital, conversion_rate_realized, np.nan)
    df["add_to_cart_rate"] = np.where(is_digital, add_to_cart_rate, np.nan)

    return df


def _generate_chunk(
    calendar_chunk: pd.DataFrame,
    store_master: pd.DataFrame,
    sku_master: pd.DataFrame,
    channels: pd.DataFrame,
    marketing_table: pd.DataFrame,
    promotion_table: pd.DataFrame,
    chunk_seed: int,
) -> pd.DataFrame:
    """
    Run the full row-level generation pipeline for a single date-range
    chunk (e.g. one month) and return the finished, dtype-optimized rows.

    Generation is chunked by date so that peak memory during the
    date x store x sku x channel cross-join and all downstream row-wise
    math stays bounded (~100-150k rows per chunk) regardless of how many
    total days/SKUs/stores the simulator is configured for. This mirrors
    how the same workload would be partitioned in a Spark/Databricks
    pipeline rather than materializing one monolithic in-memory table.
    """
    rng = np.random.default_rng(chunk_seed)

    # Cross join date x store x sku x channel for this chunk only.
    df = (
        calendar_chunk
        .merge(store_master, how="cross")
        .merge(sku_master, how="cross")
        .merge(channels, how="cross")
    )

    # Bring in marketing spend (date x region x category) and promotions
    # (date x sku).
    df = df.merge(marketing_table, on=["date", "region", "category"], how="left")
    df = df.merge(promotion_table, on=["date", "sku"], how="left")

    # Weekly pattern
    day_multipliers = {
        "Monday": 0.90,
        "Tuesday": 0.92,
        "Wednesday": 0.95,
        "Thursday": 1.00,
        "Friday": 1.10,
        "Saturday": 1.25,
        "Sunday": 1.15,
    }

    df["day_multiplier"] = df["day_of_week"].map(day_multipliers)

    # Product-level seasonality adjustment:
    # Premium products react less to event multipliers.
    # Commodity and Seasonal products react more.
    df["event_multiplier"] = 1 + (
        (df["base_event_multiplier"] - 1)
        * df["seasonality_strength"]
    )

    # Keep event multiplier from becoming negative during pre/post slumps
    df["event_multiplier"] = df["event_multiplier"].clip(lower=0.60)

    # Add random price movement around base price, then apply the
    # promotion discount on top to get the final selling price.
    df["price_index"] = rng.normal(loc=1.0, scale=0.06, size=len(df))
    df["price_index"] = df["price_index"].clip(0.80, 1.20)

    df["list_price"] = (df["base_price"] * df["price_index"]).round(2)
    df["price"] = (df["list_price"] * (1 - df["promotion_depth"])).round(2)

    # Price effect using log-log elasticity logic on the *final* selling
    # price, so promotional price cuts flow through true elasticity too.
    df["price_effect"] = (
        df["price"] / df["base_price"]
    ) ** df["true_price_elasticity"]

    # Random demand noise
    df["noise"] = rng.lognormal(mean=0, sigma=0.20, size=len(df))

    # Region multiplier
    region_multipliers = {
        "West": 1.08,
        "Central": 0.95,
        "East": 1.02,
    }

    df["region_multiplier"] = df["region"].map(region_multipliers)

    # Expected demand: price/promo/marketing effects are layered on top of
    # the baseline seasonal/regional/channel pattern.
    df["expected_units"] = (
        df["base_daily_demand"]
        * df["channel_multiplier"]
        * df["day_multiplier"]
        * df["region_multiplier"]
        * df["event_multiplier"]
        * df["price_effect"]
        * df["promo_lift"]
        * df["marketing_effect"]
        * df["noise"]
    )

    # Potential demand generated from a Poisson process
    df["demand_units"] = rng.poisson(lam=df["expected_units"].clip(lower=0.1))

    # Inventory capacity is based on product demand profile.
    # Commodity and promo-sensitive products receive deeper inventory positions.
    product_inventory_multiplier = {
        "Premium": 5,
        "Commodity": 7,
        "Seasonal": 6,
        "Promo Sensitive": 6,
    }

    channel_inventory_multiplier = {
        "Web": 1.00,
        "App": 0.75,
        "BOPIS": 0.90,
        "Store": 1.20,
    }

    df["inventory_capacity"] = (
        df["base_daily_demand"]
        * df["product_type"].map(product_inventory_multiplier)
        * df["channel"].map(channel_inventory_multiplier)
    ).round().astype(int)

    df["inventory_capacity"] = df["inventory_capacity"].clip(lower=20)

    # Weekly replenishment approximation:
    # Each week starts with replenished inventory near 80% of capacity.
    # Event periods receive higher planned inventory, but demand may still exceed supply.
    event_inventory_boost = np.where(df["event_phase"] == "event", 1.25, 1.00)

    df["starting_inventory"] = (
        df["inventory_capacity"]
        * 0.6
        * event_inventory_boost
        * rng.uniform(0.85, 1.10, size=len(df))
    ).round().astype(int)

    df["starting_inventory"] = df[["starting_inventory", "inventory_capacity"]].min(axis=1)

    # Actual units sold cannot exceed available inventory.
    df["units"] = np.minimum(df["demand_units"], df["starting_inventory"])

    # Lost sales represent unmet demand due to inventory constraints.
    df["lost_sales"] = df["demand_units"] - df["units"]

    df["ending_inventory"] = df["starting_inventory"] - df["units"]

    df["inventory_pct_remaining"] = (
        df["ending_inventory"] / df["inventory_capacity"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    # Stock status derived from inventory remaining.
    conditions = [
        df["ending_inventory"] <= 0,
        df["inventory_pct_remaining"] <= 0.15,
        df["inventory_pct_remaining"] <= 0.40,
        df["inventory_pct_remaining"] > 0.40,
    ]

    choices = [
        "Out Of Stock",
        "Limited Availability",
        "Low Stock",
        "In Stock",
    ]

    df["stock_status"] = np.select(
        conditions,
        choices,
        default="Unknown",
    )

    stock_message_map = {
        "In Stock": "Available Today",
        "Low Stock": "Low Stock",
        "Limited Availability": "Only A Few Left",
        "Out Of Stock": "Out Of Stock",
    }

    df["stock_message"] = df["stock_status"].map(stock_message_map)

    df["stockout_flag"] = (df["stock_status"] == "Out Of Stock").astype(int)
    df["lost_sales_flag"] = (df["lost_sales"] > 0).astype(int)

    df["revenue"] = (df["units"] * df["price"]).round(2)
    df["gross_profit"] = (df["units"] * (df["price"] - df["cost"])).round(2)

    # Digital funnel metrics (depends on final units + stock_status +
    # marketing_effect, so must run after those are finalized).
    df = add_digital_features(df, random_seed=chunk_seed)

    # Operational capacity utilization: total units fulfilled per store per
    # day, relative to that store's fulfillment capacity.
    store_daily_units = (
        df.groupby(["date", "store_id"])["units"]
        .transform("sum")
    )
    df["fulfillment_capacity"] = df["store_capacity"]
    df["capacity_utilization"] = (
        store_daily_units / df["fulfillment_capacity"]
    ).round(4)

    # Clean ordering
    selected_columns = [
        "date",
        "year",
        "month",
        "week",
        "day_of_week",
        "region",
        "store_id",
        "channel",
        "category",
        "sku",
        "product_type",
        "base_price",
        "list_price",
        "price",
        "cost",
        "true_price_elasticity",
        "seasonality_strength",
        "promo_sensitivity",
        "base_daily_demand",
        "event_name",
        "event_phase",
        "base_event_multiplier",
        "event_multiplier",
        "promotion_depth",
        "promotion_flag",
        "promo_lift",
        "search_spend",
        "social_spend",
        "display_spend",
        "email_flag",
        "marketing_effect",
        "price_index",
        "price_effect",
        "expected_units",
        "demand_units",
        "units",
        "lost_sales",
        "inventory_capacity",
        "starting_inventory",
        "ending_inventory",
        "inventory_pct_remaining",
        "stock_status",
        "stock_message",
        "stockout_flag",
        "lost_sales_flag",
        "sessions",
        "page_views",
        "conversion_rate",
        "add_to_cart_rate",
        "store_capacity",
        "fulfillment_capacity",
        "labor_hours",
        "bopis_capacity",
        "capacity_utilization",
        "revenue",
        "gross_profit",
    ]

    chunk_df = df[selected_columns]

    try:
        from utils import optimize_dtypes
    except ImportError:  # running as a package (src.utils) rather than a script
        from .utils import optimize_dtypes

    return optimize_dtypes(chunk_df)


def iter_retail_data_chunks(
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    random_seed: int = RANDOM_SEED,
    verbose: bool = False,
):
    """
    Generate the full Nova Retail simulated dataset (v3), one
    calendar-month chunk at a time (see `_generate_chunk`).

    This is a generator so callers can stream chunks straight to disk
    (see the `__main__` block) instead of holding the full ~3M row / 55
    column dataset in memory at once. Each yielded chunk is already
    dtype-optimized (see `utils.optimize_dtypes`).

    Yields
    ------
    pd.DataFrame
        One month's worth of rows at date x store x sku x channel grain,
        including commercial, promotional, marketing, digital, inventory,
        and operational capacity variables.
    """
    sku_master = create_sku_master(random_seed=random_seed)
    store_master = create_store_master(random_seed=random_seed)
    calendar = create_date_calendar(start_date=start_date, end_date=end_date)
    calendar = add_event_features(calendar)

    categories = sorted(sku_master["category"].unique())
    regions = sorted(store_master["region"].unique())

    marketing_table = create_marketing_table(
        calendar, categories, regions, random_seed=random_seed
    )
    promotion_table = create_promotion_table(
        calendar, sku_master, random_seed=random_seed
    )

    channels = pd.DataFrame(
        {
            "channel": ["Web", "App", "BOPIS", "Store"],
            "channel_multiplier": [1.00, 0.85, 0.70, 1.15],
        }
    )

    month_starts = pd.period_range(
        start=calendar["date"].min(), end=calendar["date"].max(), freq="M"
    )

    for i, period in enumerate(month_starts):
        month_mask = (
            (calendar["date"].dt.year == period.year)
            & (calendar["date"].dt.month == period.month)
        )
        calendar_chunk = calendar.loc[month_mask].copy()

        if calendar_chunk.empty:
            continue

        if verbose:
            print(f"Generating {period} ({len(calendar_chunk)} days)...", flush=True)

        chunk = _generate_chunk(
            calendar_chunk=calendar_chunk,
            store_master=store_master.drop(columns=["_key"], errors="ignore"),
            sku_master=sku_master.drop(columns=["_key"], errors="ignore"),
            channels=channels,
            marketing_table=marketing_table,
            promotion_table=promotion_table,
            chunk_seed=random_seed + i,
        )
        yield chunk


def generate_retail_data(
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    random_seed: int = RANDOM_SEED,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generate the full Nova Retail simulated dataset (v3) as a single
    in-memory DataFrame.

    Convenience wrapper around `iter_retail_data_chunks` for small/medium
    date ranges (e.g. notebook exploration on a few months of data). For
    the full ~2 year / ~3M row dataset, prefer running this module as a
    script, which streams chunks straight to CSV instead of holding
    everything in memory at once.
    """
    chunks = list(
        iter_retail_data_chunks(
            start_date=start_date,
            end_date=end_date,
            random_seed=random_seed,
            verbose=verbose,
        )
    )
    return pd.concat(chunks, ignore_index=True)


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Generate Nova Retail simulated data, streaming chunks to CSV."
    )
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--mode",
        default="w",
        choices=["w", "a"],
        help="'w' to start a fresh file, 'a' to append to an existing one "
        "(useful for generating a large date range across multiple runs).",
    )
    parser.add_argument(
        "--output",
        default=(
            "01-retail-pricing-optimization/"
            "data/simulated/nova_retail_simulated_data_v3_full.csv"
        ),
    )
    args = parser.parse_args()

    start = time.time()
    total_rows = 0

    for i, chunk in enumerate(
        iter_retail_data_chunks(start_date=args.start, end_date=args.end, verbose=True)
    ):
        is_first_write = (args.mode == "w") and (i == 0)
        chunk.to_csv(
            args.output,
            mode="w" if is_first_write else "a",
            header=is_first_write,
            index=False,
        )
        total_rows += len(chunk)
        print(
            f"  -> wrote {len(chunk):,} rows "
            f"(running total {total_rows:,}, {time.time() - start:,.1f}s elapsed)",
            flush=True,
        )

    print(f"Nova Retail simulated dataset chunk range [{args.start}, {args.end}] done.")
    print(f"Total rows written this run: {total_rows:,}")
    print(f"Output: {args.output}")
    print(f"Elapsed: {time.time() - start:,.1f}s")
