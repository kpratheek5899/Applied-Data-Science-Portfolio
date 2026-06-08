"""
Data generation module for the Retail Pricing & Capacity Optimization Engine.

This module simulates the first version of Nova Retail's omnichannel retail economy.

Day 3 Simulator v1 includes:
- SKU master table
- Store master table
- Date calendar
- Event calendar
- Product-level elasticity
- Product-level seasonality strength
- Price variation
- Demand generation
- Units, revenue, and gross profit

Inventory, marketing, and capacity will be added in later versions.
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
                    "base_daily_demand": round(base_daily_demand, 2),
                }
            )

            sku_counter += 1

    return pd.DataFrame(rows)


def create_store_master() -> pd.DataFrame:
    """
    Create 20 stores across three regions.
    """
    regions = (
        ["West"] * 7
        + ["Central"] * 6
        + ["East"] * 7
    )

    rows = []

    for i, region in enumerate(regions, start=1):
        rows.append(
            {
                "store_id": f"STORE_{i:02d}",
                "region": region,
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


def add_event_features(calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Add event, pre-event, and post-event effects to the date calendar.
    """
    calendar = calendar.copy()

    calendar["event_name"] = "No Event"
    calendar["event_phase"] = "normal"
    calendar["base_event_multiplier"] = 1.00

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

    return calendar


def generate_retail_data(
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generate Nova Retail simulated data v1.

    Returns
    -------
    pd.DataFrame
        Simulated retail data at date × store × sku × channel grain.
    """
    rng = np.random.default_rng(random_seed)

    sku_master = create_sku_master(random_seed=random_seed)
    store_master = create_store_master()
    calendar = create_date_calendar(start_date=start_date, end_date=end_date)
    calendar = add_event_features(calendar)

    channels = pd.DataFrame(
        {
            "channel": ["Web", "App", "BOPIS", "Store"],
            "channel_multiplier": [1.00, 0.85, 0.70, 1.15],
        }
    )

    # Cross join date × store × sku × channel
    calendar["_key"] = 1
    store_master["_key"] = 1
    sku_master["_key"] = 1
    channels["_key"] = 1

    df = (
        calendar
        .merge(store_master, on="_key")
        .merge(sku_master, on="_key")
        .merge(channels, on="_key")
        .drop(columns="_key")
    )

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

    # Add random price movement around base price
    df["price_index"] = rng.normal(loc=1.0, scale=0.06, size=len(df))
    df["price_index"] = df["price_index"].clip(0.80, 1.20)

    df["price"] = (df["base_price"] * df["price_index"]).round(2)

    # Price effect using log-log elasticity logic:
    # demand_multiplier = (price / base_price) ^ elasticity
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

    # Expected demand
    df["expected_units"] = (
        df["base_daily_demand"]
        * df["channel_multiplier"]
        * df["day_multiplier"]
        * df["region_multiplier"]
        * df["event_multiplier"]
        * df["price_effect"]
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
        "price",
        "cost",
        "true_price_elasticity",
        "seasonality_strength",
        "base_daily_demand",
        "event_name",
        "event_phase",
        "base_event_multiplier",
        "event_multiplier",
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
        "revenue",
        "gross_profit",
    ]

    return df[selected_columns]


if __name__ == "__main__":
    output_path = (
        "01-retail-pricing-optimization/"
        "data/simulated/nova_retail_simulated_data_v2_inventory.csv"
    )

    retail_df = generate_retail_data()
    retail_df.to_csv(output_path, index=False)

    print("Nova Retail simulated dataset with inventory created.")
    print(f"Rows: {len(retail_df):,}")
    print(f"Columns: {len(retail_df.columns):,}")

    print(f"Output: {output_path}")
    print(retail_df.head())