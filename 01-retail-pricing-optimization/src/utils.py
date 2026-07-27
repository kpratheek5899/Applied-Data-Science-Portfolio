"""
Utility functions for the Retail Pricing & Capacity Optimization Engine.
"""

import pandas as pd


def clean_column_names(df):
    """Convert column names to lowercase snake_case."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


# Columns that repeat a small, fixed set of values across ~3M rows.
# Casting these to `category` avoids storing a fresh Python string object
# per cell, which is what makes a naive read of the full simulated dataset
# blow past a few GB of RAM.
CATEGORY_COLUMNS = [
    "region",
    "store_id",
    "channel",
    "category",
    "sku",
    "product_type",
    "event_name",
    "event_phase",
    "day_of_week",
    "stock_status",
    "stock_message",
]

# Integer columns that only ever hold small values (flags, calendar parts).
SMALL_INT_COLUMNS = [
    "month",
    "week",
    "is_weekend",
    "promotion_flag",
    "email_flag",
    "stockout_flag",
    "lost_sales_flag",
]


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast a Nova Retail dataframe to memory-efficient dtypes in place
    of the pandas defaults (object strings, float64, int64).

    This keeps the ~3M-row simulated dataset within a few hundred MB
    instead of several GB, so it can be read and analyzed on a laptop
    (or a memory-constrained container) without special handling.
    """
    df = df.copy()

    for col in CATEGORY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    for col in SMALL_INT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("int8")

    if "year" in df.columns:
        df["year"] = df["year"].astype("int16")

    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    int_cols = df.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


def load_retail_data(path: str) -> pd.DataFrame:
    """
    Read a Nova Retail simulated CSV with memory-efficient dtypes from the
    start, instead of loading it with pandas' defaults (object strings,
    float64) and downcasting afterwards.

    A plain `pd.read_csv(path)` on the full ~3M-row / 55-column dataset
    briefly needs several GB of RAM while the C parser materializes every
    string cell as its own Python object. Passing `dtype=` up front lets
    the parser build category codes directly, which is the difference
    between this comfortably running on a laptop and it not running at all.
    """
    dtype_map = {col: "category" for col in CATEGORY_COLUMNS}
    dtype_map.update({col: "int8" for col in SMALL_INT_COLUMNS})

    df = pd.read_csv(path, parse_dates=["date"], dtype=dtype_map)

    if "year" in df.columns:
        df["year"] = df["year"].astype("int16")

    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    return df