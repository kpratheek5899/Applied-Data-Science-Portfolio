"""
Utility functions for the Retail Pricing & Capacity Optimization Engine.
"""


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