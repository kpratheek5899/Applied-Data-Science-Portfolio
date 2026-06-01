"""
Streamlit app for Retail Pricing & Capacity Optimization Engine.
"""

import streamlit as st


st.set_page_config(
    page_title="Retail Pricing Optimization Engine",
    layout="wide"
)

st.title("Retail Pricing & Capacity Optimization Engine")

st.write(
    """
    This dashboard will present pricing recommendations, elasticity estimates,
    and operational constraint scenarios for Nova Retail.
    """
)