"""Shared page styling. Call `inject_metric_css()` once near the top of
each page, right after `st.set_page_config`.
"""

import streamlit as st


def inject_metric_css() -> None:
    """
    Streamlit's default st.metric value text is single-line and truncates
    with an ellipsis once it's wider than its column -- seen for real with
    a long date-range string on the landing page. Column width isn't
    predictable (depends on window size, column count, screen), so no fixed
    string length is ever safe to assume. Structural fix instead of a
    per-value guess: shrink the value font a bit and let it wrap, so a
    too-long value drops to a second line rather than ever getting cut off.
    """
    st.markdown(
        """
        <style>
        [data-testid="stMetricValue"] {
            font-size: 1.6rem;
            white-space: normal;
            overflow-wrap: break-word;
            line-height: 1.25;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
