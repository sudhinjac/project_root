"""
Streamlit frontend for Stock Analysis.

Frontend only handles:
- User input (ticker)
- Display of data from API

All computation is done by the FastAPI backend.
"""

import os
import pandas as pd
import streamlit as st
import requests

# Config - API base URL (env var for flexibility)
API_BASE = os.getenv("STOCK_API_URL", "http://localhost:8000")


def fetch_analysis(ticker: str) -> dict | None:
    """Fetch stock analysis from API. Returns None on error."""
    url = f"{API_BASE}/api/v1/analyze/{ticker}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return None


def render_metrics(data: dict) -> None:
    """Render metrics in an elegant table."""
    rows = [
        ("Annual Return (%)", f"{data.get('annual_return', 0):.2f}"),
        ("Volatility (%)", f"{data.get('volatility', 0):.2f}"),
        ("VaR 95% (1-day, %)", f"{data.get('value_at_risk_95', 0):.2f}"),
        ("RSI", f"{data.get('rsi', 0):.1f}"),
        ("MACD", f"{data.get('macd', 0):.4f}"),
        ("MACD Signal", f"{data.get('macd_signal', 0):.4f}"),
        ("MACD Histogram", f"{data.get('macd_histogram', 0):.4f}"),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])

    st.markdown(
        """
        <style>
        .stDataFrame { font-family: 'Segoe UI', sans-serif; }
        div[data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric", width="medium"),
            "Value": st.column_config.TextColumn("Value", width="small"),
        },
    )


def main() -> None:
    st.set_page_config(
        page_title="Stock Analysis",
        page_icon="📊",
        layout="wide",
    )
    st.title("📊 Stock Analysis Dashboard")
    st.caption("Enter a ticker to get annual return, volatility, VaR, RSI, and MACD")

    ticker = st.text_input(
        "Ticker Symbol",
        placeholder="e.g. AAPL, MSFT, GOOGL",
        value="",
    ).strip().upper()

    if st.button("Submit", type="primary"):
        if not ticker:
            st.warning("Please enter a ticker symbol.")
            return

        with st.spinner("Fetching analysis..."):
            data = fetch_analysis(ticker)

        if data:
            st.success(f"Analysis for **{ticker}**")
            render_metrics(data)
        else:
            st.error(f"Could not retrieve data for {ticker}. Check ticker or API connection.")


if __name__ == "__main__":
    main()
