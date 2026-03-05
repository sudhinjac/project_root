"""
Streamlit frontend for Stock Analysis.

Frontend only handles:
- User input (ticker)
- Display of data from API

All computation is done by the FastAPI backend.
"""

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def fetch_indicators(ticker: str) -> dict | None:
    """Fetch RSI and MACD time series from API. Returns None on error."""
    url = f"{API_BASE}/api/v1/analyze/{ticker}/indicators"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return None


def render_metrics(data: dict) -> None:
    """Render metrics table (Annual Return, Volatility, VaR only)."""
    rows = [
        ("Annual Return (%)", f"{data.get('annual_return', 0):.2f}"),
        ("Volatility (%)", f"{data.get('volatility', 0):.2f}"),
        ("VaR 95% (1-day, %)", f"{data.get('value_at_risk_95', 0):.2f}"),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])

    st.markdown(
        """
        <style>
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


def render_charts(ticker: str, indicators: dict) -> None:
    """Render RSI and MACD charts with plotly."""
    dates = pd.to_datetime(indicators["dates"])
    rsi = indicators["rsi"]
    macd = indicators["macd"]
    macd_signal = indicators["macd_signal"]
    macd_histogram = indicators["macd_histogram"]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("RSI (14)", "MACD (12, 26, 9)"),
        row_heights=[0.45, 0.55],
    )

    # RSI
    fig.add_trace(
        go.Scatter(x=dates, y=rsi, name="RSI", line=dict(color="#6366f1", width=2)),
        row=1,
        col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", opacity=0.6, row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", opacity=0.6, row=1, col=1)
    fig.update_yaxes(range=[0, 100], title_text="RSI", row=1, col=1)

    # MACD
    fig.add_trace(
        go.Scatter(x=dates, y=macd, name="MACD", line=dict(color="#0ea5e9", width=2)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=macd_signal, name="Signal", line=dict(color="#f59e0b", width=1.5)),
        row=2,
        col=1,
    )
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in macd_histogram]
    fig.add_trace(
        go.Bar(x=dates, y=macd_histogram, name="Histogram", marker_color=colors),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="MACD", row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(t=40, b=40, l=50, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True,
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Stock Analysis",
        page_icon="📊",
        layout="wide",
    )
    st.title("📊 Stock Analysis Dashboard")
    st.caption("Enter a ticker to get metrics and RSI/MACD charts")

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
            indicators = fetch_indicators(ticker)
            if indicators:
                st.subheader("📈 Technical Indicators")
                render_charts(ticker, indicators)
        else:
            st.error(f"Could not retrieve data for {ticker}. Check ticker or API connection.")


if __name__ == "__main__":
    main()
