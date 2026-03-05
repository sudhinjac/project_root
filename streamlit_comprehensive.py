"""
Comprehensive Stock Analysis Streamlit App.

Architecture: Dependency injection, repository pattern, service layer.
References: tech5.py, ALKYLAMINES.ipynb
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from app.repositories.financial_repository import YahooFinancialRepository
from app.repositories.news_repository import GoogleNewsRepository
from app.repositories.stock_repository import StockRepository
from app.services.financial_ratio_service import FinancialRatioService
from app.services.ollama_llm_service import OllamaLLMService
from app.services.sentiment_service import SentimentService
from app.services.technical_analysis_service import TechnicalAnalysisService


# --- Dependency Injection ---
_stock_repo = StockRepository()
_financial_repo = YahooFinancialRepository()
_news_repo = GoogleNewsRepository()
_financial_service = FinancialRatioService(_financial_repo)
_technical_service = TechnicalAnalysisService(_stock_repo)
_sentiment_service = SentimentService(_news_repo)
_llm_service = OllamaLLMService(model=os.getenv("OLLAMA_MODEL", "deepseek-r1:14b"))


def _render_metrics_table(metrics: dict, title: str = "Metrics") -> None:
    """Render metrics as a styled table."""
    rows = [(k, f"{v:.4f}" if isinstance(v, float) else str(v)) for k, v in metrics.items()]
    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_financial_section(ticker: str) -> None:
    """Financial statements and ratios - P&L, Balance Sheet, Cash Flow in tables."""
    result = _financial_service.compute_ratios(ticker)
    if not result:
        st.warning("Financial data not available for this ticker.")
        return

    ratios = result.get("ratios", {})
    if ratios:
        st.subheader("📊 Financial Ratios (ROCE, ROE, etc.)")
        ratio_rows = [(k, v if v is None else f"{v:.2f}") for k, v in ratios.items()]
        st.dataframe(pd.DataFrame(ratio_rows, columns=["Ratio", "Value"]), use_container_width=True, hide_index=True)

    st.subheader("📋 Profit & Loss Statement")
    inc = result.get("income_statement", pd.DataFrame())
    if not inc.empty:
        st.dataframe(inc, use_container_width=True)
    else:
        st.info("No P&L data available.")

    st.subheader("📋 Balance Sheet")
    bs = result.get("balance_sheet", pd.DataFrame())
    if not bs.empty:
        st.dataframe(bs, use_container_width=True)
    else:
        st.info("No balance sheet data available.")

    st.subheader("📋 Cash Flow Statement")
    cf = result.get("cashflow", pd.DataFrame())
    if not cf.empty:
        st.dataframe(cf, use_container_width=True)
    else:
        st.info("No cash flow data available.")


def _render_sentiment_pie(company_query: str):
    """Sentiment analysis with pie chart. Returns headlines for news section."""
    result = _sentiment_service.analyze(company_query)
    if not result or result.total == 0:
        st.warning("No sentiment data available.")
        return []

    cleaned = {k: v for k, v in [
        ("Positive", result.positive),
        ("Neutral", result.neutral),
        ("Negative", result.negative),
    ] if v > 0}

    if not cleaned:
        st.warning("Not enough sentiment data to plot.")
        return result.headlines if result else []

    fig = go.Figure(data=[go.Pie(
        labels=list(cleaned.keys()),
        values=list(cleaned.values()),
        hole=0.4,
        marker_colors=["#22c55e", "#94a3b8", "#ef4444"],
    )])
    fig.update_layout(
        title=f"Sentiment Analysis: {company_query}",
        height=400,
        showlegend=True,
        margin=dict(t=50, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)
    return result.headlines


def _render_news_updates(company_query: str, headlines: list[str]) -> None:
    """Render latest news and updates section - shown last, after Ollama review."""
    st.header("📰 News Updates")
    st.caption("Latest news and important updates")
    if not headlines:
        st.info("No news headlines available.")
        return
    for i, h in enumerate(headlines[:20], 1):
        st.write(f"**{i}.** {h}")


def _render_technical_section(ticker: str) -> None:
    """Technical analysis: metrics, charts, Monte Carlo, forecast, HMM."""
    analysis = _technical_service.get_full_analysis(ticker)
    if not analysis:
        st.error("Insufficient price data for technical analysis.")
        return

    df = analysis.get("df")
    if df is None or df.empty:
        return

    # Metrics tables
    col1, col2 = st.columns(2)
    with col1:
        if analysis.get("price_metrics"):
            st.subheader("📈 Price Metrics")
            _render_metrics_table(analysis["price_metrics"])
    with col2:
        if analysis.get("stock_metrics"):
            st.subheader("📉 Risk Metrics")
            _render_metrics_table(analysis["stock_metrics"])

    # Decision and signals
    st.subheader("🎯 Trade Signals")
    signals = analysis.get("signals", {})
    signal_df = pd.DataFrame(list(signals.items()), columns=["Indicator", "Signal"])
    st.dataframe(signal_df, use_container_width=True, hide_index=True)
    decision = analysis.get("decision", "Hold")
    st.info(f"**Decision:** {decision}")

    # ML prediction
    if analysis.get("ml_prediction"):
        mp = analysis["ml_prediction"]
        st.write(f"**Bullish Probability:** {mp['bullish_probability']:.2%} | **Bearish Probability:** {mp['bearish_probability']:.2%}")

    # Price chart with Bollinger, SMA, Volume
    st.subheader("📉 Price Chart (Bollinger, SMA, Volume)")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["20SMA"], name="20 SMA", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="Upper BB", line=dict(color="green", dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="Lower BB", line=dict(color="red", dash="dot")))
    fig.add_trace(go.Scatter(x=df.index, y=df["50SMA"], name="50 SMA", line=dict(color="purple")))
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", yaxis="y2", opacity=0.5, marker_color="lightgray"))
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500,
        yaxis2=dict(overlaying="y", side="right", showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    # RSI
    st.subheader("📊 RSI")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="orange")))
    fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
    fig_rsi.update_layout(height=300)
    st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD
    st.subheader("📊 MACD")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="purple")))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal", line=dict(color="gray")))
    fig_macd.update_layout(height=300)
    st.plotly_chart(fig_macd, use_container_width=True)

    # Monte Carlo
    mc = analysis.get("monte_carlo", {})
    if mc:
        st.subheader("🔮 Monte Carlo Simulation")
        st.write(f"**Max:** {mc['max_price']:.2f} | **Min:** {mc['min_price']:.2f} | **Mean:** {mc['mean_price']:.2f}")
        st.write(f"**5th Percentile:** {mc['percentile_5']:.2f} | **95th Percentile:** {mc['percentile_95']:.2f}")
        paths = mc.get("price_paths")
        if paths is not None:
            fig_mc = go.Figure()
            for i in range(min(100, paths.shape[1])):
                fig_mc.add_trace(go.Scatter(y=paths[:, i], mode="lines", line=dict(width=0.5), opacity=0.2, showlegend=False))
            fig_mc.update_layout(title="Monte Carlo Paths", xaxis_title="Days", yaxis_title="Price", height=400)
            st.plotly_chart(fig_mc, use_container_width=True)

    # Forecast
    fc = analysis.get("forecast")
    if fc and (fc.get("arima") is not None or fc.get("holt_winters") is not None):
        st.subheader("🔮 Price Forecast (ARIMA vs Holt-Winters)")
        fig_fc = go.Figure()
        hist = fc.get("historical")
        if hist is not None:
            fig_fc.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Historical", line=dict(color="blue")))
        if fc.get("arima") is not None:
            ar = fc["arima"]
            fig_fc.add_trace(go.Scatter(x=ar.index, y=ar.values, name="ARIMA", line=dict(color="green", dash="dot")))
        if fc.get("holt_winters") is not None:
            hw = fc["holt_winters"]
            fig_fc.add_trace(go.Scatter(x=hw.index, y=hw.values, name="Holt-Winters", line=dict(color="orange", dash="dot")))
        fig_fc.update_layout(height=400)
        st.plotly_chart(fig_fc, use_container_width=True)

    # HMM regime
    hmm = analysis.get("hmm")
    if hmm:
        st.subheader("🔍 HMM Regime Detection")
        regime_series = hmm.get("regime_series")
        if regime_series is not None:
            fig_hmm = go.Figure()
            for r in regime_series.unique():
                mask = regime_series == r
                subset = df.loc[mask]
                fig_hmm.add_trace(go.Scatter(x=subset.index, y=subset["Close"], name=f"Regime {r}", mode="lines"))
            fig_hmm.update_layout(height=350)
            st.plotly_chart(fig_hmm, use_container_width=True)
        curr = hmm.get("current_regime", 0)
        st.info(f"**Current Regime:** {curr} | Suggested: {'BUY' if curr == 0 else 'SELL'}")


def _build_financial_summary(ticker: str) -> str:
    """Build text summary of financials for LLM (ratios + key line items from statements)."""
    result = _financial_service.compute_ratios(ticker)
    if not result:
        return "No financial data available."
    parts = []
    ratios = result.get("ratios", {})
    if ratios:
        parts.append("Ratios: " + ", ".join(f"{k}={v}" for k, v in ratios.items() if v is not None))
    for name, df in [
        ("P&L", result.get("income_statement")),
        ("Balance Sheet", result.get("balance_sheet")),
        ("Cash Flow", result.get("cashflow")),
    ]:
        if df is not None and not df.empty:
            col0 = df.iloc[:, 0]
            items = [f"{idx}: {val}" for idx, val in col0.head(8).items() if pd.notna(val)]
            if items:
                parts.append(f"{name} (key items): " + "; ".join(items))
    return "\n\n".join(parts) if parts else "No financial data."


def _build_technical_summary(analysis: Optional[dict]) -> str:
    """Build text summary of technical analysis for LLM."""
    if not analysis:
        return "No technical data available."
    parts = []
    if analysis.get("price_metrics"):
        parts.append("Price metrics: " + str(analysis["price_metrics"]))
    if analysis.get("stock_metrics"):
        parts.append("Risk metrics: " + str(analysis["stock_metrics"]))
    if analysis.get("signals"):
        parts.append("Signals: " + str(analysis["signals"]))
    if analysis.get("decision"):
        parts.append("Decision: " + analysis["decision"])
    return "\n".join(parts) if parts else "No technical summary."


def _build_sentiment_summary(company_query: str) -> str:
    """Build text summary of sentiment for LLM."""
    result = _sentiment_service.analyze(company_query)
    if not result or result.total == 0:
        return "No sentiment data available."
    pct_pos = result.positive / result.total * 100 if result.total else 0
    pct_neg = result.negative / result.total * 100 if result.total else 0
    pct_neu = result.neutral / result.total * 100 if result.total else 0
    return f"Positive: {pct_pos:.1f}%, Neutral: {pct_neu:.1f}%, Negative: {pct_neg:.1f}%"


def main() -> None:
    st.set_page_config(
        page_title="Comprehensive Stock Analysis",
        page_icon="📈",
        layout="wide",
    )
    st.markdown("""
        <style>
        .stApp { background-color: #f8fafc; }
        </style>
    """, unsafe_allow_html=True)

    st.title("📈 Comprehensive Stock Analysis Dashboard")
    st.caption("Enter a ticker for financials, technicals, sentiment, and AI perspective")

    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL", placeholder="e.g. AAPL, MSFT, ALKYLAMINE.NS")
    company_name = st.sidebar.text_input("Company Name (for news/sentiment)", value="", placeholder="e.g. Apple Inc")
    use_ollama = st.sidebar.checkbox("Get AI Perspective (Ollama DeepSeek)", value=False)

    ticker = ticker.strip().upper() if ticker else ""
    company_query = company_name.strip() or ticker

    if not ticker:
        st.info("Enter a ticker symbol to begin.")
        return

    with st.spinner("Loading analysis..."):
        # 1. Financial section
        st.header("🏦 Fundamental Analysis")
        _render_financial_section(ticker)

        st.divider()

        # 2. Technical section
        st.header("📊 Technical Analysis")
        analysis = _technical_service.get_full_analysis(ticker)
        _render_technical_section(ticker)

        st.divider()

        # 3. Sentiment section (pie chart only)
        st.header("📊 Sentiment Analysis")
        headlines = _render_sentiment_pie(company_query)

        st.divider()

        # 4. AI Perspective (Ollama review)
        if use_ollama:
            st.header("🤖 AI Company Perspective (Ollama DeepSeek)")
            with st.spinner("Generating AI analysis..."):
                fin_sum = _build_financial_summary(ticker)
                tech_sum = _build_technical_summary(analysis)
                sent_sum = _build_sentiment_summary(company_query)
                perspective = _llm_service.get_company_perspective(ticker, fin_sum, tech_sum, sent_sum)
            if perspective:
                st.markdown(perspective)
            else:
                st.warning("Ollama/DeepSeek not available. Install ollama and run: ollama pull deepseek")

        st.divider()

        # 5. News Updates (last - after Ollama review)
        _render_news_updates(company_query, headlines or [])


if __name__ == "__main__":
    main()
