"""Build HTML/PDF report from stock analysis data."""

import html as html_module
from datetime import datetime
from typing import Any, Optional

import pandas as pd


def _df_to_html(df: pd.DataFrame, title: str = "") -> str:
    """Convert DataFrame to styled HTML table."""
    if df is None or df.empty:
        return f"<h3>{title}</h3><p>No data available.</p>"
    html = df.to_html(classes="report-table", index=True, escape=False)
    return f"<h3>{title}</h3>{html}"


def _fmt_val(v: Any) -> str:
    """Format value for HTML display."""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 0.01 or abs(v) > 10000 else f"{v:.2f}"
    return str(v)


def _dict_to_html(d: dict, title: str = "") -> str:
    """Convert dict to HTML table."""
    if not d:
        return f"<h3>{title}</h3><p>No data available.</p>"
    rows = "".join(
        f"<tr><td>{html_module.escape(str(k))}</td><td>{_fmt_val(v)}</td></tr>"
        for k, v in d.items()
    )
    return f"""<h3>{title}</h3>
    <table class="report-table"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>"""


def build_html_report(
    ticker: str,
    company_query: str,
    fin_result: Optional[dict],
    analysis: Optional[dict],
    sentiment_result: Optional[Any],
    headlines: list[str],
    ai_perspective: str = "",
) -> str:
    """Build full HTML report from analysis data."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Analysis Report - {ticker}</title>
    <style>
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 2rem; max-width: 900px; margin-inline: auto; }}
        h1 {{ color: #0ea5e9; }}
        h2 {{ color: #38bdf8; margin-top: 2rem; border-bottom: 1px solid #334155; padding-bottom: 0.25rem; }}
        h3 {{ color: #94a3b8; margin-top: 1rem; }}
        .report-table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
        .report-table th, .report-table td {{ border: 1px solid #334155; padding: 0.5rem 0.75rem; text-align: left; }}
        .report-table th {{ background: #1e293b; color: #38bdf8; }}
        .meta {{ color: #64748b; font-size: 0.9rem; margin-bottom: 2rem; }}
        .section {{ margin-bottom: 2rem; }}
        .ai-perspective {{ background: #f1f5f9; padding: 1rem; border-radius: 8px; white-space: pre-wrap; }}
    </style>
</head>
<body>
    <h1>Stock Analysis Report: {ticker}</h1>
    <p class="meta">Company: {company_query} | Generated: {now}</p>
"""]

    # Financial section
    parts.append('<div class="section"><h2>Fundamental Analysis</h2>')
    if fin_result:
        ratios = fin_result.get("ratios", {})
        if ratios:
            ratio_d = {k: v for k, v in ratios.items() if v is not None}
            parts.append(_dict_to_html(ratio_d, "Financial Ratios"))
        for name, key in [("Profit & Loss", "income_statement"), ("Balance Sheet", "balance_sheet"), ("Cash Flow", "cashflow")]:
            df = fin_result.get(key)
            if df is not None and not df.empty:
                parts.append(_df_to_html(df, name))
    else:
        parts.append("<p>No financial data available.</p>")
    parts.append("</div>")

    # Technical section
    parts.append('<div class="section"><h2>Technical Analysis</h2>')
    if analysis:
        unified = []
        if fin_result:
            ratios = fin_result.get("ratios", {})
            for k in ["ROE (%)", "ROCE (%)", "Debt/Equity", "Current Ratio", "Enterprise Value", "Altman Z-Score", "Piotroski F-Score"]:
                if k in ratios and ratios[k] is not None:
                    unified.append((k, ratios[k]))
        pm = analysis.get("price_metrics", {})
        sm = analysis.get("stock_metrics", {})
        for k in ["CAGR (%)", "Volatility (%)", "Max Drawdown (%)"]:
            if k in pm:
                unified.append((k, pm[k]))
        if sm:
            if sm.get("sharpe_ratio") is not None:
                unified.append(("Sharpe Ratio", sm["sharpe_ratio"]))
            if sm.get("profit_probability_pct") is not None:
                unified.append(("Odds of Winning (%)", sm["profit_probability_pct"]))
            if sm.get("loss_probability_pct") is not None:
                unified.append(("Odds of Losing (%)", sm["loss_probability_pct"]))
        if unified:
            uf = dict(unified)
            parts.append(_dict_to_html(uf, "Key Metrics"))
        if analysis.get("price_metrics"):
            parts.append(_dict_to_html(analysis["price_metrics"], "Price Metrics"))
        if analysis.get("stock_metrics"):
            parts.append(_dict_to_html(analysis["stock_metrics"], "Risk Metrics"))
        if analysis.get("signals"):
            parts.append(_dict_to_html(analysis["signals"], "Trade Signals"))
        parts.append(f"<p><strong>Decision:</strong> {analysis.get('decision', 'N/A')}</p>")
        if analysis.get("ml_prediction"):
            mp = analysis["ml_prediction"]
            parts.append(f"<p>Bullish: {mp.get('bullish_probability', 0):.1%} | Bearish: {mp.get('bearish_probability', 0):.1%}</p>")
    else:
        parts.append("<p>No technical data available.</p>")
    parts.append("</div>")

    # Sentiment
    parts.append('<div class="section"><h2>Sentiment Analysis</h2>')
    if sentiment_result and hasattr(sentiment_result, "total") and sentiment_result.total > 0:
        parts.append(f"<p>Positive: {sentiment_result.positive} | Neutral: {sentiment_result.neutral} | Negative: {sentiment_result.negative}</p>")
    else:
        parts.append("<p>No sentiment data.</p>")
    parts.append("</div>")

    # AI Perspective
    if ai_perspective:
        parts.append('<div class="section"><h2>AI Company Perspective</h2>')
        safe = html_module.escape(ai_perspective).replace("\n", "<br>")
        parts.append(f'<div class="ai-perspective">{safe}</div>')
        parts.append("</div>")

    # News
    parts.append('<div class="section"><h2>News Updates</h2>')
    if headlines:
        parts.append("<ul>")
        for h in headlines[:20]:
            parts.append(f"<li>{h}</li>")
        parts.append("</ul>")
    else:
        parts.append("<p>No news available.</p>")
    parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)


def html_to_pdf(html_content: str) -> Optional[bytes]:
    """Convert HTML to PDF bytes. Returns None if xhtml2pdf not available."""
    try:
        from xhtml2pdf import pisa
        from io import BytesIO
        buf = BytesIO()
        pisa.CreatePDF(html_content, dest=buf, encoding="utf-8")
        return buf.getvalue()
    except ImportError:
        return None
