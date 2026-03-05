"""Financial ratio service - ROCE, ROE, Altman Z, Piotroski, Enterprise Value, etc."""

from typing import Optional

import pandas as pd
import yfinance as yf

from app.domain.interfaces import IFinancialRatioService, IFinancialRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _get_value(df: pd.DataFrame, *keys: str) -> Optional[float]:
    """Get first matching value from DataFrame index (Yahoo uses various naming)."""
    if df is None or df.empty:
        return None
    idx = df.index.astype(str)
    for key in keys:
        matches = [i for i, v in enumerate(idx) if key.lower() in v.lower()]
        if matches:
            val = df.iloc[matches[0]].iloc[0]
            if pd.notna(val) and val != 0:
                return float(val)
    return None


def _get_value_col(df: pd.DataFrame, col_idx: int, *keys: str) -> Optional[float]:
    """Get value from specific column (0=latest, 1=prior year)."""
    if df is None or df.empty or col_idx >= len(df.columns):
        return None
    col = df.iloc[:, col_idx]
    idx = df.index.astype(str)
    for key in keys:
        matches = [i for i, v in enumerate(idx) if key.lower() in v.lower()]
        if matches:
            val = col.iloc[matches[0]]
            if pd.notna(val) and val != 0:
                return float(val)
    return None


def _piotroski_score(inc, bs, cf) -> int:
    """Piotroski F-Score (0-9). Uses current and prior year when available."""
    score = 0
    ni = _get_value(inc, "Net Income", "Net Income Common Stockholders")
    revenue = _get_value(inc, "Total Revenue", "Revenue")
    total_assets = _get_value(bs, "Total Assets")
    equity = _get_value(bs, "Total Stockholder Equity", "Stockholders Equity", "Total Equity")
    cfo = _get_value(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    current_assets = _get_value(bs, "Current Assets")
    current_liab = _get_value(bs, "Current Liabilities")
    gross_profit = _get_value(inc, "Gross Profit")
    ni_prev = _get_value_col(inc, 1, "Net Income", "Net Income Common Stockholders") if inc is not None and len(inc.columns) > 1 else None
    roa_prev = None
    if inc is not None and bs is not None and len(inc.columns) > 1 and len(bs.columns) > 1:
        ta_prev = _get_value_col(bs, 1, "Total Assets")
        if ni_prev and ta_prev and ta_prev != 0:
            roa_prev = ni_prev / ta_prev
    roa = (ni / total_assets) if ni and total_assets and total_assets != 0 else None

    # 1. Net income > 0
    if ni is not None and ni > 0:
        score += 1
    # 2. ROA > 0
    if roa is not None and roa > 0:
        score += 1
    # 3. Operating cash flow > 0
    if cfo is not None and cfo > 0:
        score += 1
    # 4. CFO > NI (quality of earnings)
    if cfo is not None and ni is not None and cfo > ni:
        score += 1
    # 5. ROA change (improvement)
    if roa is not None and roa_prev is not None and roa > roa_prev:
        score += 1
    # 6. Leverage change - skip without long-term debt comparison
    # 7. Current ratio improvement
    cr = (current_assets / current_liab) if current_assets and current_liab and current_liab != 0 else None
    if cr is not None and cr > 1.0:
        score += 1
    # 8. Gross margin (no dilution)
    if gross_profit and revenue and revenue != 0 and (gross_profit / revenue) > 0:
        score += 1
    # 9. Asset turnover improvement
    if revenue and total_assets and total_assets != 0 and (revenue / total_assets) > 0.1:
        score += 1
    return min(score, 9)


class FinancialRatioService(IFinancialRatioService):
    """Service for computing ROCE, ROE and other financial ratios."""

    def __init__(self, repo: IFinancialRepository) -> None:
        self._repo = repo

    def compute_ratios(self, ticker: str) -> Optional[dict]:
        """
        Compute ROCE, ROE and other ratios from financial statements.

        Returns:
            Dict with ratio names and values, or None if data unavailable.
        """
        stmts = self._repo.get_financials(ticker)
        if not stmts:
            return None

        bs = stmts.balance_sheet
        inc = stmts.income_statement
        cf = stmts.cashflow

        # Use most recent period (first column)
        def col0(df: pd.DataFrame):
            return df.iloc[:, 0] if len(df.columns) > 0 else pd.Series()

        net_income = _get_value(inc, "Net Income", "Net Income Common Stockholders")
        equity = _get_value(bs, "Total Stockholder Equity", "Stockholders Equity", "Total Equity")
        ebit = _get_value(inc, "EBIT", "Operating Income", "Ebit")
        total_assets = _get_value(bs, "Total Assets")
        current_liab = _get_value(bs, "Current Liabilities")
        total_liab = _get_value(bs, "Total Liabilities")
        revenue = _get_value(inc, "Total Revenue", "Revenue")
        operating_cashflow = _get_value(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
        free_cashflow = _get_value(cf, "Free Cash Flow")
        debt = _get_value(bs, "Total Debt", "Long Term Debt")

        ratios: dict[str, Optional[float]] = {}

        # ROE = Net Income / Shareholders' Equity
        if net_income and equity and equity != 0:
            ratios["ROE (%)"] = (net_income / abs(equity)) * 100
        else:
            ratios["ROE (%)"] = None

        # ROCE = EBIT / (Total Assets - Current Liabilities)
        capital_employed = None
        if total_assets is not None and current_liab is not None:
            capital_employed = total_assets - current_liab
        if ebit and capital_employed and capital_employed != 0:
            ratios["ROCE (%)"] = (ebit / abs(capital_employed)) * 100
        else:
            ratios["ROCE (%)"] = None

        # ROA = Net Income / Total Assets
        if net_income and total_assets and total_assets != 0:
            ratios["ROA (%)"] = (net_income / abs(total_assets)) * 100
        else:
            ratios["ROA (%)"] = None

        # Operating Margin = Operating Income / Revenue
        if ebit and revenue and revenue != 0:
            ratios["Operating Margin (%)"] = (ebit / abs(revenue)) * 100
        else:
            ratios["Operating Margin (%)"] = None

        # Net Margin = Net Income / Revenue
        if net_income and revenue and revenue != 0:
            ratios["Net Margin (%)"] = (net_income / abs(revenue)) * 100
        else:
            ratios["Net Margin (%)"] = None

        # Debt/Equity
        if debt and equity and equity != 0:
            ratios["Debt/Equity"] = debt / abs(equity)
        else:
            ratios["Debt/Equity"] = None

        # Current Ratio (need Current Assets)
        current_assets = _get_value(bs, "Current Assets")
        if current_assets and current_liab and current_liab != 0:
            ratios["Current Ratio"] = current_assets / current_liab
        else:
            ratios["Current Ratio"] = None

        # Enterprise Value and market cap (for Altman)
        mcap = ev = None
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            mcap = info.get("marketCap") or info.get("enterpriseValue")
            ev = info.get("enterpriseValue")
            cash = _get_value(bs, "Cash", "Cash And Cash Equivalents")
            if ev is not None:
                ratios["Enterprise Value"] = ev
            elif mcap is not None and debt is not None and cash is not None:
                ratios["Enterprise Value"] = mcap + debt - cash
            elif mcap is not None:
                ratios["Enterprise Value"] = mcap
            else:
                ratios["Enterprise Value"] = None
        except Exception:
            ratios["Enterprise Value"] = None

        # Altman Z-Score: 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
        working_capital = (current_assets - current_liab) if (current_assets is not None and current_liab is not None) else None
        retained_earnings = _get_value(bs, "Retained Earnings")
        sales = revenue
        mve = mcap
        if working_capital is not None and total_assets and total_assets != 0 and retained_earnings is not None and ebit is not None and total_liab and total_liab != 0 and sales:
            a = working_capital / total_assets
            b = retained_earnings / total_assets
            c = ebit / total_assets
            d = (mve / total_liab) if mve else 0.0
            e = sales / total_assets
            z = 1.2 * a + 1.4 * b + 3.3 * c + 0.6 * d + 1.0 * e
            ratios["Altman Z-Score"] = round(z, 2)
        else:
            ratios["Altman Z-Score"] = None

        # Piotroski F-Score
        ratios["Piotroski F-Score"] = _piotroski_score(inc, bs, cf)

        return {
            "ticker": ticker,
            "ratios": {k: round(v, 4) if isinstance(v, float) else v for k, v in ratios.items()},
            "balance_sheet": bs,
            "income_statement": inc,
            "cashflow": cf,
        }
