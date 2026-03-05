"""Financial ratio service - ROCE, ROE and other ratios from Yahoo Finance."""

from typing import Optional

import pandas as pd

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

        return {
            "ticker": ticker,
            "ratios": {k: round(v, 4) if v is not None else None for k, v in ratios.items()},
            "balance_sheet": bs,
            "income_statement": inc,
            "cashflow": cf,
        }
