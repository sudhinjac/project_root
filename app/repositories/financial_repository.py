"""Financial statements repository - Yahoo Finance."""

from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

from app.domain.entities import FinancialStatements
from app.domain.interfaces import IFinancialRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _prefer_recent(df_annual: Optional[pd.DataFrame], df_quarterly: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Use quarterly if it has more recent data than annual; otherwise use annual."""
    if df_annual is None or df_annual.empty:
        return df_quarterly if (df_quarterly is not None and not df_quarterly.empty) else None
    if df_quarterly is None or df_quarterly.empty:
        return df_annual
    try:
        ann_dates = pd.to_datetime(df_annual.columns, errors="coerce")
        qtr_dates = pd.to_datetime(df_quarterly.columns, errors="coerce")
        ann_latest = ann_dates.max()
        qtr_latest = qtr_dates.max()
        if pd.notna(qtr_latest) and pd.notna(ann_latest) and qtr_latest > ann_latest:
            return df_quarterly
    except Exception:
        pass
    return df_annual


class YahooFinancialRepository(IFinancialRepository):
    """Repository for fetching financial statements from Yahoo Finance."""

    def get_financials(self, ticker: str) -> Optional[FinancialStatements]:
        """
        Fetch balance sheet, income statement, and cashflow.
        Prefers quarterly data when it is more recent than annual (e.g., 2024 quarterly vs 2023 annual).

        Returns:
            FinancialStatements if successful, None otherwise.
        """
        try:
            ticker = ticker.strip().upper()
            stock = yf.Ticker(ticker)
            bs_ann = stock.balance_sheet
            inc_ann = stock.financials
            cf_ann = stock.cashflow
            bs_qtr = getattr(stock, "quarterly_balance_sheet", None) or getattr(stock, "quarterly_balancesheet", None)
            inc_qtr = getattr(stock, "quarterly_income_stmt", None) or getattr(stock, "quarterly_financials", None)
            cf_qtr = getattr(stock, "quarterly_cashflow", None) or getattr(stock, "quarterly_cash_flow", None)

            balance_sheet = _prefer_recent(bs_ann, bs_qtr) or bs_ann
            income_statement = _prefer_recent(inc_ann, inc_qtr) or inc_ann
            cashflow = _prefer_recent(cf_ann, cf_qtr) or cf_ann

            if balance_sheet is None or balance_sheet.empty:
                logger.warning("No balance sheet for %s", ticker)
                return None
            if income_statement is None or income_statement.empty:
                logger.warning("No income statement for %s", ticker)
                return None
            if cashflow is None or cashflow.empty:
                logger.warning("No cashflow for %s", ticker)
                return None

            return FinancialStatements(
                ticker=ticker,
                balance_sheet=balance_sheet,
                income_statement=income_statement,
                cashflow=cashflow,
                fetched_at=datetime.utcnow(),
            )
        except Exception as e:
            logger.exception("Error fetching financials for %s: %s", ticker, e)
            return None
