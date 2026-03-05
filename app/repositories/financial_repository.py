"""Financial statements repository - Yahoo Finance."""

from datetime import datetime
from typing import Optional

import yfinance as yf

from app.domain.entities import FinancialStatements
from app.domain.interfaces import IFinancialRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)


class YahooFinancialRepository(IFinancialRepository):
    """Repository for fetching financial statements from Yahoo Finance."""

    def get_financials(self, ticker: str) -> Optional[FinancialStatements]:
        """
        Fetch balance sheet, income statement, and cashflow.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            FinancialStatements if successful, None otherwise.
        """
        try:
            ticker = ticker.strip().upper()
            stock = yf.Ticker(ticker)
            balance_sheet = stock.balance_sheet
            income_statement = stock.financials
            cashflow = stock.cashflow

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
