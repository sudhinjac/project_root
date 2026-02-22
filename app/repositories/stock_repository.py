"""Stock data repository implementation."""

from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

from app.domain.entities import StockPriceData
from app.domain.interfaces import IStockRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)


class StockRepository(IStockRepository):
    """Repository for fetching stock price data from Yahoo Finance."""

    def get_price_data(self, ticker: str, period: str = "2y") -> Optional[StockPriceData]:
        """
        Fetch price data for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g., AAPL, MSFT).
            period: Data period (e.g., '2y', '1y', '6mo').

        Returns:
            StockPriceData if successful, None otherwise.
        """
        try:
            ticker = ticker.strip().upper()
            raw = yf.download(ticker, period=period, progress=False, threads=False)

            if raw.empty:
                logger.warning("No data returned for ticker %s", ticker)
                return None

            # Handle multi-level columns from yfinance
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            required = ["Close"]
            if not all(col in raw.columns for col in required):
                logger.warning("Missing required columns for %s: %s", ticker, raw.columns.tolist())
                return None

            return StockPriceData(ticker=ticker, data=raw, fetched_at=datetime.utcnow())

        except Exception as e:
            logger.exception("Error fetching data for %s: %s", ticker, e)
            return None
