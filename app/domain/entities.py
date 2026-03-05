"""Domain entities."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass
class StockPriceData:
    """Entity representing raw stock price data."""

    ticker: str
    data: pd.DataFrame
    fetched_at: datetime

    @property
    def is_empty(self) -> bool:
        """Check if price data is empty."""
        return self.data.empty or len(self.data) < 30

    @property
    def has_close_column(self) -> bool:
        """Check if Close column exists."""
        return "Close" in self.data.columns


@dataclass
class FinancialStatements:
    """Entity representing financial statements from Yahoo Finance."""

    ticker: str
    balance_sheet: pd.DataFrame
    income_statement: pd.DataFrame
    cashflow: pd.DataFrame
    fetched_at: datetime


@dataclass
class SentimentResult:
    """Entity representing sentiment analysis result."""

    positive: int
    neutral: int
    negative: int
    headlines: list[str]

    @property
    def total(self) -> int:
        return self.positive + self.neutral + self.negative
