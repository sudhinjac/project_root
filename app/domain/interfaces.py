"""Domain interfaces (abstract contracts)."""

from abc import ABC, abstractmethod
from typing import Optional

from app.domain.entities import StockPriceData


class IStockRepository(ABC):
    """Interface for stock data repository."""

    @abstractmethod
    def get_price_data(self, ticker: str, period: str = "2y") -> Optional[StockPriceData]:
        """Fetch price data for a ticker."""
        pass


class IStockAnalysisService(ABC):
    """Interface for stock analysis service."""

    @abstractmethod
    def analyze_ticker(self, ticker: str) -> Optional[dict]:
        """Analyze a single ticker and return metrics."""
        pass
