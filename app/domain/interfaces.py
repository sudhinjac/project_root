"""Domain interfaces (abstract contracts)."""

from abc import ABC, abstractmethod
from typing import Optional

from app.domain.entities import FinancialStatements, SentimentResult, StockPriceData


class IStockRepository(ABC):
    """Interface for stock data repository."""

    @abstractmethod
    def get_price_data(self, ticker: str, period: str = "2y") -> Optional[StockPriceData]:
        """Fetch price data for a ticker."""
        pass


class IFinancialRepository(ABC):
    """Interface for financial statements repository."""

    @abstractmethod
    def get_financials(self, ticker: str) -> Optional[FinancialStatements]:
        """Fetch balance sheet, income statement, and cashflow."""
        pass


class INewsRepository(ABC):
    """Interface for news/headlines repository."""

    @abstractmethod
    def get_headlines(self, company_query: str, limit: int = 30) -> list[str]:
        """Fetch news headlines for sentiment analysis."""
        pass


class IStockAnalysisService(ABC):
    """Interface for stock analysis service."""

    @abstractmethod
    def analyze_ticker(self, ticker: str) -> Optional[dict]:
        """Analyze a single ticker and return metrics."""
        pass


class IFinancialRatioService(ABC):
    """Interface for financial ratio calculation."""

    @abstractmethod
    def compute_ratios(self, ticker: str) -> Optional[dict]:
        """Compute ROCE, ROE and other ratios from financial statements."""
        pass


class ISentimentService(ABC):
    """Interface for sentiment analysis."""

    @abstractmethod
    def analyze(self, company_query: str) -> Optional[SentimentResult]:
        """Analyze sentiment from news headlines."""
        pass


class ILLMService(ABC):
    """Interface for LLM-based company analysis."""

    @abstractmethod
    def get_company_perspective(
        self,
        ticker: str,
        financial_summary: str,
        technical_summary: str,
        sentiment_summary: str,
    ) -> Optional[str]:
        """Get AI-generated company perspective from aggregated data."""
        pass
