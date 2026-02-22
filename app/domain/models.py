"""Domain models for stock analysis."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StockMetrics:
    """Stock analysis metrics output model."""

    ticker: str
    annual_return: float
    volatility: float
    value_at_risk_95: float
    rsi: float
    macd: float
    macd_signal: float
    macd_histogram: float

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "ticker": self.ticker,
            "annual_return": round(self.annual_return, 4),
            "volatility": round(self.volatility, 4),
            "value_at_risk_95": round(self.value_at_risk_95, 4),
            "rsi": round(self.rsi, 2),
            "macd": round(self.macd, 4),
            "macd_signal": round(self.macd_signal, 4),
            "macd_histogram": round(self.macd_histogram, 4),
        }


@dataclass
class TickerRequest:
    """Request model for single ticker analysis."""

    ticker: str

    def __post_init__(self) -> None:
        self.ticker = self.ticker.strip().upper()


@dataclass
class MultiTickerRequest:
    """Request model for multi-ticker analysis (future use)."""

    tickers: list[str]

    def __post_init__(self) -> None:
        self.tickers = [t.strip().upper() for t in self.tickers if t.strip()]
