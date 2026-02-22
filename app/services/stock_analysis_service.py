"""Stock analysis service - computes metrics from price data."""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from app.domain.entities import StockPriceData
from app.domain.interfaces import IStockAnalysisService, IStockRepository
from app.domain.models import StockMetrics
from app.utils.logger import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 252
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VAR_CONFIDENCE = 0.95


class StockAnalysisService(IStockAnalysisService):
    """Service for computing stock metrics: annual return, volatility, VaR, RSI, MACD."""

    def __init__(self, repository: IStockRepository) -> None:
        self._repo = repository

    def analyze_ticker(self, ticker: str) -> Optional[dict]:
        """
        Analyze a single ticker and return metrics.

        Returns:
            Dict with metrics or None if analysis fails.
        """
        price_data = self._repo.get_price_data(ticker)
        if not price_data or price_data.is_empty or not price_data.has_close_column:
            return None

        try:
            metrics = self._compute_metrics(price_data)
            return metrics.to_dict()
        except Exception as e:
            logger.exception("Error analyzing %s: %s", ticker, e)
            return None

    def _compute_metrics(self, price_data: StockPriceData) -> StockMetrics:
        """Compute all metrics from price data."""
        df = price_data.data.copy()
        close = df["Close"].dropna()

        if len(close) < 30:
            raise ValueError("Insufficient data for analysis")

        returns = np.log(close / close.shift(1)).dropna()

        annual_return = self._annual_return(returns)
        volatility = self._volatility(returns)
        var_95 = self._value_at_risk(returns, VAR_CONFIDENCE)
        rsi = self._rsi(close)
        macd, macd_signal, macd_hist = self._macd(close)

        return StockMetrics(
            ticker=price_data.ticker,
            annual_return=annual_return,
            volatility=volatility,
            value_at_risk_95=var_95,
            rsi=rsi,
            macd=macd,
            macd_signal=macd_signal,
            macd_histogram=macd_hist,
        )

    @staticmethod
    def _annual_return(returns: pd.Series) -> float:
        """Annualized log return (as percentage)."""
        mean_daily = returns.mean()
        return float(mean_daily * TRADING_DAYS * 100)

    @staticmethod
    def _volatility(returns: pd.Series) -> float:
        """Annualized volatility (as percentage)."""
        std_daily = returns.std()
        return float(std_daily * np.sqrt(TRADING_DAYS) * 100)

    @staticmethod
    def _value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Parametric VaR (1-day) as percentage.
        Negative value indicates potential loss.
        """
        mean_daily = returns.mean()
        std_daily = returns.std()
        if std_daily == 0:
            return 0.0
        z = stats.norm.ppf(1 - confidence)
        var_1d = mean_daily + z * std_daily
        return float(var_1d * 100)

    @staticmethod
    def _rsi(close: pd.Series, window: int = RSI_WINDOW) -> float:
        """Relative Strength Index (most recent value)."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

    @staticmethod
    def _macd(close: pd.Series) -> tuple[float, float, float]:
        """
        MACD, signal line, and histogram.
        Returns most recent values.
        """
        exp1 = close.ewm(span=MACD_FAST, adjust=False).mean()
        exp2 = close.ewm(span=MACD_SLOW, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        histogram = macd_line - signal_line

        macd_val = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0
        signal_val = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0
        hist_val = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0

        return macd_val, signal_val, hist_val
