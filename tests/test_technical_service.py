"""Tests for TechnicalAnalysisService."""

import pytest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.domain.entities import StockPriceData
from app.services.technical_analysis_service import (
    TechnicalAnalysisService,
    _cagr,
    _volatility,
    _max_drawdown,
)


def _make_price_data(ticker: str, n: int = 300) -> StockPriceData:
    """Create sample price data."""
    dates = pd.date_range(end=datetime.now(), periods=n, freq="B")
    np.random.seed(42)
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    df = pd.DataFrame(
        {"Close": close, "Open": close * 0.99, "High": close * 1.01, "Low": close * 0.98, "Volume": 1e6},
        index=dates,
    )
    return StockPriceData(ticker=ticker, data=df, fetched_at=datetime.utcnow())


def test_cagr():
    """CAGR returns float."""
    df = pd.DataFrame({"Close": [100, 110, 121]})
    df["Close"] = df["Close"].pct_change()
    df["Close"].iloc[0] = 0
    df["cum_return"] = (1 + df["Close"]).cumprod()
    n = 2 / 252
    expected = (df["cum_return"].iloc[-1] ** (1 / n)) - 1
    assert isinstance(_cagr(_make_price_data("X").data), (float, np.floating))


def test_volatility():
    """Volatility returns non-negative float."""
    df = _make_price_data("X").data
    v = _volatility(df)
    assert v >= 0
    assert isinstance(v, (float, np.floating))


def test_max_drawdown():
    """Max drawdown between 0 and 1."""
    df = _make_price_data("X").data
    md = _max_drawdown(df)
    assert 0 <= md <= 1


def test_get_price_metrics_returns_metrics():
    """get_price_metrics returns dict with expected keys."""
    repo = MagicMock()
    repo.get_price_data.return_value = _make_price_data("AAPL", n=400)
    svc = TechnicalAnalysisService(repo)
    result = svc.get_price_metrics("AAPL")
    assert result is not None
    assert "CAGR (%)" in result
    assert "Sharpe Ratio" in result
    assert "Max Drawdown (%)" in result


def test_get_price_metrics_returns_none_when_insufficient_data():
    """get_price_metrics returns None when insufficient data."""
    repo = MagicMock()
    repo.get_price_data.return_value = _make_price_data("X", n=10)
    svc = TechnicalAnalysisService(repo)
    assert svc.get_price_metrics("X") is None
