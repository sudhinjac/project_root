"""Tests for stock analysis service."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from datetime import datetime

from app.domain.entities import StockPriceData
from app.services.stock_analysis_service import StockAnalysisService


@pytest.fixture
def mock_repo() -> MagicMock:
    return MagicMock()


@pytest.fixture
def sample_price_data() -> StockPriceData:
    """Generate sample price data for testing."""
    dates = pd.date_range("2022-01-01", periods=300, freq="B")
    import numpy as np
    np.random.seed(42)
    close = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.01))
    df = pd.DataFrame({"Close": close}, index=dates)
    return StockPriceData(ticker="TEST", data=df, fetched_at=datetime.utcnow())


def test_analyze_ticker_returns_none_when_no_data(mock_repo: MagicMock) -> None:
    mock_repo.get_price_data.return_value = None
    svc = StockAnalysisService(mock_repo)
    result = svc.analyze_ticker("INVALID")
    assert result is None


def test_analyze_ticker_returns_none_when_empty_data(mock_repo: MagicMock) -> None:
    mock_repo.get_price_data.return_value = StockPriceData(
        ticker="X", data=pd.DataFrame(), fetched_at=datetime.utcnow()
    )
    svc = StockAnalysisService(mock_repo)
    result = svc.analyze_ticker("X")
    assert result is None


def test_analyze_ticker_returns_metrics(mock_repo: MagicMock, sample_price_data: StockPriceData) -> None:
    mock_repo.get_price_data.return_value = sample_price_data
    svc = StockAnalysisService(mock_repo)
    result = svc.analyze_ticker("TEST")

    assert result is not None
    assert result["ticker"] == "TEST"
    assert "annual_return" in result
    assert "volatility" in result
    assert "value_at_risk_95" in result
    assert "rsi" in result
    assert "macd" in result
    assert "macd_signal" in result
    assert "macd_histogram" in result


def test_annual_return_volatility_sign(mock_repo: MagicMock, sample_price_data: StockPriceData) -> None:
    mock_repo.get_price_data.return_value = sample_price_data
    svc = StockAnalysisService(mock_repo)
    result = svc.analyze_ticker("TEST")
    assert result is not None
    assert isinstance(result["annual_return"], (int, float))
    assert isinstance(result["volatility"], (int, float))
    assert result["volatility"] >= 0


def test_rsi_in_valid_range(mock_repo: MagicMock, sample_price_data: StockPriceData) -> None:
    mock_repo.get_price_data.return_value = sample_price_data
    svc = StockAnalysisService(mock_repo)
    result = svc.analyze_ticker("TEST")
    assert result is not None
    assert 0 <= result["rsi"] <= 100
