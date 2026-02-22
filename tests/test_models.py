"""Tests for domain models."""

import pytest

from app.domain.models import StockMetrics, TickerRequest, MultiTickerRequest


class TestStockMetrics:
    def test_to_dict(self) -> None:
        m = StockMetrics(
            ticker="AAPL",
            annual_return=12.5,
            volatility=18.2,
            value_at_risk_95=-2.1,
            rsi=55.3,
            macd=0.5,
            macd_signal=0.3,
            macd_histogram=0.2,
        )
        d = m.to_dict()
        assert d["ticker"] == "AAPL"
        assert d["annual_return"] == 12.5
        assert d["volatility"] == 18.2
        assert d["value_at_risk_95"] == -2.1
        assert d["rsi"] == 55.3

    def test_rounding(self) -> None:
        m = StockMetrics(
            ticker="X",
            annual_return=12.3456,
            volatility=18.234,
            value_at_risk_95=-2.123,
            rsi=55.678,
            macd=0.12345,
            macd_signal=0.09876,
            macd_histogram=0.02469,
        )
        d = m.to_dict()
        assert d["annual_return"] == 12.3456
        assert d["rsi"] == 55.68


class TestTickerRequest:
    def test_normalizes_ticker(self) -> None:
        r = TickerRequest(ticker="  aapl  ")
        assert r.ticker == "AAPL"

    def test_uppercase(self) -> None:
        r = TickerRequest(ticker="msft")
        assert r.ticker == "MSFT"


class TestMultiTickerRequest:
    def test_normalizes_tickers(self) -> None:
        r = MultiTickerRequest(tickers=["  aapl  ", "msft", ""])
        assert r.tickers == ["AAPL", "MSFT"]
