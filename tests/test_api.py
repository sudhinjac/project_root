"""Tests for FastAPI routes."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_analyze_ticker_empty() -> None:
    resp = client.get("/api/v1/analyze/")
    assert resp.status_code in (404, 422)


def test_analyze_ticker_valid() -> None:
    resp = client.get("/api/v1/analyze/AAPL")
    if resp.status_code == 200:
        data = resp.json()
        assert "ticker" in data
        assert "annual_return" in data
        assert "volatility" in data
        assert "value_at_risk_95" in data
        assert "rsi" in data
        assert "macd" in data
    else:
        # May fail if yfinance has no data (e.g. in CI)
        assert resp.status_code == 404


def test_analyze_batch_empty() -> None:
    resp = client.post("/api/v1/analyze/batch", json={"tickers": []})
    assert resp.status_code == 400


def test_analyze_batch_valid() -> None:
    resp = client.post("/api/v1/analyze/batch", json={"tickers": ["AAPL", "MSFT"]})
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert "errors" in data
    assert "success_count" in data
    assert "error_count" in data
