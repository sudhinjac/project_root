"""Tests for FinancialRatioService."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd

from app.domain.entities import FinancialStatements
from app.services.financial_ratio_service import FinancialRatioService, _get_value


def test_get_value_finds_match():
    """_get_value returns first matching row value."""
    df = pd.DataFrame({"col": [100, 200]}, index=["Net Income", "Revenue"])
    assert _get_value(df, "Net Income") == 100
    assert _get_value(df, "Revenue") == 200


def test_get_value_no_match():
    """_get_value returns None when no match."""
    df = pd.DataFrame({"col": [100]}, index=["Other"])
    assert _get_value(df, "Net Income") is None


def test_compute_ratios_returns_none_when_no_data():
    """compute_ratios returns None when repository returns None."""
    repo = MagicMock()
    repo.get_financials.return_value = None
    svc = FinancialRatioService(repo)
    assert svc.compute_ratios("AAPL") is None


def test_compute_ratios_computes_roe():
    """compute_ratios computes ROE when data available."""
    bs = pd.DataFrame({0: [1000]}, index=["Total Stockholder Equity"])
    inc = pd.DataFrame({0: [100]}, index=["Net Income"])
    cf = pd.DataFrame({0: [50]}, index=["Operating Cash Flow"])
    stmts = FinancialStatements(
        ticker="AAPL",
        balance_sheet=bs,
        income_statement=inc,
        cashflow=cf,
        fetched_at=datetime.utcnow(),
    )
    repo = MagicMock()
    repo.get_financials.return_value = stmts
    svc = FinancialRatioService(repo)
    result = svc.compute_ratios("AAPL")
    assert result is not None
    assert result["ratios"]["ROE (%)"] == 10.0  # 100/1000 * 100
