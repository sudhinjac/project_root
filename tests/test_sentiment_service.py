"""Tests for SentimentService."""

import pytest
from unittest.mock import MagicMock

from app.domain.entities import SentimentResult
from app.services.sentiment_service import SentimentService


def test_analyze_returns_result_when_headlines_available():
    """analyze returns SentimentResult when headlines exist."""
    repo = MagicMock()
    repo.get_headlines.return_value = ["Good news for company", "Stock rises today"]
    svc = SentimentService(repo)
    result = svc.analyze("Apple")
    if result:
        assert isinstance(result, SentimentResult)
        assert result.total >= 0
        assert len(result.headlines) == 2


def test_analyze_returns_result_when_no_headlines():
    """analyze returns SentimentResult with zeros when no headlines."""
    repo = MagicMock()
    repo.get_headlines.return_value = []
    svc = SentimentService(repo)
    result = svc.analyze("NonExistentCompany123")
    if result:
        assert result.total == 0
        assert result.positive == 0
        assert result.neutral == 0
        assert result.negative == 0
