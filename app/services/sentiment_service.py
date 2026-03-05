"""Sentiment analysis service - VADER on news headlines."""

from typing import Optional

from app.domain.entities import SentimentResult
from app.domain.interfaces import INewsRepository, ISentimentService
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _analyzer = SentimentIntensityAnalyzer()
except (ImportError, Exception):
    _analyzer = None


class SentimentService(ISentimentService):
    """Service for sentiment analysis of news headlines."""

    def __init__(self, news_repo: INewsRepository) -> None:
        self._repo = news_repo

    def analyze(self, company_query: str, limit: int = 30) -> Optional[SentimentResult]:
        """
        Analyze sentiment from news headlines.

        Args:
            company_query: Company name or ticker for news search.
            limit: Max headlines to fetch.

        Returns:
            SentimentResult with positive/neutral/negative counts and headlines.
        """
        if _analyzer is None:
            logger.warning("VADER not available. Install nltk and run nltk.download('vader_lexicon')")
            return None

        headlines = self._repo.get_headlines(company_query, limit=limit)
        if not headlines:
            return SentimentResult(positive=0, neutral=0, negative=0, headlines=[])

        positive = neutral = negative = 0
        for h in headlines:
            scores = _analyzer.polarity_scores(h)
            compound = scores.get("compound", 0)
            if compound >= 0.05:
                positive += 1
            elif compound <= -0.05:
                negative += 1
            else:
                neutral += 1

        return SentimentResult(
            positive=positive,
            neutral=neutral,
            negative=negative,
            headlines=headlines,
        )
