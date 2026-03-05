"""News headlines repository - Google News RSS."""

import feedparser
from urllib.parse import quote

from app.domain.interfaces import INewsRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)


class GoogleNewsRepository(INewsRepository):
    """Repository for fetching news headlines from Google News RSS."""

    def get_headlines(self, company_query: str, limit: int = 30) -> list[str]:
        """
        Fetch news headlines for a company/stock query.

        Args:
            company_query: Company name or ticker for search.
            limit: Maximum number of headlines to return.

        Returns:
            List of headline strings.
        """
        try:
            encoded = quote(company_query)
            feed_url = f"https://news.google.com/rss/search?q={encoded}+stock"
            feed = feedparser.parse(feed_url)
            headlines = [entry.title for entry in feed.entries]
            return headlines[:limit]
        except Exception as e:
            logger.exception("Error fetching news for %s: %s", company_query, e)
            return []
