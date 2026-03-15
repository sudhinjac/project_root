"""Web search utility for promoter red flags, company intelligence."""

from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from duckduckgo_search import DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    _DDGS_AVAILABLE = False


def search_company_intelligence(company_name: str, ticker: str, max_results: int = 8) -> str:
    """
    Search for promoter red flags, company news, and financial intelligence.
    Returns concatenated snippets for Ollama context.
    """
    if not _DDGS_AVAILABLE:
        logger.debug("duckduckgo_search not installed. pip install duckduckgo-search")
        return ""

    queries = [
        f"{company_name} {ticker} promoter red flags fraud",
        f"{company_name} promoter pledge shares SEBI",
        f"{company_name} {ticker} debt cash flow analysis",
        f"{company_name} financial health news",
    ]
    results = []
    seen = set()
    try:
        ddgs = DDGS()
        for q in queries:
            try:
                for r in ddgs.text(q, max_results=3):
                    title = r.get("title", "")
                    body = r.get("body", "")
                    key = (title[:80], body[:100])
                    if key not in seen:
                        seen.add(key)
                        results.append(f"[{title}]\n{body}")
                        if len(results) >= max_results:
                            break
            except Exception as e:
                logger.debug("Search query failed %s: %s", q[:50], e)
            if len(results) >= max_results:
                break
    except Exception as e:
        logger.warning("Web search failed: %s", e)
        return ""

    if not results:
        return ""
    return "\n\n---\n\n".join(results[:max_results])
