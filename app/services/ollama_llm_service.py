"""Ollama LLM service - company perspective via DeepSeek."""

from typing import Optional

from app.domain.interfaces import ILLMService
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from ollama import chat as ollama_chat
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False


class OllamaLLMService(ILLMService):
    """Service for AI-generated company perspective using Ollama (DeepSeek)."""

    def __init__(self, model: str = "deepseek") -> None:
        self._model = model

    def get_company_perspective(
        self,
        ticker: str,
        financial_summary: str,
        technical_summary: str,
        sentiment_summary: str,
    ) -> Optional[str]:
        """
        Get AI-generated company perspective from aggregated data.

        Args:
            ticker: Stock ticker.
            financial_summary: Summary of financial ratios and statements.
            technical_summary: Summary of technical metrics and signals.
            sentiment_summary: Summary of news sentiment.

        Returns:
            AI-generated analysis text or None if Ollama unavailable.
        """
        if not _OLLAMA_AVAILABLE:
            logger.warning("Ollama not installed. pip install ollama")
            return None

        prompt = f"""You are an expert financial analyst. Analyze the following data for {ticker} and provide a professional investment perspective (3-5 paragraphs). Act as a senior financial analyst would when advising a client.

Focus on:
1. Financial health: ROCE, ROE, margins, balance sheet strength, cash flow
2. Technical outlook: momentum, risk metrics (Sharpe, Sortino, drawdown), signals
3. Sentiment and market perception from news
4. Overall recommendation: Buy / Hold / Sell with clear rationale

Financial Data (P&L, Balance Sheet, Cash Flow, Ratios):
{financial_summary}

Technical Analysis:
{technical_summary}

Sentiment:
{sentiment_summary}

Provide a balanced, professional analysis suitable for an investment decision."""

        try:
            response = ollama_chat(model=self._model, messages=[{"role": "user", "content": prompt}])
            if hasattr(response, "message"):
                content = getattr(response.message, "content", "") or ""
            elif isinstance(response, dict):
                content = response.get("message", {}).get("content", "")
            else:
                content = ""
            return content if content else None
        except Exception as e:
            logger.exception("Ollama request failed: %s", e)
            return None
