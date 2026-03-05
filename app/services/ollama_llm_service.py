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
        news_headlines: str = "",
        annual_report_summary: str = "",
    ) -> Optional[str]:
        """
        Get AI-generated company perspective from aggregated data.

        Args:
            ticker: Stock ticker.
            financial_summary: Summary of financial ratios and statements.
            technical_summary: Summary of technical metrics and signals.
            sentiment_summary: Summary of news sentiment.
            news_headlines: Latest company news headlines.
            annual_report_summary: Extracted text from annual report (10-K).

        Returns:
            AI-generated analysis text or None if Ollama unavailable.
        """
        if not _OLLAMA_AVAILABLE:
            logger.warning("Ollama not installed. pip install ollama")
            return None

        extra_sections = []
        if news_headlines:
            extra_sections.append(f"Latest Company News:\n{news_headlines}")
        if annual_report_summary:
            extra_sections.append(f"Annual Report (10-K) Excerpt:\n{annual_report_summary[:12000]}")

        extra = "\n\n".join(extra_sections) if extra_sections else ""

        prompt = f"""You are an expert financial analyst. Analyze ALL the following data for {ticker} and provide a COMPREHENSIVE investment report. Your report MUST include these sections:

## 1. RECOMMENDATION (Required)
Clearly state: **BUY** / **HOLD** / **SELL** with a one-sentence rationale.

## 2. FINANCIAL POSITION
- Is the company profitable or making a loss? (cite net income, margins)
- Balance sheet strength (debt/equity, current ratio, Altman Z-Score)
- Cash flow health
- Piotroski F-Score interpretation (0-3 weak, 4-6 average, 7-9 strong)

## 3. FUTURE PROSPECTS
- Growth potential based on financials and annual report
- Key risks from balance sheet and sentiment
- Management discussion highlights if from annual report

## 4. TECHNICAL & RISK METRICS
- CAGR, volatility, drawdown, Sharpe ratio
- Odds of winning vs losing
- Trade signals summary

## 5. SENTIMENT & NEWS
- Market perception from news
- Key headlines impact

---

Financial Data (Ratios: ROE, ROCE, Debt/Equity, Current Ratio, Enterprise Value, Altman Z, Piotroski; P&L, Balance Sheet, Cash Flow):
{financial_summary}

Technical Analysis (CAGR, Return, Volatility, Drawdown, Sharpe, Odds of Winning/Losing):
{technical_summary}

Sentiment:
{sentiment_summary}
{chr(10) + extra if extra else ""}

Provide a balanced, professional analysis. Be specific. Use the data provided."""

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
