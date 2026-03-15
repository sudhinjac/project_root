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
        web_search_context: str = "",
    ) -> Optional[str]:
        """
        Get AI-generated company perspective from aggregated data.

        Args:
            ticker: Stock ticker.
            financial_summary: Full P&L, Balance Sheet, Cash Flow + ratios.
            technical_summary: Technical metrics and signals.
            sentiment_summary: News sentiment.
            news_headlines: Latest company news headlines.
            annual_report_summary: Extracted text from annual report (10-K).
            web_search_context: Web search results (promoter red flags, etc.).

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
        if web_search_context:
            extra_sections.append(f"Web Search (Promoter/Company Intelligence):\n{web_search_context[:4000]}")

        extra = "\n\n".join(extra_sections) if extra_sections else ""

        prompt = f"""You are a senior financial advisor providing HIGH-GRADE INTELLIGENCE for investment decisions. Analyze ALL the following data for {ticker} and provide a comprehensive report. Act as a top-tier analyst would when advising institutional clients.

Your report MUST include these sections:

## 1. RECOMMENDATION (Required)
Clearly state: **BUY** / **HOLD** / **SELL** with a one-sentence rationale.

## 2. DEBT & LEVERAGE ANALYSIS
- Total debt, debt/equity ratio, interest coverage
- Is the company overleveraged or underleveraged?
- Debt maturity profile if visible from balance sheet

## 3. CASH FLOW GROWTH & QUALITY
- Operating cash flow trend (growth or decline)
- Free cash flow, capex intensity
- Cash flow vs net income (quality of earnings)
- Working capital changes

## 4. BALANCE SHEET STRENGTH
- Current ratio, quick ratio
- Asset quality, receivables, inventory
- Altman Z-Score interpretation (bankruptcy risk)
- Piotroski F-Score (0-3 weak, 4-6 average, 7-9 strong)

## 5. PROMOTER & RED FLAGS
- Any red flags on promoters from web search (pledging, insider selling, governance issues)
- Fraud or regulatory concerns
- Related party transactions if mentioned

## 6. OVERALL FINANCIAL HEALTH
- Is the company profitable or making a loss?
- How is the company doing vs peers?
- Key strengths and weaknesses

## 7. TECHNICAL & RISK
- CAGR, volatility, drawdown, Sharpe
- Odds of winning vs losing
- Trade signals

## 8. SENTIMENT & NEWS
- Market perception
- Key headlines impact

---

FULL FINANCIAL DATA (P&L, Balance Sheet, Cash Flow - use ALL line items for your analysis):
{financial_summary}

Technical Analysis:
{technical_summary}

Sentiment:
{sentiment_summary}
{chr(10) + extra if extra else ""}

Provide a balanced, professional, high-grade analysis. Be specific. Cite numbers from the financial statements. Flag any red flags."""

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
