"""Annual report repository - fetches and extracts text from company annual reports."""

from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


def _fetch_sec_10k_summary(ticker: str) -> Optional[str]:
    """Fetch 10-K filing summary from SEC EDGAR for US tickers."""
    if not _REQUESTS_AVAILABLE:
        return None
    ticker_clean = ticker.split(".")[0].upper()
    if any(ticker.endswith(s) for s in [".NS", ".BO", ".NSE", ".BSE"]):
        return None
    try:
        headers = {"User-Agent": "StockAnalysisApp/1.0 (contact@example.com)"}
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        r = requests.get(tickers_url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        cik = None
        cik_str = None
        for company in data.values():
            if company.get("ticker", "").upper() == ticker_clean:
                cik_str = str(company.get("cik_str", "")).zfill(10)
                break
        if not cik_str:
            return None
        sub_url = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
        r2 = requests.get(sub_url, headers=headers, timeout=10)
        r2.raise_for_status()
        subs = r2.json()
        recent = subs.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        for i, form in enumerate(forms):
            if form == "10-K":
                acc = recent.get("accessionNumber", [])
                if i < len(acc):
                    acc_num = acc[i]
                    prim_list = recent.get("primaryDocument", [])
                    primary_doc = prim_list[i] if i < len(prim_list) else "10-K"
                    doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik_str)}/{acc_num}/{primary_doc}.htm"
                    try:
                        r3 = requests.get(doc_url, headers=headers, timeout=15)
                        if r3.status_code == 200 and len(r3.text) > 500:
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(r3.text, "html.parser")
                            text = soup.get_text(separator=" ", strip=True)
                            return text[:15000] + "..." if len(text) > 15000 else text
                    except Exception:
                        pass
                break
    except Exception as e:
        logger.debug("SEC 10-K fetch failed for %s: %s", ticker, e)
    return None


def get_annual_report_summary(ticker: str) -> Optional[str]:
    """
    Get annual report text summary for Ollama analysis.
    US tickers: SEC EDGAR 10-K. NSE/others: returns None (integration pending).
    """
    return _fetch_sec_10k_summary(ticker)
