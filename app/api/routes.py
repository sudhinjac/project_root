"""FastAPI routes for stock analysis."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.repositories.stock_repository import StockRepository
from app.services.stock_analysis_service import StockAnalysisService

router = APIRouter(prefix="/api/v1", tags=["stock-analysis"])

# Dependency injection - sync services
_stock_repo = StockRepository()
_stock_service = StockAnalysisService(_stock_repo)


class BatchRequest(BaseModel):
    """Request body for batch analysis."""

    tickers: list[str]


@router.get("/analyze/{ticker}")
def analyze_ticker(ticker: str) -> dict:
    """
    Analyze a single ticker. Returns annual return, volatility, VaR, RSI, MACD.

    Use this endpoint for single-ticker analysis. For multiple tickers,
    call this endpoint in a loop or use the batch endpoint when available.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker cannot be empty")

    result = _stock_service.analyze_ticker(ticker)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No data or analysis available for ticker: {ticker}",
        )
    return result


@router.post("/analyze/batch")
def analyze_batch(req: BatchRequest) -> dict:
    """
    Analyze multiple tickers. Returns a dict mapping ticker -> metrics.

    Designed for future multi-ticker API calls from external clients.
    """
    tickers = [t.strip().upper() for t in (req.tickers or []) if t and t.strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="At least one ticker required")

    results = {}
    errors = []

    for ticker in tickers:
        result = _stock_service.analyze_ticker(ticker)
        if result:
            results[ticker] = result
        else:
            errors.append(ticker)

    return {
        "data": results,
        "errors": errors,
        "success_count": len(results),
        "error_count": len(errors),
    }
