# Stock Analysis App

Scalable stock analysis application with **FastAPI** backend and **Streamlit** frontends. Includes a **Comprehensive Stock Analysis** dashboard with financials, technicals, sentiment, and AI perspective.

## Architecture

```
project_root/
├── app/
│   ├── main.py                    # FastAPI app
│   ├── domain/
│   │   ├── models.py              # StockMetrics, TickerRequest
│   │   ├── entities.py            # StockPriceData, FinancialStatements, SentimentResult
│   │   └── interfaces.py          # IStockRepository, IFinancialRepository, INewsRepository,
│   │                              # IStockAnalysisService, IFinancialRatioService,
│   │                              # ISentimentService, ILLMService
│   ├── services/
│   │   ├── stock_analysis_service.py      # Basic metrics (RSI, MACD, VaR)
│   │   ├── financial_ratio_service.py     # ROCE, ROE, ROA, margins, ratios
│   │   ├── technical_analysis_service.py # CAGR, Sharpe, Monte Carlo, ARIMA, HMM, ML
│   │   ├── sentiment_service.py          # VADER sentiment on news
│   │   └── ollama_llm_service.py         # Ollama/DeepSeek company perspective
│   ├── repositories/
│   │   ├── stock_repository.py            # Yahoo Finance price data
│   │   ├── financial_repository.py        # Balance sheet, P&L, cashflow
│   │   └── news_repository.py             # Google News RSS
│   ├── api/
│   │   └── routes.py
│   └── utils/
│       └── logger.py
├── streamlit_app.py               # Basic frontend (API-backed)
├── streamlit_comprehensive.py     # Comprehensive dashboard (standalone)
├── tests/
│   ├── test_models.py
│   ├── test_services.py
│   ├── test_api.py
│   ├── test_financial_service.py
│   ├── test_technical_service.py
│   └── test_sentiment_service.py
├── requirements.txt
└── README.md
```

## Architecture Flow (Comprehensive App)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         streamlit_comprehensive.py                          │
│                    (User Input: Ticker, Company Name)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────────┐           ┌─────────────────┐
│ Financial     │           │ Technical          │           │ Sentiment       │
│ RatioService │           │ AnalysisService    │           │ Service         │
│               │           │                    │           │                 │
│ • ROCE, ROE   │           │ • CAGR, Sharpe     │           │ • VADER on      │
│ • ROA         │           │ • Monte Carlo      │           │   news headlines│
│ • Margins     │           │ • ARIMA/HW forecast│           │ • Pie chart     │
└───────┬───────┘           │ • HMM regime      │           └────────┬────────┘
        │                    │ • ML prediction   │                    │
        │                    └─────────┬─────────┘                    │
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────┐           ┌───────────────────┐           ┌─────────────────┐
│ YahooFinancial│           │ StockRepository   │           │ GoogleNews       │
│ Repository    │           │ (Yahoo Finance     │           │ Repository      │
│               │           │  price data)       │           │ (RSS feed)      │
└───────────────┘           └───────────────────┘           └─────────────────┘
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                      ▼
                        ┌─────────────────────────┐
                        │ OllamaLLMService        │
                        │ (DeepSeek via Ollama)    │
                        │                          │
                        │ Aggregates all data →    │
                        │ AI company perspective  │
                        └─────────────────────────┘
```

## Design Principles

- **DRY**: Calculations from `tech5.py` and `ALKYLAMINES.ipynb` centralized in services
- **Dependency Injection**: Services receive repositories via constructor
- **Repository Pattern**: Data access abstracted behind interfaces
- **Single Responsibility**: Each service handles one domain (financial, technical, sentiment, LLM)

## Quick Start

### 1. Install dependencies

```bash
cd project_root
pip install -r requirements.txt
```

### 2. (Optional) Run FastAPI backend (for basic streamlit_app.py)

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Run Comprehensive Streamlit App

```bash
streamlit run streamlit_comprehensive.py
```

### 4. (Optional) Ollama for AI Perspective

```bash
# Install Ollama from https://ollama.ai
ollama pull deepseek
```

Set `OLLAMA_MODEL=deepseek` if using a different model name.

## Comprehensive App Features

| Section | Content |
|---------|---------|
| **Fundamental** | P&L, Balance Sheet, Cash Flow (full tables); ROCE, ROE, ROA, margins, Debt/Equity, Current Ratio |
| **Technical** | CAGR, Volatility, Sharpe, Sortino, Max Drawdown, Calmar; Beta, Jensen's Alpha, Treynor |
| **Signals** | RSI, MACD, Bollinger, ADX, Ichimoku; Buy/Hold/Sell decision |
| **Charts** | Candlestick + Bollinger + SMA + Volume; RSI; MACD |
| **Monte Carlo** | Price simulation paths; min/max/mean; 5th/95th percentiles |
| **Forecast** | ARIMA and Holt-Winters 30-day forecast |
| **HMM** | Regime detection (Bull/Bear) |
| **ML** | Logistic regression bullish/bearish probability |
| **Sentiment** | News headlines; Positive/Neutral/Negative pie chart |
| **AI** | Ollama DeepSeek expert financial analyst perspective (optional); receives full financial data |

## API Endpoints (Basic App)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/analyze/{ticker}` | Analyze single ticker |
| GET | `/api/v1/analyze/{ticker}/indicators` | RSI & MACD time series |
| POST | `/api/v1/analyze/batch` | Analyze multiple tickers |

## Dependencies

Core: FastAPI, Uvicorn, Streamlit, Plotly, Pandas, NumPy, SciPy, yfinance, requests, httpx.

Comprehensive app: feedparser, nltk, statsmodels, scikit-learn, hmmlearn, ollama, beautifulsoup4, matplotlib, vaderSentiment.

See `requirements.txt` for versions.

## Environment

- `STOCK_API_URL`: API base URL for basic Streamlit (default: `http://localhost:8000`)
- `OLLAMA_MODEL`: Ollama model for AI perspective (default: `deepseek`)
- `LOG_DIR`, `LOG_FILE`, `LOG_MAX_BYTES`, `LOG_BACKUP_COUNT`: Logging

## Tests

```bash
pytest tests/ -v
```

## References

- `tech5.py`: Technical indicators, Monte Carlo, ARIMA, HMM, sentiment, ML prediction
- `ALKYLAMINES.ipynb`: CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, beta, Monte Carlo
