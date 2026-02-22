# Stock Analysis App

Scalable stock analysis application with **FastAPI** backend and **Streamlit** frontend. Computes annual return, volatility, Value at Risk (VaR), RSI, and MACD for a given ticker.

## Architecture

```
project_root/
├── app/
│   ├── main.py              # FastAPI app
│   ├── domain/
│   │   ├── models.py        # StockMetrics, TickerRequest
│   │   ├── entities.py      # StockPriceData
│   │   └── interfaces.py   # IStockRepository, IStockAnalysisService
│   ├── services/
│   │   └── stock_analysis_service.py
│   ├── repositories/
│   │   └── stock_repository.py
│   ├── api/
│   │   └── routes.py
│   └── utils/
│       └── logger.py
├── streamlit_app.py        # Frontend (input + display only)
├── tests/
│   ├── test_models.py
│   ├── test_services.py
│   └── test_api.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
cd project_root
pip install -r requirements.txt
```

### 2. Run FastAPI backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Run Streamlit frontend

```bash
streamlit run streamlit_app.py
```

### 4. Use the app

- Open the Streamlit URL (default: http://localhost:8501)
- Enter a ticker (e.g. AAPL, MSFT)
- Click **Submit** to fetch annual return, volatility, VaR, RSI, and MACD

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/analyze/{ticker}` | Analyze single ticker |
| POST | `/api/v1/analyze/batch` | Analyze multiple tickers (body: `{"tickers": ["AAPL","MSFT"]}`) |

### Example: Single ticker

```bash
curl http://localhost:8000/api/v1/analyze/AAPL
```

### Example: Batch (multiple tickers)

```bash
curl -X POST http://localhost:8000/api/v1/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL","MSFT","GOOGL"]}'
```

## Environment

- `STOCK_API_URL`: API base URL for Streamlit (default: `http://localhost:8000`)
- `LOG_DIR`: Directory for log files (default: `logs`)
- `LOG_FILE`: Log filename (default: `stock_analysis.log`)
- `LOG_MAX_BYTES`: Max size per log file in bytes before rotation (default: 5 MB)
- `LOG_BACKUP_COUNT`: Number of rotated log files to keep (default: 5)

## Tests

```bash
pytest tests/ -v
```

## Design Principles

- **Separation of concerns**: Domain, services, repositories, API, and frontend are layered
- **Sync FastAPI**: All routes use sync functions for straightforward multi-ticker calls
- **Frontend thin**: Streamlit only handles input and display; all logic lives in the API
- **Testable**: Pytest tests for models, services, and API routes
