"""Technical analysis service - metrics and indicators from tech5.py and ALKYLAMINES.ipynb."""

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from hmmlearn.hmm import GaussianHMM

from app.domain.interfaces import IStockRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)

BENCHMARK = "^NSEI"
RF = 0.07
TRADING_DAYS = 252


def _cagr(df: pd.DataFrame) -> float:
    """Cumulative Annual Growth Rate."""
    d = df.copy()
    d["daily_ret"] = d["Close"].pct_change()
    d["cum_return"] = (1 + d["daily_ret"]).cumprod()
    n = len(d) / TRADING_DAYS
    if n <= 0:
        return 0.0
    return float((d["cum_return"].iloc[-1] ** (1 / n)) - 1)


def _volatility(df: pd.DataFrame) -> float:
    """Annualized volatility."""
    ret = df["Close"].pct_change()
    return float(ret.std() * np.sqrt(TRADING_DAYS))


def _sortino(df: pd.DataFrame, rf: float = RF) -> float:
    """Sortino ratio."""
    ret = df["Close"].pct_change()
    neg_vol = ret[ret < 0].std() * np.sqrt(TRADING_DAYS)
    if neg_vol == 0:
        return 0.0
    return float((_cagr(df) - rf) / neg_vol)


def _max_drawdown(df: pd.DataFrame) -> float:
    """Maximum drawdown."""
    d = df.copy()
    d["cum_return"] = (1 + d["Close"].pct_change()).cumprod()
    d["cum_roll_max"] = d["cum_return"].cummax()
    d["drawdown_pct"] = (d["cum_roll_max"] - d["cum_return"]) / d["cum_roll_max"]
    return float(d["drawdown_pct"].max())


def _calmar(df: pd.DataFrame) -> float:
    """Calmar ratio."""
    md = _max_drawdown(df)
    if md == 0:
        return 0.0
    return float(_cagr(df) / md)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI indicator."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, short: int = 12, long: int = 26, signal: int = 9) -> tuple:
    """MACD line and signal."""
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def _bollinger(close: pd.Series, period: int = 20) -> tuple:
    """Bollinger bands."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return upper, lower


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return tr.rolling(period).mean()


class TechnicalAnalysisService:
    """Service for technical metrics, indicators, predictions, and regime detection."""

    def __init__(self, repo: IStockRepository) -> None:
        self._repo = repo

    def get_stock_metrics(self, ticker: str, benchmark: str = BENCHMARK) -> Optional[dict]:
        """Beta, Sharpe, Treynor, Jensen's Alpha, CV, loss/profit probability, odds."""
        try:
            bench_data = self._repo.get_price_data(benchmark, "3y")
            if bench_data is None:
                bench_data = self._repo.get_price_data("^GSPC", "3y")
            if bench_data is None:
                return None

            stock_data = self._repo.get_price_data(ticker, "3y")
            if not stock_data:
                return None

            data = pd.DataFrame()
            data[benchmark] = bench_data.data["Close"]
            data[ticker] = stock_data.data["Close"]
            data = data.dropna()

            market_returns = np.log(data[benchmark] / data[benchmark].shift(1)).dropna()
            sec_returns = np.log(data[ticker] / data[ticker].shift(1)).dropna()
            aligned = pd.concat([market_returns, sec_returns], axis=1).dropna()

            mr = aligned.iloc[:, 0]
            sr = aligned.iloc[:, 1]
            market_var = mr.var() * TRADING_DAYS
            cov = aligned.cov().iloc[0, 1] * TRADING_DAYS
            beta = cov / market_var if market_var != 0 else 0
            expected_return = sr.mean() * TRADING_DAYS * 100
            vols = sr.std() * np.sqrt(TRADING_DAYS) * 100
            sharpe = (expected_return - RF * 100) / vols if vols != 0 else 0
            treynor = (expected_return - RF * 100) / beta if beta != 0 else 0
            jensen_alpha = expected_return - (RF * 100 + beta * (mr.mean() * TRADING_DAYS * 100 - RF * 100))
            cv = vols / expected_return * 100 if expected_return != 0 else 0
            z_score = (0 - expected_return) / vols if vols != 0 else 0
            loss_prob = stats.norm.cdf(z_score) * 100
            profit_prob = (1 - stats.norm.cdf(z_score)) * 100
            ow = profit_prob / loss_prob if loss_prob > 0 else float("inf")

            return {
                "beta": beta,
                "sharpe_ratio": sharpe,
                "treynor_ratio": treynor,
                "jensen_alpha": jensen_alpha,
                "coefficient_of_variation": cv,
                "loss_probability_pct": loss_prob,
                "profit_probability_pct": profit_prob,
                "odds_of_profit": ow,
                "volatility_pct": vols,
            }
        except Exception as e:
            logger.exception("Error computing stock metrics for %s: %s", ticker, e)
            return None

    def get_price_metrics(self, ticker: str) -> Optional[dict]:
        """CAGR, Volatility, Sharpe, Sortino, Max Drawdown, Calmar."""
        price_data = self._repo.get_price_data(ticker, "3y")
        if not price_data or price_data.is_empty:
            return None
        df = price_data.data.copy()
        if "Close" not in df.columns:
            return None
        df = df[["Close"]].copy()
        if "High" in price_data.data.columns and "Low" in price_data.data.columns:
            df["High"] = price_data.data["High"]
            df["Low"] = price_data.data["Low"]
        df.dropna(inplace=True)
        if len(df) < 30:
            return None

        cagr_val = _cagr(df)
        vol = _volatility(df)
        sharpe = (cagr_val - RF) / vol if vol != 0 else 0
        sortino_val = _sortino(df, RF)
        max_dd = _max_drawdown(df)
        calmar_val = _calmar(df)

        return {
            "CAGR (%)": cagr_val * 100,
            "Volatility (%)": vol * 100,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino_val,
            "Max Drawdown (%)": max_dd * 100,
            "Calmar Ratio": calmar_val,
        }

    def get_full_analysis(
        self,
        ticker: str,
        include_forecast: bool = True,
        include_hmm: bool = True,
        include_ml_prediction: bool = True,
    ) -> Optional[dict]:
        """
        Full technical analysis: indicators, signals, Monte Carlo, ARIMA/HW forecast,
        HMM regime, ML prediction.
        """
        price_data = self._repo.get_price_data(ticker, "3y")
        if not price_data or price_data.is_empty:
            return None

        df = price_data.data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        required = ["Open", "High", "Low", "Close", "Volume"]
        for c in required:
            if c not in df.columns:
                df[c] = np.nan
        df = df[required].dropna(how="all")
        df["Close"] = df["Close"].ffill().bfill()
        df = df.dropna(subset=["Close"])
        if len(df) < 100:
            return None

        result: dict = {}

        # Price metrics
        result["price_metrics"] = self.get_price_metrics(ticker)
        result["stock_metrics"] = self.get_stock_metrics(ticker)

        # Indicators
        df["RSI"] = _rsi(df["Close"])
        macd_line, signal_line = _macd(df["Close"])
        df["MACD"] = macd_line
        df["Signal"] = signal_line
        upper, lower = _bollinger(df["Close"])
        df["BB_Upper"] = upper
        df["BB_Lower"] = lower
        df["20SMA"] = df["Close"].rolling(20).mean()
        df["50SMA"] = df["Close"].rolling(50).mean()
        df["100EWMA"] = df["Close"].ewm(span=100, adjust=False).mean()
        df["ATR"] = _atr(df)

        # ATR-based +DI, -DI, ADX
        tr = np.maximum(
            df["High"] - df["Low"],
            np.maximum(
                abs(df["High"] - df["Close"].shift(1)),
                abs(df["Low"] - df["Close"].shift(1)),
            ),
        )
        atr = tr.rolling(14).mean()
        plus_dm = df["High"].diff()
        minus_dm = -df["Low"].diff()
        df["+DI"] = 100 * (plus_dm.rolling(14).mean() / atr)
        df["-DI"] = 100 * (minus_dm.rolling(14).mean() / atr)
        df["ADX"] = 100 * np.abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])

        # Ichimoku
        high_9 = df["High"].rolling(9).max()
        low_9 = df["Low"].rolling(9).min()
        df["Tenkan"] = (high_9 + low_9) / 2
        high_26 = df["High"].rolling(26).max()
        low_26 = df["Low"].rolling(26).min()
        df["Kijun"] = (high_26 + low_26) / 2
        df["SpanA"] = ((df["Tenkan"] + df["Kijun"]) / 2).shift(26)
        high_52 = df["High"].rolling(52).max()
        low_52 = df["Low"].rolling(52).min()
        df["SpanB"] = ((high_52 + low_52) / 2).shift(26)
        df["Chikou"] = df["Close"].shift(-26)

        latest = df.iloc[-1]
        result["signals"] = {
            "RSI": "Bullish" if latest["RSI"] < 30 else "Bearish" if latest["RSI"] > 70 else "Neutral",
            "MACD": "Bullish" if latest["MACD"] > latest["Signal"] else "Bearish",
            "Bollinger": "Bullish" if latest["Close"] < latest["BB_Lower"] else "Bearish" if latest["Close"] > latest["BB_Upper"] else "Neutral",
            "ADX": "Strong Trend" if latest["ADX"] > 25 else "Weak/No Trend",
            "Ichimoku": "Bullish" if latest["Close"] > max(latest["SpanA"], latest["SpanB"]) else "Bearish" if latest["Close"] < min(latest["SpanA"], latest["SpanB"]) else "Neutral",
        }

        # Decision (simplified)
        rsi_last = latest["RSI"]
        stoch_k = ((df["Close"] - df["Low"].rolling(14).min()) / (df["High"].rolling(14).max() - df["Low"].rolling(14).min()) * 100).iloc[-1]
        buy_signal = rsi_last < 30 and stoch_k < 20 and latest["MACD"] > latest["Signal"]
        sell_signal = rsi_last > 70 and stoch_k > 80 and latest["MACD"] < latest["Signal"]
        result["decision"] = "Buy now" if buy_signal else "Wait, price may go down" if sell_signal else "Hold"

        # Monte Carlo
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        mu = returns.mean()
        sigma = returns.std()
        S0 = df["Close"].iloc[-1]
        t_intervals, iterations = 250, 1000
        daily_returns = np.exp((mu - 0.5 * sigma**2) + sigma * np.random.standard_normal((t_intervals, iterations)))
        price_list = np.zeros_like(daily_returns)
        price_list[0] = S0
        for t in range(1, t_intervals):
            price_list[t] = price_list[t - 1] * daily_returns[t]
        result["monte_carlo"] = {
            "max_price": float(price_list.max()),
            "min_price": float(price_list.min()),
            "mean_price": float(price_list.mean()),
            "percentile_5": float(np.percentile(price_list, 5)),
            "percentile_95": float(np.percentile(price_list, 95)),
            "price_paths": price_list,
        }

        # ARIMA / Holt-Winters forecast
        if include_forecast and len(df) >= 50:
            try:
                fc_df = df[["Close"]].dropna()
                future_dates = pd.date_range(start=fc_df.index[-1] + pd.Timedelta(days=1), periods=30, freq="B")
                arima_pred = None
                hw_pred = None
                try:
                    arima_model = ARIMA(fc_df["Close"], order=(5, 1, 0))
                    arima_fit = arima_model.fit()
                    arima_pred = arima_fit.forecast(steps=30)
                    arima_pred.index = future_dates
                except Exception:
                    pass
                try:
                    hw_model = ExponentialSmoothing(fc_df["Close"], trend="add", seasonal=None)
                    hw_fit = hw_model.fit()
                    hw_pred = hw_fit.forecast(steps=30)
                    hw_pred.index = future_dates
                except Exception:
                    pass
                result["forecast"] = {"arima": arima_pred, "holt_winters": hw_pred, "historical": fc_df}
            except Exception as e:
                logger.warning("Forecast failed: %s", e)
                result["forecast"] = None

        # HMM regime
        if include_hmm and len(df) >= 100:
            try:
                ret_series = np.log(df["Close"] / df["Close"].shift(1)).dropna()
                X = ret_series.values.reshape(-1, 1)
                model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000)
                model.fit(X)
                df["Regime"] = model.predict(X)
                result["hmm"] = {"regime_series": df["Regime"], "current_regime": int(df["Regime"].iloc[-1])}
            except Exception as e:
                logger.warning("HMM failed: %s", e)
                result["hmm"] = None

        # ML prediction (Logistic Regression)
        if include_ml_prediction and len(df) >= 200:
            try:
                d = df.copy()
                d["Return"] = d["Close"].pct_change()
                d["50_MA"] = d["Close"].rolling(50).mean()
                d["200_MA"] = d["Close"].rolling(200).mean()
                d["RSI_ml"] = _rsi(d["Close"])
                d["MACD_ml"] = d["Close"].ewm(span=12, adjust=False).mean() - d["Close"].ewm(span=26, adjust=False).mean()
                d = d.dropna()
                d["Label"] = (d["Return"] > 0.01).astype(int)
                features = d[["50_MA", "200_MA", "RSI_ml", "MACD_ml"]].values
                labels = d["Label"].values
                X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
                model = LogisticRegression()
                model.fit(X_train, y_train)
                next_day = d[["50_MA", "200_MA", "RSI_ml", "MACD_ml"]].iloc[-1].values.reshape(1, -1)
                proba = model.predict_proba(next_day)[0]
                result["ml_prediction"] = {
                    "bullish_probability": float(proba[1]),
                    "bearish_probability": float(proba[0]),
                }
            except Exception as e:
                logger.warning("ML prediction failed: %s", e)
                result["ml_prediction"] = None

        result["df"] = df
        return result
