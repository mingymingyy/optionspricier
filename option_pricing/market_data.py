"""Market data helpers built on Yahoo Finance.

Deliberately free of any Streamlit import so the functions stay testable and
reusable; the app wraps them in `st.cache_data` for caching.

Yahoo's free feed is delayed and frequently returns no bid/ask outside market
hours, so `option_chain` reports which price each row came from and the caller
decides whether to trust it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "TRADING_DAYS",
    "get_spot",
    "get_history",
    "realised_volatility",
    "get_expiries",
    "get_option_chain",
]

# Volatility is annualised over trading days, not calendar days: returns are
# only generated on days the market is open.
TRADING_DAYS = 252


def get_spot(symbol: str) -> float:
    """Latest available price for a ticker."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    info = ticker.fast_info
    for key in ("last_price", "lastPrice", "previous_close"):
        try:
            value = info[key]
        except (KeyError, TypeError):
            continue
        if value:
            return float(value)

    history = ticker.history(period="5d")
    if history.empty:
        raise ValueError("No price data returned for {!r}.".format(symbol))
    return float(history["Close"].iloc[-1])


def get_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Daily OHLCV history with a tz-naive DatetimeIndex."""
    import yfinance as yf

    data = yf.Ticker(symbol).history(period=period)
    if data.empty:
        raise ValueError("No history returned for {!r}.".format(symbol))

    data = data.copy()
    if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    return data


def realised_volatility(closes, window: int | None = None) -> float:
    """Annualised standard deviation of daily log returns.

    This is the volatility the underlying *has* had. Comparing it with the
    implied volatility the options are quoting is the simplest read on whether
    options look rich or cheap.
    """
    closes = pd.Series(closes).dropna()
    if window:
        closes = closes.iloc[-(int(window) + 1) :]
    if len(closes) < 3:
        raise ValueError("Need at least three closing prices to estimate volatility.")

    log_returns = np.log(closes / closes.shift(1)).dropna()
    return float(log_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def get_expiries(symbol: str) -> list[str]:
    """Available option expiry dates, as 'YYYY-MM-DD' strings."""
    import yfinance as yf

    return list(yf.Ticker(symbol).options)


def get_option_chain(symbol: str, expiry: str) -> pd.DataFrame:
    """Calls and puts for one expiry, stacked into a single frame.

    Adds two derived columns: `price`, the bid-ask mid where a two-sided quote
    exists and the last traded price otherwise, and `source`, saying which of
    the two was used.
    """
    import yfinance as yf

    chain = yf.Ticker(symbol).option_chain(expiry)
    columns = [
        "strike",
        "bid",
        "ask",
        "lastPrice",
        "lastTradeDate",
        "volume",
        "openInterest",
        "impliedVolatility",
    ]

    frames = []
    for kind, table in (("call", chain.calls), ("put", chain.puts)):
        part = table[columns].copy()
        part["type"] = kind
        frames.append(part)

    combined = pd.concat(frames, ignore_index=True)

    quoted = (combined["bid"] > 0) & (combined["ask"] > combined["bid"])
    combined["price"] = np.where(
        quoted, (combined["bid"] + combined["ask"]) / 2.0, combined["lastPrice"]
    )
    combined["source"] = np.where(quoted, "bid/ask mid", "last traded")
    combined["volume"] = combined["volume"].fillna(0)
    combined["openInterest"] = combined["openInterest"].fillna(0)

    return combined
