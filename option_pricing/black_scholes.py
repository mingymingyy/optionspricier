"""Black-Scholes-Merton option pricing and Greeks.

Every function is vectorised: pass plain floats or NumPy arrays for any
argument and you get back the same shape.

Conventions used throughout this module:

    S      spot price of the underlying
    K      strike price
    T      time to expiry, in YEARS (30 days -> 30 / 365)
    r      continuously compounded risk-free rate, as a decimal (0.045 = 4.5%)
    q      continuous dividend yield, as a decimal (0 for a non-payer)
    sigma  annualised volatility, as a decimal (0.28 = 28%)

The Greeks are returned in "raw" mathematical units - the partial derivative
with respect to one full unit of the variable, and per year for theta. Traders
quote them rescaled, so `scaled_greeks()` converts vega to "per 1 vol point"
and theta to "per calendar day".
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "d1_d2",
    "bs_price",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "greeks",
    "scaled_greeks",
]

# Guards against log(0) and division by zero when T or sigma collapse to zero.
_TINY = 1e-12

_CALL = "call"
_PUT = "put"


def _check_type(option_type: str) -> str:
    kind = str(option_type).strip().lower()
    if kind not in (_CALL, _PUT):
        raise ValueError("option_type must be 'call' or 'put', got " + repr(option_type))
    return kind


def _as_arrays(*args):
    return [np.asarray(a, dtype=float) for a in args]


def _unwrap(x):
    """Return a Python float for 0-d results so scalar inputs give scalar output."""
    arr = np.asarray(x, dtype=float)
    return float(arr) if arr.ndim == 0 else arr


def d1_d2(S, K, T, r, sigma, q=0.0):
    """The two standardised moneyness terms behind every Black-Scholes formula.

        d1 = [ln(S/K) + (r - q + sigma^2 / 2) T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

    N(d2) is the risk-neutral probability that a call expires in the money;
    N(d1) is that same probability under the share-price numeraire, which is
    why it also turns out to be the (undiscounted) delta.
    """
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)

    total_vol = np.maximum(sigma, _TINY) * np.sqrt(np.maximum(T, _TINY))
    moneyness = np.log(np.maximum(S, _TINY) / np.maximum(K, _TINY))

    d1 = (moneyness + (r - q) * T) / total_vol + 0.5 * total_vol
    d2 = d1 - total_vol
    return d1, d2


def _degenerate(T, sigma):
    """True where the option has no time value left (expired, or zero vol)."""
    T, sigma = _as_arrays(T, sigma)
    return (T <= 0) | (sigma <= 0)


def bs_price(S, K, T, r, sigma, q=0.0, option_type=_CALL):
    """Black-Scholes-Merton price of a European call or put.

        call = S e^(-qT) N(d1) - K e^(-rT) N(d2)
        put  = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)

    With T = 0 or sigma = 0 there is no optionality left, so the function falls
    back to the discounted intrinsic value max(S e^(-qT) - K e^(-rT), 0).
    """
    kind = _check_type(option_type)
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)

    disc_r = np.exp(-r * np.maximum(T, 0.0))
    disc_q = np.exp(-q * np.maximum(T, 0.0))

    if kind == _CALL:
        value = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
        floor = np.maximum(S * disc_q - K * disc_r, 0.0)
    else:
        value = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)
        floor = np.maximum(K * disc_r - S * disc_q, 0.0)

    return _unwrap(np.where(_degenerate(T, sigma), floor, value))


def delta(S, K, T, r, sigma, q=0.0, option_type=_CALL):
    """dV/dS - the hedge ratio, i.e. shares of stock per option held."""
    kind = _check_type(option_type)
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * np.maximum(T, 0.0))

    if kind == _CALL:
        value = disc_q * norm.cdf(d1)
        expired = np.where(S > K, 1.0, 0.0)
    else:
        value = disc_q * (norm.cdf(d1) - 1.0)
        expired = np.where(S < K, -1.0, 0.0)

    return _unwrap(np.where(_degenerate(T, sigma), expired, value))


def gamma(S, K, T, r, sigma, q=0.0):
    """d2V/dS2 - how fast delta moves. Identical for calls and puts."""
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * np.maximum(T, 0.0))

    denom = (
        np.maximum(S, _TINY)
        * np.maximum(sigma, _TINY)
        * np.sqrt(np.maximum(T, _TINY))
    )
    value = disc_q * norm.pdf(d1) / denom
    return _unwrap(np.where(_degenerate(T, sigma), 0.0, value))


def vega(S, K, T, r, sigma, q=0.0):
    """dV/dsigma per 1.00 (i.e. 100 vol points) of vol. Same for calls and puts.

    Divide by 100 for the more useful "P&L per 1 vol point" number.
    """
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * np.maximum(T, 0.0))

    value = S * disc_q * norm.pdf(d1) * np.sqrt(np.maximum(T, 0.0))
    return _unwrap(np.where(_degenerate(T, sigma), 0.0, value))


def theta(S, K, T, r, sigma, q=0.0, option_type=_CALL):
    """dV/dT per YEAR (negative for most long options). Divide by 365 for per-day.

    Three economic pieces: decay of the optionality (always negative for a long
    option), carry on the discounted strike, and dividend drag on the spot leg.
    """
    kind = _check_type(option_type)
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)

    disc_r = np.exp(-r * np.maximum(T, 0.0))
    disc_q = np.exp(-q * np.maximum(T, 0.0))
    decay = -(S * disc_q * norm.pdf(d1) * np.maximum(sigma, _TINY)) / (
        2.0 * np.sqrt(np.maximum(T, _TINY))
    )

    if kind == _CALL:
        value = decay - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
    else:
        value = decay + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)

    return _unwrap(np.where(_degenerate(T, sigma), 0.0, value))


def rho(S, K, T, r, sigma, q=0.0, option_type=_CALL):
    """dV/dr per 1.00 of rates. Divide by 100 for the P&L per 1% move."""
    kind = _check_type(option_type)
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * np.maximum(T, 0.0))

    if kind == _CALL:
        value = K * np.maximum(T, 0.0) * disc_r * norm.cdf(d2)
    else:
        value = -K * np.maximum(T, 0.0) * disc_r * norm.cdf(-d2)

    return _unwrap(np.where(_degenerate(T, sigma), 0.0, value))


def greeks(S, K, T, r, sigma, q=0.0, option_type=_CALL) -> dict:
    """Price plus the five Greeks, all in raw units, as a dict."""
    kind = _check_type(option_type)
    return {
        "price": bs_price(S, K, T, r, sigma, q, kind),
        "delta": delta(S, K, T, r, sigma, q, kind),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q),
        "theta": theta(S, K, T, r, sigma, q, kind),
        "rho": rho(S, K, T, r, sigma, q, kind),
    }


def scaled_greeks(
    S, K, T, r, sigma, q=0.0, option_type=_CALL, days_per_year: float = 365.0
) -> dict:
    """The same Greeks in the units a trading desk actually quotes.

    delta and gamma are unchanged, vega becomes P&L per 1 vol point, theta
    becomes P&L per calendar day, and rho becomes P&L per 1% move in rates.
    """
    raw = greeks(S, K, T, r, sigma, q, option_type)
    return {
        "price": raw["price"],
        "delta": raw["delta"],
        "gamma": raw["gamma"],
        "vega": raw["vega"] / 100.0,
        "theta": raw["theta"] / days_per_year,
        "rho": raw["rho"] / 100.0,
    }
