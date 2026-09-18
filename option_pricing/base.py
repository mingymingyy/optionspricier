"""Shared vocabulary for the three pricing models.

Every model in this package prices the same contract and takes the same
arguments in the same order:

    S      spot price of the underlying
    K      strike price
    T      time to expiry, in YEARS
    r      continuously compounded risk-free rate, as a decimal
    sigma  annualised volatility, as a decimal
    q      continuous dividend yield, as a decimal

so that `black_scholes_price(...)`, `binomial_price(...)` and
`monte_carlo_price(...)` are drop-in comparisons for one another.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "CALL",
    "PUT",
    "EUROPEAN",
    "AMERICAN",
    "OPTION_TYPES",
    "EXERCISE_STYLES",
    "normalise_option_type",
    "normalise_exercise",
    "payoff",
]

CALL = "call"
PUT = "put"
OPTION_TYPES = (CALL, PUT)

EUROPEAN = "european"
AMERICAN = "american"
EXERCISE_STYLES = (EUROPEAN, AMERICAN)


def normalise_option_type(option_type) -> str:
    """Accept 'Call', 'CALL', 'call' and return the canonical 'call'."""
    kind = str(option_type).strip().lower()
    if kind not in OPTION_TYPES:
        raise ValueError("option_type must be 'call' or 'put', got " + repr(option_type))
    return kind


def normalise_exercise(exercise) -> str:
    """Accept 'European', 'american', etc. and return the canonical form."""
    style = str(exercise).strip().lower()
    if style not in EXERCISE_STYLES:
        raise ValueError(
            "exercise must be 'european' or 'american', got " + repr(exercise)
        )
    return style


def payoff(S, K, option_type=CALL):
    """Intrinsic value at exercise: max(S - K, 0) for a call, max(K - S, 0) for a put."""
    kind = normalise_option_type(option_type)
    S = np.asarray(S, dtype=float)
    if kind == CALL:
        return np.maximum(S - K, 0.0)
    return np.maximum(K - S, 0.0)
