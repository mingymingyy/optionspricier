"""Option pricing models.

Three ways to price the same European contract, plus the tools to interrogate
the result:

* `black_scholes` - the closed form, and the analytic Greeks that come with it
* `binomial`      - a Cox-Ross-Rubinstein lattice, which also handles American exercise
* `monte_carlo`   - risk-neutral simulation, with a standard error on the estimate
* `implied_vol`   - inverts Black-Scholes to recover the volatility a quote implies
* `market_data`   - Yahoo Finance helpers for spot, history and option chains

All three pricers share a signature, `(S, K, T, r, sigma, q, option_type)`, so
they can be compared against one another directly.
"""

from .base import AMERICAN, CALL, EUROPEAN, PUT, payoff
from .binomial import binomial_price, lattice_parameters
from .black_scholes import (
    bs_price,
    d1_d2,
    delta,
    gamma,
    greeks,
    rho,
    scaled_greeks,
    theta,
    vega,
)
from .implied_vol import (
    ImpliedVolError,
    ImpliedVolResult,
    implied_vol,
    implied_vol_bisection,
    implied_vol_newton,
    price_bounds,
)
from .monte_carlo import MonteCarloResult, monte_carlo_price, simulate_paths

__all__ = [
    "CALL",
    "PUT",
    "EUROPEAN",
    "AMERICAN",
    "payoff",
    "bs_price",
    "d1_d2",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "greeks",
    "scaled_greeks",
    "binomial_price",
    "lattice_parameters",
    "monte_carlo_price",
    "simulate_paths",
    "MonteCarloResult",
    "implied_vol",
    "implied_vol_newton",
    "implied_vol_bisection",
    "price_bounds",
    "ImpliedVolError",
    "ImpliedVolResult",
]
