"""Numerical implied-volatility solver.

Black-Scholes takes volatility as an input and returns a price. In practice the
price is the thing you observe on a screen and volatility is the thing you want
to know, so you have to invert the formula:

    solve for sigma such that    BS(S, K, T, r, q, sigma) = C_market

There is no closed form, so this module solves it numerically. Two methods are
implemented because each fails where the other works:

* Newton-Raphson uses vega (the analytic derivative of price w.r.t. sigma) and
  converges quadratically - usually 3-4 iterations. It breaks down when vega is
  near zero, which happens for deep in- or out-of-the-money options and for
  options very close to expiry: the price barely responds to volatility, so the
  Newton step explodes.
* Bisection only needs the price to be monotonically increasing in sigma (it
  always is, since vega > 0). It cannot diverge, but it needs ~50 iterations to
  reach the same precision.

`implied_vol()` runs Newton and silently falls back to bisection if Newton
wanders outside the bracket or stalls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .black_scholes import bs_price, vega

__all__ = [
    "ImpliedVolError",
    "ImpliedVolResult",
    "price_bounds",
    "implied_vol",
    "implied_vol_bisection",
    "implied_vol_newton",
]

# Volatilities outside this bracket are not economically meaningful for listed
# equity options; 500% is already far beyond anything quoted.
VOL_MIN = 1e-6
VOL_MAX = 5.0


class ImpliedVolError(ValueError):
    """Raised when no volatility can reproduce the observed price."""


@dataclass
class ImpliedVolResult:
    """What the solver found, plus enough detail to sanity-check it."""

    sigma: float
    iterations: int
    method: str
    price_error: float  # BS(sigma) - market price, in currency units

    @property
    def sigma_pct(self) -> float:
        return self.sigma * 100.0


def price_bounds(S, K, T, r, q=0.0, option_type="call"):
    """No-arbitrage bounds on a European option price.

    A call must trade between its discounted intrinsic value (at sigma -> 0)
    and the discounted spot (the sigma -> infinity limit). A quote outside this
    range cannot be inverted - it usually means a stale or crossed market, or
    the wrong rate / dividend assumption.
    """
    T = max(float(T), 0.0)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    if str(option_type).lower() == "call":
        lower = max(S * disc_q - K * disc_r, 0.0)
        upper = S * disc_q
    else:
        lower = max(K * disc_r - S * disc_q, 0.0)
        upper = K * disc_r

    return float(lower), float(upper)


def _validate(price, S, K, T, r, q, option_type):
    if T <= 0:
        raise ImpliedVolError(
            "Option has expired (T = 0), so its price carries no volatility information."
        )
    if price <= 0:
        raise ImpliedVolError("Market price must be positive.")

    lower, upper = price_bounds(S, K, T, r, q, option_type)
    if price < lower - 1e-10:
        raise ImpliedVolError(
            "Price {:.4f} is below the no-arbitrage floor of {:.4f} "
            "(intrinsic value).".format(price, lower)
        )
    if price > upper + 1e-10:
        raise ImpliedVolError(
            "Price {:.4f} is above the no-arbitrage ceiling of {:.4f}.".format(price, upper)
        )
    return lower, upper


def implied_vol_bisection(
    price, S, K, T, r, q=0.0, option_type="call", tol=1e-8, max_iter=200
) -> ImpliedVolResult:
    """Invert Black-Scholes by bisection. Slow but it cannot fail to converge.

    Relies on one fact only: option price is strictly increasing in sigma, so
    the sign of (model price - market price) tells us which half to keep.
    """
    _validate(price, S, K, T, r, q, option_type)

    lo, hi = VOL_MIN, VOL_MAX
    f_lo = bs_price(S, K, T, r, lo, q, option_type) - price
    f_hi = bs_price(S, K, T, r, hi, q, option_type) - price

    if f_lo > 0 or f_hi < 0:
        raise ImpliedVolError(
            "No volatility in [{:.0%}, {:.0%}] reproduces this price.".format(VOL_MIN, VOL_MAX)
        )

    mid = 0.5 * (lo + hi)
    for i in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        f_mid = bs_price(S, K, T, r, mid, q, option_type) - price

        if abs(f_mid) < tol or (hi - lo) < tol:
            return ImpliedVolResult(mid, i, "bisection", f_mid)

        if f_mid < 0:  # model too cheap -> need more vol
            lo = mid
        else:
            hi = mid

    return ImpliedVolResult(
        mid, max_iter, "bisection", bs_price(S, K, T, r, mid, q, option_type) - price
    )


def implied_vol_newton(
    price, S, K, T, r, q=0.0, option_type="call", tol=1e-8, max_iter=50
) -> ImpliedVolResult:
    """Invert Black-Scholes by Newton-Raphson, using vega as the derivative.

        sigma_{n+1} = sigma_n - [BS(sigma_n) - price] / vega(sigma_n)

    Raises ImpliedVolError if the iteration stalls (vega collapses) or leaves
    the sensible vol bracket - the caller is expected to fall back to bisection.
    """
    _validate(price, S, K, T, r, q, option_type)

    # Brenner-Subrahmanyam seed: for an at-the-money option the BS price is
    # approximately 0.4 * S * sigma * sqrt(T), which rearranges to this.
    sigma = float(np.sqrt(2.0 * np.pi / T) * price / S)
    sigma = float(np.clip(sigma, 0.05, 2.0))

    for i in range(1, max_iter + 1):
        diff = bs_price(S, K, T, r, sigma, q, option_type) - price
        if abs(diff) < tol:
            return ImpliedVolResult(sigma, i, "newton", diff)

        v = vega(S, K, T, r, sigma, q)
        if v < 1e-8:  # price is insensitive to vol here - Newton is hopeless
            raise ImpliedVolError("Vega collapsed to zero; Newton cannot converge.")

        sigma -= diff / v
        if not (VOL_MIN <= sigma <= VOL_MAX) or not np.isfinite(sigma):
            raise ImpliedVolError("Newton iteration left the volatility bracket.")

    raise ImpliedVolError("Newton did not converge in {} iterations.".format(max_iter))


def implied_vol(
    price, S, K, T, r, q=0.0, option_type="call", tol=1e-8
) -> ImpliedVolResult:
    """Implied volatility: fast Newton first, robust bisection as a safety net."""
    try:
        return implied_vol_newton(price, S, K, T, r, q, option_type, tol=tol)
    except ImpliedVolError:
        # Newton failed on a hard case (deep ITM/OTM, near expiry). Bisection
        # still works because the price is monotone in sigma.
        return implied_vol_bisection(price, S, K, T, r, q, option_type, tol=tol)
