"""Sanity tests for the pricer and the implied-vol solver.

Run with `pytest -q`, or plainly with `python test_pricing.py`.

These check the properties that must hold for any correct Black-Scholes
implementation, rather than hard-coding numbers from another library:
put-call parity, analytic Greeks against finite differences, and a round trip
price -> implied vol -> price.
"""

from __future__ import annotations

import numpy as np

from black_scholes import bs_price, delta, gamma, greeks, theta, vega
from implied_vol import (
    ImpliedVolError,
    implied_vol,
    implied_vol_bisection,
    implied_vol_newton,
    price_bounds,
)

# A representative spread of contracts: in, at and out of the money, short and
# long dated, calm and panicked volatility.
CASES = [
    # S,    K,     T,       r,     sigma, q
    (100.0, 100.0, 1.0, 0.05, 0.20, 0.00),
    (250.0, 260.0, 30 / 365, 0.045, 0.284, 0.00),
    (250.0, 200.0, 90 / 365, 0.045, 0.350, 0.015),
    (50.0, 65.0, 2.0, 0.03, 0.60, 0.02),
    (4200.0, 4000.0, 7 / 365, 0.052, 0.12, 0.018),
]


def test_put_call_parity():
    """C - P must equal S e^(-qT) - K e^(-rT) exactly, for any inputs."""
    for S, K, T, r, sigma, q in CASES:
        call = bs_price(S, K, T, r, sigma, q, "call")
        put = bs_price(S, K, T, r, sigma, q, "put")
        expected = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs((call - put) - expected) < 1e-10


def test_prices_respect_no_arbitrage_bounds():
    for S, K, T, r, sigma, q in CASES:
        for kind in ("call", "put"):
            price = bs_price(S, K, T, r, sigma, q, kind)
            lower, upper = price_bounds(S, K, T, r, q, kind)
            assert lower - 1e-10 <= price <= upper + 1e-10


def test_price_is_increasing_in_volatility():
    """Monotonicity in sigma is what makes implied vol unique and bisection safe."""
    for S, K, T, r, _, q in CASES:
        vols = np.linspace(0.01, 2.0, 200)
        for kind in ("call", "put"):
            prices = bs_price(S, K, T, r, vols, q, kind)
            assert np.all(np.diff(prices) > -1e-12)


def test_delta_matches_finite_difference():
    h = 1e-4
    for S, K, T, r, sigma, q in CASES:
        for kind in ("call", "put"):
            up = bs_price(S + h, K, T, r, sigma, q, kind)
            down = bs_price(S - h, K, T, r, sigma, q, kind)
            numeric = (up - down) / (2 * h)
            assert abs(delta(S, K, T, r, sigma, q, kind) - numeric) < 1e-5


def test_gamma_matches_finite_difference():
    h = 1e-3
    for S, K, T, r, sigma, q in CASES:
        up = bs_price(S + h, K, T, r, sigma, q, "call")
        mid = bs_price(S, K, T, r, sigma, q, "call")
        down = bs_price(S - h, K, T, r, sigma, q, "call")
        numeric = (up - 2 * mid + down) / (h**2)
        assert abs(gamma(S, K, T, r, sigma, q) - numeric) < 1e-4 * max(1.0, S / 100)


def test_vega_matches_finite_difference():
    h = 1e-5
    for S, K, T, r, sigma, q in CASES:
        up = bs_price(S, K, T, r, sigma + h, q, "call")
        down = bs_price(S, K, T, r, sigma - h, q, "call")
        numeric = (up - down) / (2 * h)
        assert abs(vega(S, K, T, r, sigma, q) - numeric) < 1e-4 * max(1.0, S / 100)


def test_theta_matches_finite_difference():
    """Theta is dV/dT, so shrinking T must reduce value for a long option."""
    h = 1e-6
    for S, K, T, r, sigma, q in CASES:
        up = bs_price(S, K, T + h, r, sigma, q, "call")
        down = bs_price(S, K, T - h, r, sigma, q, "call")
        numeric = -(up - down) / (2 * h)  # theta is quoted as decay per unit time
        assert abs(theta(S, K, T, r, sigma, q, "call") - numeric) < 1e-3 * max(1.0, S / 100)


def test_gamma_and_vega_are_type_independent():
    for S, K, T, r, sigma, q in CASES:
        call = greeks(S, K, T, r, sigma, q, "call")
        put = greeks(S, K, T, r, sigma, q, "put")
        assert abs(call["gamma"] - put["gamma"]) < 1e-12
        assert abs(call["vega"] - put["vega"]) < 1e-12


def test_delta_bounds():
    for S, K, T, r, sigma, q in CASES:
        assert 0.0 <= delta(S, K, T, r, sigma, q, "call") <= 1.0
        assert -1.0 <= delta(S, K, T, r, sigma, q, "put") <= 0.0


def test_expired_option_is_worth_intrinsic():
    assert bs_price(120.0, 100.0, 0.0, 0.05, 0.3, 0.0, "call") == 20.0
    assert bs_price(80.0, 100.0, 0.0, 0.05, 0.3, 0.0, "call") == 0.0
    assert bs_price(80.0, 100.0, 0.0, 0.05, 0.3, 0.0, "put") == 20.0


def test_implied_vol_round_trip():
    """Price an option at a known vol, then recover that vol from the price."""
    for S, K, T, r, sigma, q in CASES:
        for kind in ("call", "put"):
            price = bs_price(S, K, T, r, sigma, q, kind)
            result = implied_vol(price, S, K, T, r, q, kind)
            assert abs(result.sigma - sigma) < 1e-6


def test_both_solvers_agree():
    for S, K, T, r, sigma, q in CASES:
        price = bs_price(S, K, T, r, sigma, q, "call")
        newton = implied_vol_newton(price, S, K, T, r, q, "call")
        bisect = implied_vol_bisection(price, S, K, T, r, q, "call")
        assert abs(newton.sigma - bisect.sigma) < 1e-6
        assert newton.iterations < bisect.iterations  # Newton is the fast path


def test_implied_vol_rejects_arbitrage_violations():
    # Below intrinsic value.
    try:
        implied_vol(0.01, 250.0, 200.0, 30 / 365, 0.045, 0.0, "call")
    except ImpliedVolError:
        pass
    else:
        raise AssertionError("expected ImpliedVolError below the intrinsic floor")

    # Above the discounted spot.
    try:
        implied_vol(300.0, 250.0, 200.0, 30 / 365, 0.045, 0.0, "call")
    except ImpliedVolError:
        pass
    else:
        raise AssertionError("expected ImpliedVolError above the price ceiling")


def test_vectorised_inputs():
    spots = np.array([90.0, 100.0, 110.0])
    prices = bs_price(spots, 100.0, 1.0, 0.05, 0.2, 0.0, "call")
    assert prices.shape == (3,)
    assert np.all(np.diff(prices) > 0)  # calls are worth more as spot rises


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS  {}".format(name))
            passed += 1
    print("\n{} tests passed.".format(passed))
