"""Sanity tests for the three pricing models and the implied-vol solver.

Run with `pytest -q`, or plainly with `python test_pricing.py`.

These check the properties that must hold for any correct implementation,
rather than hard-coding numbers from another library: put-call parity, analytic
Greeks against finite differences, a price -> implied vol -> price round trip,
and agreement between the closed form, the lattice and the simulation.
"""

from __future__ import annotations

import numpy as np

from option_pricing import payoff
from option_pricing.binomial import binomial_price, lattice_parameters
from option_pricing.black_scholes import bs_price, delta, gamma, greeks, theta, vega
from option_pricing.implied_vol import (
    ImpliedVolError,
    implied_vol,
    implied_vol_bisection,
    implied_vol_newton,
    price_bounds,
)
from option_pricing.monte_carlo import monte_carlo_price, simulate_paths

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


# ------------------------------------------------------------ Black-Scholes ---


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


def test_vectorised_inputs():
    spots = np.array([90.0, 100.0, 110.0])
    prices = bs_price(spots, 100.0, 1.0, 0.05, 0.2, 0.0, "call")
    assert prices.shape == (3,)
    assert np.all(np.diff(prices) > 0)  # calls are worth more as spot rises


# ---------------------------------------------------------------- binomial ---


def test_binomial_converges_to_black_scholes():
    """With enough steps the lattice must reproduce the closed form."""
    for S, K, T, r, sigma, q in CASES:
        for kind in ("call", "put"):
            closed_form = bs_price(S, K, T, r, sigma, q, kind)
            lattice = binomial_price(S, K, T, r, sigma, q, kind, steps=2000)
            assert abs(lattice - closed_form) < 0.01 * max(1.0, S / 100)


def test_binomial_error_shrinks_with_steps():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
    exact = bs_price(S, K, T, r, sigma, q, "call")
    coarse = abs(binomial_price(S, K, T, r, sigma, q, "call", steps=10) - exact)
    fine = abs(binomial_price(S, K, T, r, sigma, q, "call", steps=1000) - exact)
    assert fine < coarse


def test_risk_neutral_probability_is_a_probability():
    for S, K, T, r, sigma, q in CASES:
        params = lattice_parameters(T, r, sigma, q, steps=500)
        assert 0.0 < params["p"] < 1.0
        assert params["u"] > 1.0 > params["d"] > 0.0


def test_american_put_is_worth_at_least_european():
    """Early exercise is an extra right, so it can never reduce the value."""
    for S, K, T, r, sigma, q in CASES:
        euro = binomial_price(S, K, T, r, sigma, q, "put", steps=300, exercise="european")
        amer = binomial_price(S, K, T, r, sigma, q, "put", steps=300, exercise="american")
        assert amer >= euro - 1e-9


def test_american_put_premium_is_strictly_positive_when_rates_are_high():
    """A deep in-the-money put on a high rate should be exercised early."""
    euro = binomial_price(100.0, 140.0, 1.0, 0.10, 0.25, 0.0, "put", 500, "european")
    amer = binomial_price(100.0, 140.0, 1.0, 0.10, 0.25, 0.0, "put", 500, "american")
    assert amer > euro + 0.5


def test_american_call_on_non_dividend_payer_equals_european():
    """The classic result: never exercise an American call early without dividends."""
    euro = binomial_price(100.0, 90.0, 1.0, 0.08, 0.30, 0.0, "call", 500, "european")
    amer = binomial_price(100.0, 90.0, 1.0, 0.08, 0.30, 0.0, "call", 500, "american")
    assert abs(amer - euro) < 1e-10


def test_binomial_expired_option_is_intrinsic():
    assert binomial_price(120.0, 100.0, 0.0, 0.05, 0.3, 0.0, "call") == 20.0


# ------------------------------------------------------------- Monte Carlo ---


def test_monte_carlo_agrees_with_black_scholes():
    """The closed-form price should sit inside the simulation's 95% interval."""
    for S, K, T, r, sigma, q in CASES:
        for kind in ("call", "put"):
            exact = bs_price(S, K, T, r, sigma, q, kind)
            estimate = monte_carlo_price(
                S, K, T, r, sigma, q, kind, n_paths=200_000, seed=12345
            )
            assert estimate.ci_low <= exact <= estimate.ci_high


def test_monte_carlo_error_shrinks_as_paths_grow():
    """Standard error falls as 1/sqrt(n): 100x the paths, 10x the precision."""
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0
    small = monte_carlo_price(S, K, T, r, sigma, q, "call", n_paths=1_000, seed=1)
    large = monte_carlo_price(S, K, T, r, sigma, q, "call", n_paths=100_000, seed=1)
    ratio = small.std_error / large.std_error
    assert 5.0 < ratio < 20.0  # theory says 10


def test_antithetic_variates_reduce_variance():
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0
    plain = monte_carlo_price(
        S, K, T, r, sigma, q, "call", n_paths=100_000, antithetic=False, seed=3
    )
    paired = monte_carlo_price(
        S, K, T, r, sigma, q, "call", n_paths=100_000, antithetic=True, seed=3
    )
    assert paired.std_error < plain.std_error


def test_monte_carlo_is_reproducible_with_a_seed():
    args = (100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "call")
    first = monte_carlo_price(*args, n_paths=10_000, seed=42)
    second = monte_carlo_price(*args, n_paths=10_000, seed=42)
    assert first.price == second.price


def test_simulated_paths_start_at_spot_and_have_the_right_shape():
    paths = simulate_paths(100.0, 1.0, 0.05, 0.2, 0.0, n_paths=50, n_steps=30, seed=0)
    assert paths.shape == (31, 50)
    assert np.allclose(paths[0], 100.0)
    assert np.all(paths > 0)  # GBM cannot go negative


def test_monte_carlo_terminal_mean_matches_the_forward():
    """Under the risk-neutral measure, E[S_T] must equal the forward price."""
    S, T, r, q = 100.0, 1.0, 0.05, 0.02
    paths = simulate_paths(S, T, r, 0.2, q, n_paths=400_000, n_steps=1, seed=5)
    forward = S * np.exp((r - q) * T)
    assert abs(paths[-1].mean() - forward) < 0.15


# --------------------------------------------------------------- the models ---


def test_all_three_models_agree():
    """The headline claim of the app: closed form, lattice and simulation match."""
    for S, K, T, r, sigma, q in CASES:
        for kind in ("call", "put"):
            closed_form = bs_price(S, K, T, r, sigma, q, kind)
            lattice = binomial_price(S, K, T, r, sigma, q, kind, steps=1500)
            simulated = monte_carlo_price(
                S, K, T, r, sigma, q, kind, n_paths=200_000, seed=999
            )
            tolerance = 0.02 * max(1.0, S / 100)
            assert abs(lattice - closed_form) < tolerance
            assert abs(simulated.price - closed_form) < 3 * (
                1.96 * simulated.std_error + tolerance
            )


def test_payoff_is_the_hockey_stick():
    assert payoff(120.0, 100.0, "call") == 20.0
    assert payoff(80.0, 100.0, "call") == 0.0
    assert payoff(80.0, 100.0, "put") == 20.0
    assert payoff(120.0, 100.0, "put") == 0.0


# ------------------------------------------------------- implied volatility ---


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


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS  {}".format(name))
            passed += 1
    print("\n{} tests passed.".format(passed))
