"""Monte Carlo pricing under geometric Brownian motion.

Risk-neutral valuation says an option is worth the discounted expected payoff:

    V = e^(-rT) * E[ payoff(S_T) ]

Monte Carlo estimates that expectation by drawing many terminal prices and
averaging. Under GBM the terminal price has an exact solution,

    S_T = S * exp( (r - q - sigma^2 / 2) T + sigma * sqrt(T) * Z ),  Z ~ N(0, 1)

so a European payoff needs **one** draw per path rather than a simulated path.
Stepping through time adds cost and no accuracy here; paths are simulated only
for the chart (and would be genuinely required for a path-dependent payoff such
as an Asian or barrier option).

Unlike a closed form, a Monte Carlo price is an estimate with a confidence
interval, so every result carries its standard error. Halving that error costs
four times the paths - the 1/sqrt(n) convergence that makes variance reduction
worth doing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import CALL, normalise_option_type, payoff

__all__ = ["MonteCarloResult", "monte_carlo_price", "simulate_paths", "convergence"]

# 95% of a normal distribution lies within this many standard errors.
Z_95 = 1.959963984540054


@dataclass
class MonteCarloResult:
    """A Monte Carlo price is an estimate, so it comes with its error bars."""

    price: float
    std_error: float
    n_paths: int
    antithetic: bool

    @property
    def ci_low(self) -> float:
        return self.price - Z_95 * self.std_error

    @property
    def ci_high(self) -> float:
        return self.price + Z_95 * self.std_error

    def __str__(self) -> str:
        return "{:.4f} +/- {:.4f} (95% CI, {:,} paths)".format(
            self.price, Z_95 * self.std_error, self.n_paths
        )


def _terminal_prices(S, T, r, sigma, q, normals):
    """Exact GBM solution for the price at expiry given standard normal draws."""
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * normals
    return S * np.exp(drift + diffusion)


def monte_carlo_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = CALL,
    n_paths: int = 100_000,
    antithetic: bool = True,
    seed: int | None = None,
) -> MonteCarloResult:
    """Estimate a European option price by simulation.

    With `antithetic=True` every draw Z is paired with -Z and the pair is
    averaged before taking the mean. The two payoffs are negatively correlated,
    so the variance of their average is lower than that of two independent
    draws, and the estimator is still unbiased. The standard error is computed
    across pairs rather than across individual paths, because the two halves of
    a pair are not independent observations.
    """
    kind = normalise_option_type(option_type)
    n_paths = max(int(n_paths), 2)
    rng = np.random.default_rng(seed)

    if T <= 0 or sigma <= 0:
        value = float(payoff(S * np.exp((r - q) * max(T, 0.0)), K, kind)) * float(
            np.exp(-r * max(T, 0.0))
        )
        return MonteCarloResult(value, 0.0, n_paths, antithetic)

    discount = float(np.exp(-r * T))

    if antithetic:
        half = n_paths // 2
        normals = rng.standard_normal(half)
        up = payoff(_terminal_prices(S, T, r, sigma, q, normals), K, kind)
        down = payoff(_terminal_prices(S, T, r, sigma, q, -normals), K, kind)
        samples = discount * 0.5 * (up + down)
        n_paths = half * 2
    else:
        normals = rng.standard_normal(n_paths)
        samples = discount * payoff(_terminal_prices(S, T, r, sigma, q, normals), K, kind)

    price = float(samples.mean())
    # ddof=1: the sample standard deviation, since the mean is estimated too.
    std_error = float(samples.std(ddof=1) / np.sqrt(samples.size))
    return MonteCarloResult(price, std_error, n_paths, antithetic)


def simulate_paths(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n_paths: int = 200,
    n_steps: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate full GBM price paths, for visualisation.

    Returns an array of shape (n_steps + 1, n_paths); row 0 is the spot price.
    Each step uses the same exact log-normal solution applied over `dt`, so the
    paths carry no discretisation bias either.
    """
    rng = np.random.default_rng(seed)
    n_steps = max(int(n_steps), 1)
    n_paths = max(int(n_paths), 1)

    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    vol_step = sigma * np.sqrt(dt)

    shocks = drift + vol_step * rng.standard_normal((n_steps, n_paths))
    log_paths = np.vstack([np.zeros((1, n_paths)), np.cumsum(shocks, axis=0)])
    return S * np.exp(log_paths)


def convergence(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = CALL,
    n_paths: int = 50_000,
    seed: int | None = None,
    points: int = 60,
):
    """Running estimate and standard error as paths accumulate.

    One sample is drawn and then read cumulatively, so this traces what a single
    simulation would have reported had it stopped early - the estimate wandering
    inside a 1/sqrt(n) funnel that narrows as paths are added.
    """
    kind = normalise_option_type(option_type)
    rng = np.random.default_rng(seed)
    n_paths = max(int(n_paths), 10)

    normals = rng.standard_normal(n_paths)
    samples = float(np.exp(-r * T)) * payoff(
        _terminal_prices(S, T, r, sigma, q, normals), K, kind
    )

    counts = np.unique(
        np.geomspace(10, n_paths, num=int(points)).astype(int)
    )  # log spacing: the interesting action is at small path counts

    cumulative_sum = np.cumsum(samples)
    cumulative_sq = np.cumsum(samples**2)

    means = cumulative_sum[counts - 1] / counts
    variances = np.maximum(cumulative_sq[counts - 1] / counts - means**2, 0.0)
    std_errors = np.sqrt(variances / counts)

    return counts, means, std_errors
