"""Cox-Ross-Rubinstein binomial tree.

The tree replaces the continuous price process with a lattice: over each small
step `dt` the underlying either multiplies by `u` or by `d = 1/u`, with the
risk-neutral probability `p` chosen so the expected return equals the risk-free
rate net of dividends. Working backwards from the payoffs at expiry gives
today's value.

As the number of steps grows the lattice converges to Black-Scholes, which the
`convergence()` helper demonstrates. The reason to use it anyway is that it
prices **American** options: at every node you can compare holding the option
with exercising it immediately, which no closed form does.
"""

from __future__ import annotations

import numpy as np

from .base import AMERICAN, CALL, EUROPEAN, normalise_exercise, normalise_option_type, payoff

__all__ = ["binomial_price", "lattice_parameters", "convergence"]


def lattice_parameters(T: float, r: float, sigma: float, q: float, steps: int) -> dict:
    """The CRR step size, up/down factors and risk-neutral probability.

    `u = exp(sigma * sqrt(dt))` matches the variance of the log return over one
    step, and `p` is set so the expected growth rate is `r - q`.
    """
    dt = T / steps
    u = float(np.exp(sigma * np.sqrt(dt)))
    d = 1.0 / u
    growth = float(np.exp((r - q) * dt))
    p = (growth - d) / (u - d)
    return {"dt": dt, "u": u, "d": d, "p": p, "discount": float(np.exp(-r * dt))}


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = CALL,
    steps: int = 500,
    exercise: str = EUROPEAN,
) -> float:
    """Price a call or put on a CRR lattice with `steps` time steps.

    With `exercise="american"` the value at each node is the greater of holding
    and exercising, which is what gives an American put its early-exercise
    premium over the European one.
    """
    kind = normalise_option_type(option_type)
    style = normalise_exercise(exercise)
    steps = max(int(steps), 1)

    if T <= 0 or sigma <= 0:
        return float(payoff(S, K, kind))

    params = lattice_parameters(T, r, sigma, q, steps)
    u, d, p, discount = params["u"], params["d"], params["p"], params["discount"]

    if not 0.0 < p < 1.0:
        # Too few steps for this volatility: the lattice cannot span the drift,
        # which would produce negative "probabilities" and an arbitrage.
        raise ValueError(
            "Risk-neutral probability {:.3f} is outside (0, 1); use more steps.".format(p)
        )

    # Terminal nodes: j up-moves and (steps - j) down-moves.
    j = np.arange(steps + 1)
    spot_at_expiry = S * (u**j) * (d ** (steps - j))
    values = payoff(spot_at_expiry, K, kind)

    # Roll the lattice back one step at a time. After the i-th iteration
    # `values` holds the option value at every node of step i.
    for i in range(steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])

        if style == AMERICAN:
            j = np.arange(i + 1)
            spot_now = S * (u**j) * (d ** (i - j))
            values = np.maximum(values, payoff(spot_now, K, kind))

    return float(values[0])


def convergence(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = CALL,
    max_steps: int = 200,
    exercise: str = EUROPEAN,
):
    """Price the same contract at every step count from 1 to `max_steps`.

    The resulting series oscillates around the Black-Scholes value, with the
    odd and even step counts approaching from opposite sides - a well known
    artefact of whether a lattice node lands exactly on the strike.
    """
    steps = np.arange(1, int(max_steps) + 1)
    prices = np.array(
        [
            binomial_price(S, K, T, r, sigma, q, option_type, int(n), exercise)
            for n in steps
        ]
    )
    return steps, prices
