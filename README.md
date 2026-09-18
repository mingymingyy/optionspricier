# Options pricer

A Streamlit dashboard that prices the same option three ways — Black-Scholes, a
binomial tree and Monte Carlo simulation — then digs into the result with Greeks,
implied volatility, scenario analysis and a live volatility smile.

The question the app is built around: **how does an option's value and risk change as
spot, volatility and time move — and do three different models agree about it?**

## What it does

### 1. Three models, one contract

| Model | Method | Error type | Handles |
|---|---|---|---|
| Black-Scholes | closed form | none — exact | European |
| Binomial tree | Cox-Ross-Rubinstein lattice | discretisation, vanishes as steps rise | European **and American** |
| Monte Carlo | risk-neutral simulation | sampling, falls as `1/sqrt(n)` | European |

All three evaluate the same expectation, `V = e^(-rT) E[payoff(S_T)]`. They are priced
side by side with their differences against the closed form shown explicitly, so
agreement is visible rather than asserted.

Two charts make the error types concrete:

- **Binomial convergence** — the lattice price oscillating around Black-Scholes, with
  odd and even step counts approaching from opposite sides, and the swings damping as
  steps are added.
- **Monte Carlo convergence** — the running estimate inside a 95% band that narrows as
  `1/sqrt(n)`, which is why every extra digit of precision costs a hundred times the paths.

Plus the simulated price paths themselves, with the strike marked and the share of paths
finishing in the money.

### 2. Greeks and scenario analysis

Delta, gamma, vega, theta and rho in the units a desk quotes them (vega per vol point,
theta per calendar day), calls and puts side by side, and the intermediate quantities
(`d1`, `d2`, `N(d2)`, forward price) with the put-call parity residual as a live
correctness check.

Then: option price versus underlying at three points in its life, delta and gamma on a
shared axis, a time-decay curve, and a spot × volatility heatmap. Reading across a row
of the heatmap isolates vega; reading down a column isolates delta and gamma.

### 3. Implied volatility

The interesting direction. Black-Scholes maps volatility to a price; here the app runs it
backwards and solves

```
find sigma such that  BS(S, K, T, r, q, sigma) = C_market
```

then reprices the Greeks at that volatility. The solver reports which method converged
and in how many iterations, and warns when vega is so small that the implied vol carries
no real information.

### 4. Volatility smile

Pulls a live option chain from Yahoo Finance, solves implied volatility strike by strike,
and plots it against strike. Black-Scholes assumes one volatility for every strike; the
chain disagrees, and the downward-sloping skew that appears is the model's best-known
limitation made visible.

### 5. Underlying

Price history, the distribution of daily returns, and **realised** volatility over the
window and the last 30 days — the number to compare against the implied volatility the
options are quoting. One click prices the contract with the realised vol instead.

## Running it

```bash
git clone https://github.com/mingymingyy/optionspricier.git
cd optionspricier
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. Only the volatility smile and underlying tabs
need an internet connection; everything else works offline.

## Project structure

```
optionspricier/
├── app.py                       # Streamlit interface and charts
├── option_pricing/
│   ├── base.py                  # shared option/exercise types and payoff
│   ├── black_scholes.py         # closed form + analytic Greeks
│   ├── binomial.py              # CRR lattice, European and American
│   ├── monte_carlo.py           # simulation, standard error, path generation
│   ├── implied_vol.py           # numerical IV solver
│   └── market_data.py           # Yahoo Finance: spot, history, option chains
├── test_pricing.py              # 29 property-based tests
├── requirements.txt
└── README.md
```

Nothing in `option_pricing/` imports Streamlit, so the models are testable, reusable and
can be dropped into a notebook on their own. The three pricers share a signature,
`(S, K, T, r, sigma, q, option_type)`, so they are drop-in comparisons for one another.

## The maths

With spot `S`, strike `K`, time to expiry `T` in years, continuously compounded rate `r`,
dividend yield `q` and volatility `sigma`:

```
d1 = [ln(S/K) + (r - q + sigma^2 / 2) T] / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)

Call = S e^(-qT) N(d1) - K e^(-rT) N(d2)
Put  = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)
```

The Greeks are the analytic partial derivatives of those expressions. Gamma and vega are
identical for a call and a put on the same contract; delta, theta and rho are not.

**The binomial tree** sets `u = exp(sigma * sqrt(dt))`, `d = 1/u`, and picks the
risk-neutral probability `p = (e^((r-q)dt) - d) / (u - d)` so the expected growth rate
matches the risk-free rate net of dividends. Backward induction from the payoffs at
expiry gives today's value; with American exercise, each node takes the greater of
holding and exercising.

**Monte Carlo** uses the exact GBM solution
`S_T = S exp((r - q - sigma^2/2)T + sigma sqrt(T) Z)`, so a European payoff needs one draw
per path rather than a simulated path — stepping through time would add cost and no
accuracy. Antithetic variates pair every `Z` with `-Z`; the payoffs are negatively
correlated, so averaging the pair cuts variance for free, and the standard error is
computed across pairs rather than across paths because the two halves are not independent.

### How the implied-vol solver works

There is no closed form for `sigma`, so the module implements two methods:

- **Newton-Raphson** uses vega, the analytic derivative of price with respect to
  volatility, and converges in three to six iterations from a Brenner-Subrahmanyam seed.
  It breaks down when vega approaches zero — deep in- or out-of-the-money, or very close
  to expiry — because the price stops responding to volatility and the step explodes.
- **Bisection** needs only that the price is strictly increasing in volatility, which it
  always is because vega is positive everywhere. It cannot diverge, but it takes roughly
  fifty iterations to reach the same precision.

`implied_vol()` runs Newton first and falls back to bisection when Newton fails. Before
either runs, the quoted price is checked against its no-arbitrage bounds — a call must
trade between its discounted intrinsic value and the discounted spot — because a price
outside that band has no implied volatility at all.

## Assumptions and limitations

Worth being explicit about, since they are the first things an interviewer will probe:

- **American exercise** is priced by the binomial tree only. Black-Scholes and the Monte
  Carlo estimate are European, and the app says so rather than quietly pricing the wrong
  contract. An American call on a non-dividend payer is never worth exercising early, so
  its premium is zero; an American put's is not.
- **Constant volatility and rates.** The whole point of the smile tab is to show that the
  first of these is false in the market.
- **Calendar time.** `T` uses a 365-day year, while realised volatility is annualised over
  252 trading days — returns are only generated on days the market is open.
- **Continuous dividend yield** rather than discrete dividends on known dates.
- **Log-normal returns.** The returns histogram on the Underlying tab shows real returns
  have fatter tails than a normal, which is part of why the market charges more for far
  out-of-the-money options than the model says.
- **Data quality.** Yahoo Finance is delayed and often returns no bid/ask outside market
  hours, in which case the chain falls back to the last traded price and labels the row as
  such. A stale print gives a stale implied vol, which is why the smile tab defaults to
  out-of-the-money strikes with real volume: in-the-money options are thinly traded and
  their premium is mostly intrinsic value, so their implied vols are noise. This is also
  why real vol surfaces are built from out-of-the-money quotes.

## Tests

```bash
python test_pricing.py     # or: pytest -q
```

29 tests that check properties rather than hard-coded numbers from another library:

- put-call parity holds to 1e-10 across in-, at- and out-of-the-money contracts
- prices stay inside their no-arbitrage bounds and increase monotonically in volatility
- every analytic Greek matches a finite-difference approximation of the same derivative
- the lattice and the simulation both reproduce the closed form, and the simulation's
  95% interval contains it
- lattice error shrinks with steps; Monte Carlo standard error falls as `1/sqrt(n)`
- antithetic variates measurably reduce variance
- American puts are worth at least European puts, and American calls on a non-dividend
  payer are worth exactly the same
- simulated paths start at spot, stay positive, and their terminal mean matches the forward
- pricing at a known volatility and solving back recovers it to 1e-6
- out-of-bounds quotes are rejected rather than silently returning a wrong volatility

## Credits

The three-model structure was inspired by
[just-krivi/option-pricing-models](https://github.com/just-krivi/option-pricing-models).
All code here is an independent implementation; this version adds dividend yields,
American exercise, Greeks, implied volatility, variance reduction with reported standard
errors, convergence analysis, the volatility smile, and a test suite.
