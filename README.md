# Options pricer

A Streamlit dashboard for European option analytics: Black-Scholes pricing and Greeks,
implied volatility solved from market quotes, interactive scenario analysis, and a
volatility smile built from a live option chain.

The question the app is built around: **how does an option's value and risk change as
spot, volatility and time move?**

## What it does

**1. Black-Scholes pricer.** Enter spot, strike, days to expiry, rate, dividend yield and
volatility; get call and put prices side by side with delta, gamma, vega, theta and rho in
the units a desk actually quotes them (vega per vol point, theta per calendar day). The
intermediate quantities (`d1`, `d2`, `N(d2)`, forward price) are shown too, along with the
put-call parity residual as a live correctness check.

**2. Implied volatility.** The interesting direction. Black-Scholes maps volatility to a
price; here the app runs it backwards and solves

```
find sigma such that  BS(S, K, T, r, q, sigma) = C_market
```

for the volatility the market is implying, then reprices the Greeks at that volatility.
The solver reports which method converged and in how many iterations, and warns when vega
is so small that the implied vol carries no real information.

**3. Scenario analysis.** Option price versus underlying (today, halfway to expiry, and at
expiry), delta and gamma versus underlying on a shared axis, a time-decay curve, and a
spot x volatility heatmap. Reading across a row of the heatmap isolates vega; reading down
a column isolates delta and gamma.

**4. Volatility smile.** Pulls a live option chain from Yahoo Finance, solves implied
volatility strike by strike from the observed prices, and plots the result against strike.
Black-Scholes assumes a single volatility for every strike; the chain disagrees, and the
downward-sloping skew that appears is the model's best-known limitation made visible.

## Running it

```bash
git clone https://github.com/mingymingyy/optionspricier.git
cd optionspricier
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. Only the volatility smile tab needs an internet
connection; the first three work entirely offline.

## Project structure

```
optionspricier/
├── app.py              # Streamlit interface and charts
├── black_scholes.py    # pricing + Greeks (vectorised, no Streamlit dependency)
├── implied_vol.py      # numerical IV solver
├── test_pricing.py     # property-based sanity tests
├── requirements.txt
└── README.md
```

`black_scholes.py` and `implied_vol.py` are plain NumPy and import nothing from Streamlit,
so they can be tested, reused or dropped into a notebook on their own.

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

### How the implied-vol solver works

There is no closed form for `sigma`, so the module implements two methods:

* **Newton-Raphson** uses vega, the analytic derivative of price with respect to
  volatility, and converges in three to six iterations from a Brenner-Subrahmanyam seed.
  It breaks down when vega approaches zero - deep in- or out-of-the-money, or very close to
  expiry - because the price stops responding to volatility and the Newton step explodes.
* **Bisection** needs only that the price is strictly increasing in volatility, which it
  always is because vega is positive everywhere. It cannot diverge, but it takes roughly
  fifty iterations to reach the same precision.

`implied_vol()` runs Newton first and falls back to bisection when Newton fails, so the
fast path is used where it is safe and the robust path covers the rest. Before either runs,
the quoted price is checked against its no-arbitrage bounds - a call must trade between its
discounted intrinsic value and the discounted spot - because a price outside that band has
no implied volatility at all and is usually a sign of a stale quote or a wrong input.

## Assumptions and limitations

Worth being explicit about, since they are the first things an interviewer will probe:

* **European exercise.** Listed US equity options are American. For non-dividend-paying
  calls the two coincide, but American puts carry an early-exercise premium that this model
  does not capture, so implied vols from the smile tab are slightly biased for puts.
* **Constant volatility and rates.** The whole point of the smile tab is to show that the
  first of these is false in the market.
* **Calendar time.** `T` uses a 365-day year. Some desks use trading days (252), which
  changes theta and the implied vol level, though not the shape of the skew.
* **Continuous dividend yield** rather than discrete dividends on known dates.
* **Data quality.** Yahoo Finance is delayed and often returns no bid/ask outside market
  hours, in which case the smile tab falls back to the last traded price and labels the row
  as such. A stale print gives a stale implied vol, which is why the tab defaults to
  out-of-the-money strikes with real volume: in-the-money options are thinly traded and
  their premium is mostly intrinsic value, so their implied vols are noise. This is also
  why real vol surfaces are built from out-of-the-money quotes.

## Tests

```bash
python test_pricing.py     # or: pytest -q
```

The tests check properties rather than hard-coded numbers from another library:

* put-call parity holds to 1e-10 across in-, at- and out-of-the-money contracts
* prices stay inside their no-arbitrage bounds and increase monotonically in volatility
* every analytic Greek matches a finite-difference approximation of the same derivative
* pricing at a known volatility and solving back recovers it to 1e-6
* the Newton and bisection solvers agree, and Newton uses fewer iterations
* out-of-bounds quotes are rejected rather than silently returning a wrong volatility
