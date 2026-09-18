"""Options analytics dashboard.

Three pricing models on one contract, plus Greeks, implied volatility and
scenario analysis.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

from option_pricing import binomial, market_data, monte_carlo
from option_pricing.black_scholes import bs_price, d1_d2, scaled_greeks, vega
from option_pricing.implied_vol import ImpliedVolError, implied_vol, price_bounds

st.set_page_config(
    page_title="Options pricer",
    page_icon=":material/candlestick_chart:",
    layout="wide",
)

DAYS_PER_YEAR = 365.0

# A Greek is only worth reading if the price actually responds to the input.
# Below roughly 1 cent of P&L per vol point, an implied vol is noise.
VEGA_RELIABILITY_FLOOR = 0.01

# Theme-matched accents for the few marks that are not part of a colour scale.
NEUTRAL = "#94A3B8"
ACCENT_RED = "#F87171"


# Widget values cannot be written after their widget exists, so buttons park a
# new value under a pending key and it is applied at the top of the next run.
for pending_key, target_key in (("_pending_spot", "spot"), ("_pending_vol", "vol_pct")):
    if pending_key in st.session_state:
        st.session_state[target_key] = st.session_state.pop(pending_key)


# ------------------------------------------------------------ cached data ---


@st.cache_data(ttl=900, show_spinner=False)
def cached_spot(symbol: str) -> float:
    return market_data.get_spot(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def cached_history(symbol: str, period: str) -> pd.DataFrame:
    return market_data.get_history(symbol, period)


@st.cache_data(ttl=900, show_spinner=False)
def cached_expiries(symbol: str) -> list[str]:
    return market_data.get_expiries(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def cached_chain(symbol: str, expiry: str) -> pd.DataFrame:
    return market_data.get_option_chain(symbol, expiry)


@st.cache_data(show_spinner=False)
def cached_binomial_convergence(S, K, T, r, sigma, q, option_type, max_steps, exercise):
    return binomial.convergence(S, K, T, r, sigma, q, option_type, max_steps, exercise)


@st.cache_data(show_spinner=False)
def cached_mc_convergence(S, K, T, r, sigma, q, option_type, n_paths, seed):
    return monte_carlo.convergence(S, K, T, r, sigma, q, option_type, n_paths, seed)


# ---------------------------------------------------------------- sidebar ---

with st.sidebar:
    st.subheader("Contract")

    option_label = st.segmented_control(
        "Option type", options=["Call", "Put"], default="Call", key="option_label"
    )
    option_type = (option_label or "Call").lower()

    exercise_label = st.segmented_control(
        "Exercise style",
        options=["European", "American"],
        default="European",
        key="exercise_label",
        help=(
            "European options can only be exercised at expiry. American options can be "
            "exercised any time, which only the binomial tree here can price."
        ),
    )
    exercise = (exercise_label or "European").lower()

    st.session_state.setdefault("spot", 250.0)
    spot = st.number_input("Spot price (S)", min_value=0.01, step=1.0, key="spot")
    strike = st.number_input("Strike price (K)", min_value=0.01, value=260.0, step=1.0)

    expiry_date = st.date_input(
        "Expiry date",
        value=date.today() + timedelta(days=30),
        min_value=date.today() + timedelta(days=1),
        max_value=date.today() + timedelta(days=3650),
    )
    days = max((expiry_date - date.today()).days, 1)

    st.session_state.setdefault("vol_pct", 28.4)
    vol_pct = st.number_input(
        "Volatility (annualised, %)",
        min_value=0.1,
        max_value=500.0,
        step=0.5,
        key="vol_pct",
    )
    st.caption("{} days to expiry.".format(days))

    st.subheader("Market")
    rate_pct = st.number_input(
        "Risk-free rate (%)", min_value=-5.0, max_value=25.0, value=4.5, step=0.1
    )
    div_pct = st.number_input(
        "Dividend yield (%)", min_value=0.0, max_value=25.0, value=0.0, step=0.1
    )

    st.subheader("Load from market")
    ticker_symbol = (
        st.text_input("Ticker", value="AAPL", max_chars=12, key="ticker").strip().upper()
    )
    if st.button("Fetch spot price", icon=":material/download:", width="stretch"):
        try:
            st.session_state["_pending_spot"] = round(cached_spot(ticker_symbol), 2)
            st.rerun()
        except Exception as exc:  # noqa: BLE001 - any network/data failure
            st.error("Could not fetch {}: {}".format(ticker_symbol, exc))

    st.caption(
        "Time is measured in calendar days over a 365-day year. Rates are "
        "continuously compounded."
    )

T = days / DAYS_PER_YEAR
r = rate_pct / 100.0
q = div_pct / 100.0
sigma = vol_pct / 100.0
is_american = exercise == "american"


# ----------------------------------------------------------------- helpers ---


def money(x: float) -> str:
    return "${:,.2f}".format(x)


def greeks_frame(S, K, T, r, sigma, q) -> pd.DataFrame:
    """Call and put Greeks side by side, in desk units."""
    call = scaled_greeks(S, K, T, r, sigma, q, "call")
    put = scaled_greeks(S, K, T, r, sigma, q, "put")
    rows = [
        ("Price", "Fair value of the contract", call["price"], put["price"]),
        ("Delta", "Change in price per $1 move in spot", call["delta"], put["delta"]),
        ("Gamma", "Change in delta per $1 move in spot", call["gamma"], put["gamma"]),
        ("Vega", "Change in price per 1 vol point", call["vega"], put["vega"]),
        ("Theta", "Change in price per calendar day", call["theta"], put["theta"]),
        ("Rho", "Change in price per 1% move in rates", call["rho"], put["rho"]),
    ]
    return pd.DataFrame(rows, columns=["Greek", "Meaning", "Call", "Put"])


def spot_grid(S: float, width_pct: float, points: int = 121) -> np.ndarray:
    lo = max(S * (1 - width_pct / 100.0), 0.01)
    hi = S * (1 + width_pct / 100.0)
    return np.linspace(lo, hi, points)


def reference_rules(S: float, K: float) -> alt.Chart:
    """Vertical markers for the current spot and the strike."""
    marks = pd.DataFrame({"x": [S, K], "label": ["Spot", "Strike"]})
    return (
        alt.Chart(marks)
        .mark_rule(color=NEUTRAL, opacity=0.8)
        .encode(
            x=alt.X("x:Q"),
            strokeDash=alt.StrokeDash("label:N", title=None),
            tooltip=["label:N", alt.Tooltip("x:Q", format="$.2f", title="Level")],
        )
    )


def horizontal_reference(value: float, label: str) -> alt.Chart:
    """A dashed horizontal line, for marking a benchmark price on a chart."""
    return (
        alt.Chart(pd.DataFrame({"y": [value], "label": [label]}))
        .mark_rule(color=NEUTRAL, strokeDash=[5, 4])
        .encode(y="y:Q", tooltip=["label:N", alt.Tooltip("y:Q", format="$.4f")])
    )


# ------------------------------------------------------------------ header ---

st.title(":material/candlestick_chart: Options pricer")
st.caption(
    "Black-Scholes, binomial tree and Monte Carlo on one contract, with Greeks, "
    "implied volatility and scenario analysis."
)

tab_pricing, tab_greeks, tab_iv, tab_smile, tab_underlying = st.tabs(
    [
        "Pricing models",
        "Greeks & scenarios",
        "Implied volatility",
        "Volatility smile",
        "Underlying",
    ]
)


# ================================================== 1. Three pricing models ==

with tab_pricing:
    st.subheader("Three models, one contract")
    st.caption(
        "A closed form, a lattice and a simulation should all agree on the same "
        "European option. Where they disagree is as informative as where they do not."
    )

    settings = st.container(horizontal=True, vertical_alignment="bottom")
    with settings:
        steps = st.select_slider(
            "Binomial steps",
            options=[10, 25, 50, 100, 250, 500, 1000, 2000],
            value=500,
            help="Time steps in the lattice. More steps converge towards Black-Scholes.",
        )
        n_paths = st.select_slider(
            "Monte Carlo paths",
            options=[1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000],
            value=100_000,
            help="More paths shrink the standard error, but only as 1/sqrt(n).",
        )
        antithetic = st.toggle(
            "Antithetic variates",
            value=True,
            help=(
                "Pair every random draw Z with -Z. The payoffs are negatively "
                "correlated, so averaging the pair cuts the variance for free."
            ),
        )
        mc_seed = st.number_input("Random seed", min_value=0, value=7, step=1)

    bs_call = bs_price(spot, strike, T, r, sigma, q, "call")
    bs_put = bs_price(spot, strike, T, r, sigma, q, "put")
    bs_selected = bs_call if option_type == "call" else bs_put

    try:
        tree_call = binomial.binomial_price(
            spot, strike, T, r, sigma, q, "call", steps, exercise
        )
        tree_put = binomial.binomial_price(
            spot, strike, T, r, sigma, q, "put", steps, exercise
        )
        tree_error = None
    except ValueError as exc:
        tree_call = tree_put = float("nan")
        tree_error = str(exc)

    mc_call = monte_carlo.monte_carlo_price(
        spot, strike, T, r, sigma, q, "call", n_paths, antithetic, int(mc_seed)
    )
    mc_put = monte_carlo.monte_carlo_price(
        spot, strike, T, r, sigma, q, "put", n_paths, antithetic, int(mc_seed)
    )

    tree_selected = tree_call if option_type == "call" else tree_put
    mc_selected = mc_call if option_type == "call" else mc_put

    col_bs, col_tree, col_mc = st.columns(3)

    with col_bs:
        with st.container(border=True):
            st.markdown("**Black-Scholes**")
            st.caption("Closed form - exact, instant, European only")
            st.metric("Call", money(bs_call))
            st.metric("Put", money(bs_put))
            st.caption("Reference value for the other two.")

    with col_tree:
        with st.container(border=True):
            st.markdown("**Binomial tree**")
            st.caption("{:,} steps - Cox-Ross-Rubinstein lattice".format(steps))
            if tree_error:
                st.error(tree_error, icon=":material/error:")
            else:
                st.metric(
                    "Call",
                    money(tree_call),
                    "{:+.4f} vs BS".format(tree_call - bs_call),
                    delta_color="off",
                )
                st.metric(
                    "Put",
                    money(tree_put),
                    "{:+.4f} vs BS".format(tree_put - bs_put),
                    delta_color="off",
                )
                if is_american:
                    st.caption("American exercise - the premium over BS is real, not error.")
                else:
                    st.caption("Difference is pure discretisation error.")

    with col_mc:
        with st.container(border=True):
            st.markdown("**Monte Carlo**")
            st.caption(
                "{:,} paths{} - European only".format(
                    mc_call.n_paths, ", antithetic" if antithetic else ""
                )
            )
            st.metric(
                "Call",
                money(mc_call.price),
                "+/- {:.4f} at 95%".format(1.96 * mc_call.std_error),
                delta_color="off",
            )
            st.metric(
                "Put",
                money(mc_put.price),
                "+/- {:.4f} at 95%".format(1.96 * mc_put.std_error),
                delta_color="off",
            )
            inside = mc_call.ci_low <= bs_call <= mc_call.ci_high
            st.caption(
                ("Black-Scholes sits inside the 95% interval." if inside
                 else "Black-Scholes falls outside the interval - unlucky draw or a bug.")
            )

    if is_american:
        euro_tree = binomial.binomial_price(
            spot, strike, T, r, sigma, q, option_type, steps, "european"
        )
        premium = tree_selected - euro_tree
        st.info(
            "Early-exercise premium on this {}: **{}**. Black-Scholes and the Monte Carlo "
            "estimate above price the European contract, so only the binomial column is "
            "pricing what you selected. An American call on a non-dividend payer is never "
            "worth exercising early, so its premium is zero - the put's is not.".format(
                option_type, money(premium)
            ),
            icon=":material/info:",
        )

    # --- convergence -------------------------------------------------------
    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("**Binomial convergence**")
            conv_steps, conv_prices = cached_binomial_convergence(
                spot, strike, T, r, sigma, q, option_type, 150, exercise
            )
            conv = pd.DataFrame({"Steps": conv_steps, "Price": conv_prices})
            line = (
                alt.Chart(conv)
                .mark_line(strokeWidth=1.5)
                .encode(
                    x=alt.X("Steps:Q", title="Time steps in the lattice"),
                    y=alt.Y("Price:Q", scale=alt.Scale(zero=False), title="Option price ($)"),
                    tooltip=[
                        alt.Tooltip("Steps:Q"),
                        alt.Tooltip("Price:Q", format="$.4f"),
                    ],
                )
            )
            st.altair_chart(
                (line + horizontal_reference(bs_selected, "Black-Scholes")).properties(
                    height=280
                )
            )
            st.caption(
                "The lattice price oscillates around Black-Scholes (dashed) and the swings "
                "decay as steps are added. Odd and even step counts approach from opposite "
                "sides, depending on whether a node lands on the strike."
            )

    with right:
        with st.container(border=True):
            st.markdown("**Monte Carlo convergence**")
            counts, means, errors = cached_mc_convergence(
                spot, strike, T, r, sigma, q, option_type, 50_000, int(mc_seed)
            )
            mc_conv = pd.DataFrame(
                {
                    "Paths": counts,
                    "Estimate": means,
                    "Low": means - 1.96 * errors,
                    "High": means + 1.96 * errors,
                }
            )
            band = (
                alt.Chart(mc_conv)
                .mark_area(opacity=0.22)
                .encode(
                    x=alt.X("Paths:Q", scale=alt.Scale(type="log"), title="Paths simulated"),
                    y=alt.Y("Low:Q", scale=alt.Scale(zero=False), title="Option price ($)"),
                    y2="High:Q",
                )
            )
            estimate = (
                alt.Chart(mc_conv)
                .mark_line(strokeWidth=1.5)
                .encode(
                    x=alt.X("Paths:Q", scale=alt.Scale(type="log")),
                    y=alt.Y("Estimate:Q", scale=alt.Scale(zero=False)),
                    tooltip=[
                        alt.Tooltip("Paths:Q", format=","),
                        alt.Tooltip("Estimate:Q", format="$.4f"),
                    ],
                )
            )
            st.altair_chart(
                (band + estimate + horizontal_reference(bs_selected, "Black-Scholes")).properties(
                    height=280
                )
            )
            st.caption(
                "The shaded 95% interval narrows as 1/sqrt(n): every extra digit of "
                "precision costs a hundred times the paths. This is why simulation is a "
                "last resort when a closed form exists."
            )

    # --- simulated paths ---------------------------------------------------
    with st.container(border=True):
        st.markdown("**Simulated price paths**")
        shown_paths = st.slider("Paths to draw", 10, 400, 120, step=10)

        paths = monte_carlo.simulate_paths(
            spot, T, r, sigma, q, n_paths=shown_paths, n_steps=min(days, 252),
            seed=int(mc_seed),
        )
        step_days = np.linspace(0, days, paths.shape[0])
        path_frame = pd.DataFrame(
            {
                "Day": np.tile(step_days, paths.shape[1]),
                "Price": paths.T.ravel(),
                "Path": np.repeat(np.arange(paths.shape[1]), paths.shape[0]),
            }
        )
        finished_itm = float(
            np.mean(
                paths[-1] > strike if option_type == "call" else paths[-1] < strike
            )
        )

        path_chart = (
            alt.Chart(path_frame)
            .mark_line(strokeWidth=0.7, opacity=0.5)
            .encode(
                x=alt.X("Day:Q", title="Days from today"),
                y=alt.Y("Price:Q", scale=alt.Scale(zero=False), title="Underlying price ($)"),
                detail="Path:N",
            )
        )
        strike_rule = (
            alt.Chart(pd.DataFrame({"y": [strike], "label": ["Strike"]}))
            .mark_rule(color=ACCENT_RED, strokeDash=[6, 4], strokeWidth=1.5)
            .encode(y="y:Q", tooltip=["label:N", alt.Tooltip("y:Q", format="$.2f")])
        )
        st.altair_chart((path_chart + strike_rule).properties(height=340))
        st.caption(
            "{:.0%} of these {} paths finished in the money. Each path is one draw from "
            "the risk-neutral distribution; the option price is the discounted average of "
            "their payoffs, not of their prices.".format(finished_itm, shown_paths)
        )

    with st.expander("How the three models relate"):
        st.markdown(
            "All three price the same expectation, `V = e^(-rT) E[payoff(S_T)]`, under "
            "the risk-neutral measure. They differ only in how they evaluate it:"
        )
        st.markdown(
            "- **Black-Scholes** solves the integral analytically, because under GBM "
            "`S_T` is log-normal and the integral of a log-normal against a hockey-stick "
            "payoff has a closed form.\n"
            "- **The binomial tree** replaces the continuous process with a lattice and "
            "evaluates the expectation by backward induction. Its error is discretisation "
            "error, and it vanishes as steps increase.\n"
            "- **Monte Carlo** samples the distribution directly. Its error is sampling "
            "error, and it vanishes as `1/sqrt(n)` - never exactly, only in probability."
        )
        st.markdown(
            "The tree earns its keep on American exercise, which has no closed form. "
            "Monte Carlo earns its keep on path-dependent payoffs - Asian, barrier, "
            "lookback - where the terminal price alone is not enough."
        )


# ============================================== 2. Greeks and scenario work ==

with tab_greeks:
    st.subheader("Greeks")

    g = scaled_greeks(spot, strike, T, r, sigma, q, option_type)

    with st.container(horizontal=True):
        st.metric("{} price".format(option_label), money(g["price"]), border=True)
        st.metric("Delta", "{:.4f}".format(g["delta"]), border=True)
        st.metric("Gamma", "{:.4f}".format(g["gamma"]), border=True)
        st.metric("Vega", "{:.4f}".format(g["vega"]), border=True, help="Per 1 vol point")
        st.metric("Theta", "{:.4f}".format(g["theta"]), border=True, help="Per calendar day")
        st.metric("Rho", "{:.4f}".format(g["rho"]), border=True, help="Per 1% move in rates")

    left, right = st.columns([3, 2])

    with left:
        with st.container(border=True):
            st.markdown("**Call vs put**")
            st.dataframe(
                greeks_frame(spot, strike, T, r, sigma, q),
                hide_index=True,
                column_config={
                    "Call": st.column_config.NumberColumn(format="%.4f"),
                    "Put": st.column_config.NumberColumn(format="%.4f"),
                },
            )
            st.caption("Gamma and vega are identical for a call and a put on one contract.")

    with right:
        with st.container(border=True):
            st.markdown("**Intermediate quantities**")
            d1, d2 = d1_d2(spot, strike, T, r, sigma, q)
            st.dataframe(
                pd.DataFrame(
                    {
                        "Quantity": ["d1", "d2", "N(d2)", "Forward price", "Moneyness (S/K)"],
                        "Value": [
                            float(d1),
                            float(d2),
                            float(norm.cdf(d2)),
                            float(spot * np.exp((r - q) * T)),
                            float(spot / strike),
                        ],
                    }
                ),
                hide_index=True,
                column_config={"Value": st.column_config.NumberColumn(format="%.4f")},
            )
            st.caption("N(d2) is the risk-neutral probability a call finishes in the money.")

    st.subheader("Scenario analysis")

    controls = st.container(horizontal=True, vertical_alignment="bottom")
    with controls:
        spot_range = st.slider("Spot range (+/- %)", 5, 60, 20, step=5)
        vol_range = st.slider("Vol range (+/- pts)", 2, 30, 10, step=1)
        grid_steps = st.select_slider("Heatmap size", options=[3, 5, 7, 9], value=5)

    grid = spot_grid(spot, spot_range)

    price_now = bs_price(grid, strike, T, r, sigma, q, option_type)
    intrinsic = (
        np.maximum(grid - strike, 0.0)
        if option_type == "call"
        else np.maximum(strike - grid, 0.0)
    )
    half_life = bs_price(grid, strike, T / 2.0, r, sigma, q, option_type)

    profile = pd.DataFrame(
        {
            "Underlying price": np.tile(grid, 3),
            "Option price": np.concatenate([price_now, half_life, intrinsic]),
            "Series": np.repeat(
                [
                    "Today ({}d)".format(days),
                    "Halfway ({}d)".format(max(days // 2, 1)),
                    "At expiry",
                ],
                len(grid),
            ),
        }
    )

    chart_left, chart_right = st.columns(2)

    with chart_left:
        with st.container(border=True):
            st.markdown("**Option price vs underlying**")
            profile_chart = (
                alt.Chart(profile)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X(
                        "Underlying price:Q",
                        scale=alt.Scale(zero=False),
                        title="Underlying price ($)",
                    ),
                    y=alt.Y("Option price:Q", title="Option price ($)"),
                    color=alt.Color("Series:N", title=None, legend=alt.Legend(orient="top")),
                    tooltip=[
                        alt.Tooltip("Underlying price:Q", format="$.2f"),
                        alt.Tooltip("Option price:Q", format="$.2f"),
                        "Series:N",
                    ],
                )
            )
            st.altair_chart(
                (profile_chart + reference_rules(spot, strike)).properties(height=320)
            )
            st.caption(
                "The gap between today's curve and the intrinsic line is time value - the "
                "part of the premium that decays away."
            )

    with chart_right:
        with st.container(border=True):
            st.markdown("**Delta and gamma vs underlying**")
            grid_greeks = scaled_greeks(grid, strike, T, r, sigma, q, option_type)
            risk = pd.DataFrame(
                {
                    "Underlying price": grid,
                    "Delta": grid_greeks["delta"],
                    "Gamma": grid_greeks["gamma"],
                }
            )
            delta_line = (
                alt.Chart(risk)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X(
                        "Underlying price:Q",
                        scale=alt.Scale(zero=False),
                        title="Underlying price ($)",
                    ),
                    y=alt.Y("Delta:Q", title="Delta"),
                    color=alt.datum("Delta"),
                    tooltip=[
                        alt.Tooltip("Underlying price:Q", format="$.2f"),
                        alt.Tooltip("Delta:Q", format=".3f"),
                    ],
                )
            )
            gamma_line = (
                alt.Chart(risk)
                .mark_line(strokeWidth=2, strokeDash=[5, 3])
                .encode(
                    x=alt.X("Underlying price:Q", scale=alt.Scale(zero=False)),
                    y=alt.Y("Gamma:Q", title="Gamma"),
                    color=alt.datum("Gamma"),
                    tooltip=[
                        alt.Tooltip("Underlying price:Q", format="$.2f"),
                        alt.Tooltip("Gamma:Q", format=".4f"),
                    ],
                )
            )
            st.altair_chart(
                alt.layer(delta_line, gamma_line)
                .resolve_scale(y="independent")
                .properties(height=320)
                .configure_legend(orient="top", title=None)
            )
            st.caption(
                "Delta on the left axis, gamma (dashed) on the right. Gamma peaks near the "
                "strike, which is where delta is changing fastest."
            )

    with st.container(border=True):
        st.markdown("**Spot x volatility heatmap**")

        spot_axis = np.linspace(
            spot * (1 - spot_range / 100.0), spot * (1 + spot_range / 100.0), grid_steps
        )
        vol_axis = np.linspace(max(vol_pct - vol_range, 1.0), vol_pct + vol_range, grid_steps)

        mesh_spot, mesh_vol = np.meshgrid(spot_axis, vol_axis, indexing="ij")
        mesh_price = bs_price(mesh_spot, strike, T, r, mesh_vol / 100.0, q, option_type)

        heat = pd.DataFrame(
            {
                "Spot": mesh_spot.ravel(),
                "Vol": mesh_vol.ravel(),
                "Price": mesh_price.ravel(),
            }
        )
        heat["Spot label"] = heat["Spot"].map("{:,.0f}".format)
        heat["Vol label"] = heat["Vol"].map("{:.0f}%".format)

        spot_order = ["{:,.0f}".format(v) for v in sorted(spot_axis, reverse=True)]
        vol_order = ["{:.0f}%".format(v) for v in vol_axis]

        base = alt.Chart(heat).encode(
            x=alt.X(
                "Vol label:O",
                sort=vol_order,
                title="Implied volatility",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("Spot label:O", sort=spot_order, title="Spot price"),
        )
        cells = base.mark_rect().encode(
            color=alt.Color(
                "Price:Q",
                scale=alt.Scale(scheme="darkblue"),
                legend=alt.Legend(title="Price ($)", format="$.2f"),
            ),
            tooltip=[
                alt.Tooltip("Spot:Q", format="$.2f"),
                alt.Tooltip("Vol:Q", format=".1f", title="Vol (%)"),
                alt.Tooltip("Price:Q", format="$.2f"),
            ],
        )
        # The colour scheme runs dark (cheap) to light (expensive), so the label
        # has to flip the other way to stay readable on both ends.
        midpoint = float(heat["Price"].max() + heat["Price"].min()) / 2.0
        labels = base.mark_text(fontSize=13, fontWeight="bold").encode(
            text=alt.Text("Price:Q", format=".2f"),
            color=alt.condition(
                alt.datum.Price > midpoint, alt.value("#0F172A"), alt.value("#E2E8F0")
            ),
        )
        st.altair_chart((cells + labels).properties(height=60 + 46 * grid_steps))
        st.caption(
            "Each cell reprices the same {} at a different spot and volatility, holding "
            "{} days to expiry fixed. Reading across a row isolates vega; reading down a "
            "column isolates delta and gamma.".format(option_type, days)
        )

    with st.container(border=True):
        st.markdown("**Time decay**")
        day_axis = np.arange(days, 0, -1)
        decay = pd.DataFrame(
            {
                "Days to expiry": day_axis,
                "Option price": bs_price(
                    spot, strike, day_axis / DAYS_PER_YEAR, r, sigma, q, option_type
                ),
            }
        )
        decay_chart = (
            alt.Chart(decay)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X(
                    "Days to expiry:Q",
                    scale=alt.Scale(reverse=True),
                    title="Days to expiry",
                ),
                y=alt.Y("Option price:Q", title="Option price ($)"),
                tooltip=[
                    alt.Tooltip("Days to expiry:Q"),
                    alt.Tooltip("Option price:Q", format="$.2f"),
                ],
            )
        )
        st.altair_chart(decay_chart.properties(height=260))
        st.caption(
            "Holding spot and volatility fixed, time value bleeds away - slowly at first, "
            "then sharply into the last few weeks."
        )

    with st.expander("The Black-Scholes formula"):
        st.latex(
            r"d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}},"
            r"\qquad d_2 = d_1 - \sigma\sqrt{T}"
        )
        st.latex(r"C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)")
        st.latex(r"P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)")
        parity = (
            bs_price(spot, strike, T, r, sigma, q, "call")
            - bs_price(spot, strike, T, r, sigma, q, "put")
            - (spot * np.exp(-q * T) - strike * np.exp(-r * T))
        )
        st.markdown(
            r"Put-call parity ties them together: $C - P = S e^{-qT} - K e^{-rT}$."
        )
        st.caption("Parity residual for these inputs: {:.2e}".format(parity))


# =============================================== 3. Implied volatility solver ==

with tab_iv:
    st.subheader("Implied volatility")
    st.caption(
        "Black-Scholes turns a volatility into a price. Here we run it backwards: given "
        "the price the market is showing, what volatility does it imply?"
    )

    lower, upper = price_bounds(spot, strike, T, r, q, option_type)
    model_px = bs_price(spot, strike, T, r, sigma, q, option_type)

    controls, output = st.columns([2, 3])

    with controls:
        with st.container(border=True):
            # Seeded from the model price once, then left alone: if the default
            # tracked the sidebar vol, solving for IV would just hand back the
            # vol you started with.
            st.session_state.setdefault("market_price", float(round(model_px * 1.15, 2)))
            market_price = st.number_input(
                "Market option price",
                min_value=0.0,
                step=0.05,
                key="market_price",
                help="The mid or last traded price of the option you are looking at.",
            )
            st.caption(
                "No-arbitrage range for this contract: {} to {}.".format(
                    money(lower), money(upper)
                )
            )

    iv_greeks = None

    with output:
        try:
            result = implied_vol(market_price, spot, strike, T, r, q, option_type)
            iv = result.sigma
            iv_greeks = scaled_greeks(spot, strike, T, r, iv, q, option_type)

            with st.container(border=True):
                head, action = st.columns([2, 1], vertical_alignment="center")
                with head:
                    st.metric(
                        "Implied volatility",
                        "{:.2f}%".format(iv * 100.0),
                        "{:+.2f} pts vs input vol".format(iv * 100.0 - vol_pct),
                        delta_color="off",
                    )
                with action:

                    def _apply_iv(value: float = iv * 100.0) -> None:
                        st.session_state["_pending_vol"] = round(value, 2)

                    st.button(
                        "Use in other tabs",
                        icon=":material/sync:",
                        on_click=_apply_iv,
                        width="stretch",
                    )

                st.caption(
                    "Solved by {} in {} iteration{} - repriced at {}, {} off the quote.".format(
                        result.method,
                        result.iterations,
                        "" if result.iterations == 1 else "s",
                        money(market_price + result.price_error),
                        money(abs(result.price_error)),
                    )
                )

        except ImpliedVolError as exc:
            st.error(str(exc), icon=":material/error:")
            st.caption(
                "Check the spot, strike and expiry in the sidebar - a price outside the "
                "no-arbitrage band usually means one of those inputs is wrong."
            )

    if iv_greeks is not None:
        st.markdown("**Greeks at the implied vol**")
        with st.container(horizontal=True):
            st.metric("Delta", "{:.4f}".format(iv_greeks["delta"]), border=True)
            st.metric("Gamma", "{:.4f}".format(iv_greeks["gamma"]), border=True)
            st.metric("Vega", "{:.4f}".format(iv_greeks["vega"]), border=True)
            st.metric("Theta", "{:.4f}".format(iv_greeks["theta"]), border=True)
            st.metric("Rho", "{:.4f}".format(iv_greeks["rho"]), border=True)

        if iv_greeks["vega"] < VEGA_RELIABILITY_FLOOR:
            st.warning(
                "Vega is only {:.4f} per vol point, so the price barely responds to "
                "volatility. This contract is too deep in or out of the money for its "
                "implied vol to mean much.".format(iv_greeks["vega"]),
                icon=":material/warning:",
            )

    with st.container(border=True):
        st.markdown("**Why the price is monotone in volatility**")
        st.caption(
            "Vega is positive everywhere, so the option price rises strictly with "
            "volatility. That guarantees exactly one solution, and lets bisection work as "
            "a fallback whenever Newton's vega-based step breaks down."
        )
        vol_ceiling = min(3.0, max(1.2, sigma * 3.0 + 0.6))
        monotone_axis = np.linspace(0.01, vol_ceiling, 160)
        curve = pd.DataFrame(
            {
                "Volatility": monotone_axis * 100.0,
                "Model price": bs_price(spot, strike, T, r, monotone_axis, q, option_type),
            }
        )
        line = (
            alt.Chart(curve)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X("Volatility:Q", title="Volatility (%)"),
                y=alt.Y("Model price:Q", title="Model price ($)"),
                tooltip=[
                    alt.Tooltip("Volatility:Q", format=".1f"),
                    alt.Tooltip("Model price:Q", format="$.2f"),
                ],
            )
        )
        quote = (
            alt.Chart(pd.DataFrame({"y": [market_price], "label": ["Market quote"]}))
            .mark_rule(color=ACCENT_RED, strokeDash=[5, 4])
            .encode(y="y:Q", tooltip=["label:N", alt.Tooltip("y:Q", format="$.2f")])
        )
        st.altair_chart((line + quote).properties(height=260))


# ================================================== 4. Volatility smile (V2) ==

with tab_smile:
    st.subheader("Volatility smile")
    st.caption(
        "Black-Scholes assumes one volatility for every strike. Real option chains do not "
        "agree: solve for implied vol strike by strike and the result curves."
    )

    controls = st.container(horizontal=True, vertical_alignment="bottom")
    with controls:
        st.caption("Using ticker **{}** from the sidebar.".format(ticker_symbol or "-"))
        fetch = st.button(
            "Load option chain", icon=":material/download:", type="primary"
        )

    if fetch:
        st.session_state["smile_symbol"] = ticker_symbol

    active_symbol = st.session_state.get("smile_symbol")

    if not active_symbol:
        st.info(
            "Load the option chain for the sidebar ticker to see the smile. Requires an "
            "internet connection (data via Yahoo Finance).",
            icon=":material/info:",
        )
    else:
        try:
            with st.spinner("Fetching {} option chain...".format(active_symbol)):
                expiries = cached_expiries(active_symbol)
                live_spot = cached_spot(active_symbol)
        except Exception as exc:  # noqa: BLE001 - surface any network/data failure
            expiries, live_spot = [], None
            st.error(
                "Could not load data for {}: {}".format(active_symbol, exc),
                icon=":material/error:",
            )

        if expiries and live_spot:
            picker = st.container(horizontal=True, vertical_alignment="bottom")
            with picker:
                expiry = st.selectbox("Expiry", expiries, index=min(2, len(expiries) - 1))
                moneyness_band = st.slider("Strikes within +/- % of spot", 5, 60, 25, step=5)
                min_volume = st.number_input(
                    "Minimum volume",
                    min_value=0,
                    value=10,
                    step=1,
                    help="Drops strikes that barely traded, whose last price is stale.",
                )
                otm_only = st.toggle(
                    "Out-of-the-money only",
                    value=True,
                    help=(
                        "In-the-money options are thinly traded and their premium is mostly "
                        "intrinsic value, so their implied vols are noise. Desks build vol "
                        "surfaces from out-of-the-money quotes for exactly this reason."
                    ),
                )

            days_left = max((pd.Timestamp(expiry) - pd.Timestamp.now().normalize()).days, 1)
            T_live = days_left / DAYS_PER_YEAR

            try:
                chain = cached_chain(active_symbol, expiry)
            except Exception as exc:  # noqa: BLE001
                chain = pd.DataFrame()
                st.error("Could not load the chain: {}".format(exc), icon=":material/error:")

            if not chain.empty:
                chain = chain[chain["price"] > 0]
                chain = chain[chain["volume"] >= min_volume]
                chain = chain[
                    chain["strike"].between(
                        live_spot * (1 - moneyness_band / 100.0),
                        live_spot * (1 + moneyness_band / 100.0),
                    )
                ]
                if otm_only:
                    chain = chain[
                        ((chain["type"] == "call") & (chain["strike"] >= live_spot))
                        | ((chain["type"] == "put") & (chain["strike"] <= live_spot))
                    ]

                solved = []
                for row in chain.itertuples(index=False):
                    try:
                        res = implied_vol(
                            row.price, live_spot, row.strike, T_live, r, q, row.type
                        )
                    except ImpliedVolError:
                        continue
                    contract_vega = (
                        vega(live_spot, row.strike, T_live, r, res.sigma, q) / 100.0
                    )
                    if contract_vega < VEGA_RELIABILITY_FLOOR:
                        continue  # price carries no volatility information
                    solved.append(
                        {
                            "Strike": float(row.strike),
                            "Type": row.type,
                            "Price": float(row.price),
                            "Source": row.source,
                            "Implied vol (%)": res.sigma * 100.0,
                            "Yahoo IV (%)": float(row.impliedVolatility) * 100.0,
                            "Volume": int(row.volume),
                            "Open interest": int(row.openInterest),
                            "Moneyness": float(row.strike) / live_spot,
                            "Last trade": pd.Timestamp(row.lastTradeDate),
                        }
                    )

                smile = pd.DataFrame(solved)

                if smile.empty:
                    st.warning(
                        "No strikes passed the liquidity and vega filters. Widen the "
                        "moneyness band or lower the minimum volume.",
                        icon=":material/filter_alt_off:",
                    )
                else:
                    atm = smile.iloc[(smile["Strike"] - live_spot).abs().argmin()]
                    with st.container(horizontal=True):
                        st.metric("Spot", money(live_spot), border=True)
                        st.metric("Days to expiry", str(days_left), border=True)
                        st.metric(
                            "ATM implied vol",
                            "{:.1f}%".format(atm["Implied vol (%)"]),
                            border=True,
                        )
                        st.metric("Strikes solved", str(len(smile)), border=True)

                    with st.container(border=True):
                        st.markdown("**Implied volatility by strike**")
                        points = (
                            alt.Chart(smile)
                            .mark_circle(size=70, opacity=0.85)
                            .encode(
                                x=alt.X(
                                    "Strike:Q", scale=alt.Scale(zero=False), title="Strike ($)"
                                ),
                                y=alt.Y(
                                    "Implied vol (%):Q",
                                    scale=alt.Scale(zero=False),
                                    title="Implied volatility (%)",
                                ),
                                color=alt.Color(
                                    "Type:N", title=None, legend=alt.Legend(orient="top")
                                ),
                                tooltip=[
                                    alt.Tooltip("Strike:Q", format="$.2f"),
                                    "Type:N",
                                    alt.Tooltip("Price:Q", format="$.2f"),
                                    "Source:N",
                                    alt.Tooltip("Implied vol (%):Q", format=".2f"),
                                    alt.Tooltip("Yahoo IV (%):Q", format=".2f"),
                                    "Volume:Q",
                                ],
                            )
                        )
                        trend = (
                            alt.Chart(smile)
                            .transform_loess(
                                "Strike", "Implied vol (%)", groupby=["Type"], bandwidth=0.45
                            )
                            .mark_line(strokeWidth=2)
                            .encode(
                                x=alt.X("Strike:Q", scale=alt.Scale(zero=False)),
                                y=alt.Y("Implied vol (%):Q", scale=alt.Scale(zero=False)),
                                color=alt.Color("Type:N", legend=None),
                            )
                        )
                        spot_rule = (
                            alt.Chart(pd.DataFrame({"x": [live_spot], "label": ["Spot"]}))
                            .mark_rule(color=NEUTRAL, strokeDash=[4, 4])
                            .encode(
                                x="x:Q",
                                tooltip=["label:N", alt.Tooltip("x:Q", format="$.2f")],
                            )
                        )
                        st.altair_chart((points + trend + spot_rule).properties(height=380))
                        st.caption(
                            "Equity options usually show a skew rather than a symmetric "
                            "smile: downside puts trade at higher implied vol than upside "
                            "calls, because crash protection is in demand. A single "
                            "Black-Scholes volatility cannot fit all of these strikes at "
                            "once, which is the model's best-known limitation."
                        )

                    with st.container(border=True):
                        st.markdown("**Solved chain**")
                        st.dataframe(
                            smile.sort_values(["Type", "Strike"]),
                            hide_index=True,
                            column_config={
                                "Strike": st.column_config.NumberColumn(format="$%.2f"),
                                "Price": st.column_config.NumberColumn(format="$%.2f"),
                                "Implied vol (%)": st.column_config.NumberColumn(format="%.2f"),
                                "Yahoo IV (%)": st.column_config.NumberColumn(format="%.2f"),
                                "Moneyness": st.column_config.NumberColumn(format="%.3f"),
                                "Last trade": st.column_config.DatetimeColumn(
                                    format="MMM DD, HH:mm"
                                ),
                            },
                        )
                        st.caption(
                            "'Implied vol' is solved here from the observed price using the "
                            "sidebar rate and dividend yield. 'Yahoo IV' is the vendor's own "
                            "number - differences come from their rate, dividend and "
                            "American-exercise assumptions. Rows priced off a last trade "
                            "rather than a live bid/ask inherit that print's staleness."
                        )


# ============================================= 5. Underlying and realised vol ==

with tab_underlying:
    st.subheader("Underlying")
    st.caption(
        "The volatility an option is quoting is a forecast. This is what the underlying "
        "has actually done."
    )

    controls = st.container(horizontal=True, vertical_alignment="bottom")
    with controls:
        st.caption("Using ticker **{}** from the sidebar.".format(ticker_symbol or "-"))
        period = st.segmented_control(
            "History window",
            options=["3mo", "6mo", "1y", "2y", "5y"],
            default="1y",
            key="history_period",
        )
        load_history = st.button("Load history", icon=":material/show_chart:", type="primary")

    if load_history:
        st.session_state["history_symbol"] = ticker_symbol

    history_symbol = st.session_state.get("history_symbol")

    if not history_symbol:
        st.info(
            "Load price history for the sidebar ticker to compare realised volatility "
            "against the volatility you are pricing with.",
            icon=":material/info:",
        )
    else:
        try:
            with st.spinner("Fetching {} history...".format(history_symbol)):
                history = cached_history(history_symbol, period or "1y")
        except Exception as exc:  # noqa: BLE001
            history = pd.DataFrame()
            st.error(
                "Could not load history for {}: {}".format(history_symbol, exc),
                icon=":material/error:",
            )

        if not history.empty:
            closes = history["Close"]
            realised_full = market_data.realised_volatility(closes)
            realised_30 = market_data.realised_volatility(closes, window=30)
            total_return = float(closes.iloc[-1] / closes.iloc[0] - 1.0)

            with st.container(horizontal=True):
                st.metric("Last close", money(float(closes.iloc[-1])), border=True)
                st.metric(
                    "Return over window",
                    "{:+.1%}".format(total_return),
                    border=True,
                )
                st.metric(
                    "Realised vol (window)",
                    "{:.1f}%".format(realised_full * 100.0),
                    border=True,
                )
                st.metric(
                    "Realised vol (30d)",
                    "{:.1f}%".format(realised_30 * 100.0),
                    "{:+.1f} pts vs pricing vol".format(realised_30 * 100.0 - vol_pct),
                    delta_color="off",
                    border=True,
                )

            def _apply_realised(value: float = realised_30 * 100.0) -> None:
                st.session_state["_pending_vol"] = round(value, 2)

            def _apply_last_close(value: float = float(closes.iloc[-1])) -> None:
                st.session_state["_pending_spot"] = round(value, 2)

            actions = st.container(horizontal=True)
            with actions:
                st.button(
                    "Price with 30-day realised vol",
                    icon=":material/sync:",
                    on_click=_apply_realised,
                )
                st.button(
                    "Use last close as spot",
                    icon=":material/sync:",
                    on_click=_apply_last_close,
                )

            price_history = pd.DataFrame(
                {"Date": closes.index, "Close": closes.to_numpy(dtype=float)}
            )

            chart_left, chart_right = st.columns([3, 2])

            with chart_left:
                with st.container(border=True):
                    st.markdown("**Price history**")
                    price_line = (
                        alt.Chart(price_history)
                        .mark_line(strokeWidth=1.8)
                        .encode(
                            x=alt.X("Date:T", title=None),
                            y=alt.Y(
                                "Close:Q", scale=alt.Scale(zero=False), title="Close ($)"
                            ),
                            tooltip=[
                                alt.Tooltip("Date:T"),
                                alt.Tooltip("Close:Q", format="$.2f"),
                            ],
                        )
                    )
                    strike_line = (
                        alt.Chart(pd.DataFrame({"y": [strike], "label": ["Strike"]}))
                        .mark_rule(color=ACCENT_RED, strokeDash=[6, 4])
                        .encode(
                            y="y:Q", tooltip=["label:N", alt.Tooltip("y:Q", format="$.2f")]
                        )
                    )
                    st.altair_chart((price_line + strike_line).properties(height=320))
                    st.caption(
                        "Red dashed line is the strike from the sidebar, for context on how "
                        "far out of the money the contract sits."
                    )

            with chart_right:
                with st.container(border=True):
                    st.markdown("**Distribution of daily returns**")
                    log_returns = np.log(closes / closes.shift(1)).dropna() * 100.0
                    returns_frame = pd.DataFrame({"Daily log return (%)": log_returns.to_numpy()})
                    histogram = (
                        alt.Chart(returns_frame)
                        .mark_bar(opacity=0.85)
                        .encode(
                            x=alt.X(
                                "Daily log return (%):Q",
                                bin=alt.Bin(maxbins=40),
                                title="Daily log return (%)",
                            ),
                            y=alt.Y("count()", title="Days"),
                            tooltip=[alt.Tooltip("count()", title="Days")],
                        )
                    )
                    st.altair_chart(histogram.properties(height=320))
                    st.caption(
                        "Black-Scholes assumes these are normally distributed. Real returns "
                        "have fatter tails than a normal, which is part of why the market "
                        "charges more for far out-of-the-money options than the model says."
                    )
