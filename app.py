"""Options analytics dashboard: Black-Scholes pricing, implied vol, and scenarios.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

from black_scholes import bs_price, d1_d2, scaled_greeks, vega
from implied_vol import ImpliedVolError, implied_vol, price_bounds

st.set_page_config(
    page_title="Options pricer",
    page_icon=":material/candlestick_chart:",
    layout="wide",
)

DAYS_PER_YEAR = 365.0

# A Greek is only worth reading if the price actually responds to the input.
# Below roughly 1 cent of P&L per vol point, an implied vol is noise.
VEGA_RELIABILITY_FLOOR = 0.01


# Widget values cannot be written after the widget exists, so the "use this IV"
# button parks the new value here and it is applied at the top of the next run.
if "_pending_vol" in st.session_state:
    st.session_state["vol_pct"] = st.session_state.pop("_pending_vol")


# ---------------------------------------------------------------- contract ---

with st.sidebar:
    st.header("Contract")

    option_label = st.segmented_control(
        "Option type",
        options=["Call", "Put"],
        default="Call",
        key="option_label",
    )
    option_type = (option_label or "Call").lower()

    spot = st.number_input("Spot price (S)", min_value=0.01, value=250.0, step=1.0)
    strike = st.number_input("Strike price (K)", min_value=0.01, value=260.0, step=1.0)
    days = st.number_input("Days to expiry", min_value=1, max_value=3650, value=30, step=1)
    # Seeded once in session state rather than with `value=`, so the "use this
    # IV" button on the implied-vol tab can write to it without Streamlit
    # warning about a widget that has both a default and a stored value.
    st.session_state.setdefault("vol_pct", 28.4)
    vol_pct = st.number_input(
        "Volatility (annualised, %)",
        min_value=0.1,
        max_value=500.0,
        step=0.5,
        key="vol_pct",
        help="Used by the pricer, the scenario charts and the heatmap.",
    )

    st.header("Market")
    rate_pct = st.number_input("Risk-free rate (%)", min_value=-5.0, max_value=25.0, value=4.5, step=0.1)
    div_pct = st.number_input("Dividend yield (%)", min_value=0.0, max_value=25.0, value=0.0, step=0.1)

    st.caption(
        "Time is measured in calendar days over a 365-day year. "
        "Rates are continuously compounded."
    )

T = days / DAYS_PER_YEAR
r = rate_pct / 100.0
q = div_pct / 100.0
sigma = vol_pct / 100.0


# ----------------------------------------------------------------- helpers ---


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
    marks = pd.DataFrame(
        {"x": [S, K], "label": ["Spot", "Strike"], "dash": [[1, 0], [4, 4]]}
    )
    return (
        alt.Chart(marks)
        .mark_rule(color="#8c8c8c", opacity=0.8)
        .encode(
            x=alt.X("x:Q"),
            strokeDash=alt.StrokeDash("label:N", title=None),
            tooltip=["label:N", alt.Tooltip("x:Q", format=".2f", title="Level")],
        )
    )


def money(x: float) -> str:
    return "${:,.2f}".format(x)


# ------------------------------------------------------------------- header ---

st.title(":material/candlestick_chart: Options pricer")
st.caption(
    "Black-Scholes pricing, implied volatility from market quotes, "
    "and scenario analysis across spot, volatility and time."
)

tab_price, tab_iv, tab_scenario, tab_smile = st.tabs(
    ["Pricer", "Implied volatility", "Scenario analysis", "Volatility smile"]
)


# ==================================================== 1. Black-Scholes pricer ==

with tab_price:
    st.subheader("Black-Scholes pricer")

    g = scaled_greeks(spot, strike, T, r, sigma, q, option_type)
    call_px = bs_price(spot, strike, T, r, sigma, q, "call")
    put_px = bs_price(spot, strike, T, r, sigma, q, "put")

    with st.container(horizontal=True):
        st.metric("Call price", money(call_px), border=True)
        st.metric("Put price", money(put_px), border=True)
        st.metric("Time to expiry", "{:.4f} yr".format(T), "{} days".format(days), delta_color="off", border=True)

    st.markdown("**{} Greeks**".format(option_label))
    with st.container(horizontal=True):
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

    with right:
        with st.container(border=True):
            st.markdown("**Intermediate quantities**")
            d1, d2 = d1_d2(spot, strike, T, r, sigma, q)
            forward = spot * np.exp((r - q) * T)
            st.dataframe(
                pd.DataFrame(
                    {
                        "Quantity": ["d1", "d2", "N(d2)", "Forward price", "Moneyness (S/K)"],
                        "Value": [
                            float(d1),
                            float(d2),
                            float(norm.cdf(d2)),
                            float(forward),
                            float(spot / strike),
                        ],
                    }
                ),
                hide_index=True,
                column_config={"Value": st.column_config.NumberColumn(format="%.4f")},
            )
            st.caption(
                "N(d2) is the risk-neutral probability the call finishes in the money."
            )

    with st.expander("The formula"):
        st.latex(r"d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}")
        st.latex(r"C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)")
        st.latex(r"P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)")
        st.markdown(
            "Put-call parity ties the two together: "
            r"$C - P = S e^{-qT} - K e^{-rT}$."
        )
        parity = call_px - put_px - (spot * np.exp(-q * T) - strike * np.exp(-r * T))
        st.caption("Parity residual for these inputs: {:.2e}".format(parity))


# =============================================== 2. Implied volatility solver ==

with tab_iv:
    st.subheader("Implied volatility")
    st.caption(
        "Black-Scholes turns a volatility into a price. Here we run it backwards: "
        "given the price the market is showing, what volatility does it imply?"
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
                    "Solved by {} in {} iteration{} - repriced at {}, "
                    "{} off the quote.".format(
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
            "volatility. That guarantees exactly one solution, and lets bisection work "
            "as a fallback whenever Newton's vega-based step breaks down."
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
            alt.Chart(pd.DataFrame({"y": [market_price]}))
            .mark_rule(color="#d62728", strokeDash=[5, 4])
            .encode(y="y:Q")
        )
        st.altair_chart((line + quote).properties(height=260))


# =================================================== 3. Scenario visualiser ===

with tab_scenario:
    st.subheader("Scenario analysis")

    settings = st.container(horizontal=True, vertical_alignment="bottom")
    with settings:
        spot_range = st.slider("Spot range (+/- %)", 5, 60, 20, step=5)
        vol_range = st.slider("Vol range (+/- pts)", 2, 30, 10, step=1)
        grid_steps = st.select_slider("Heatmap size", options=[3, 5, 7, 9], value=5)

    grid = spot_grid(spot, spot_range)

    # --- payoff / price profile -------------------------------------------
    price_now = bs_price(grid, strike, T, r, sigma, q, option_type)
    if option_type == "call":
        intrinsic = np.maximum(grid - strike, 0.0)
    else:
        intrinsic = np.maximum(strike - grid, 0.0)
    half_life = bs_price(grid, strike, T / 2.0, r, sigma, q, option_type)

    profile = pd.DataFrame(
        {
            "Underlying price": np.tile(grid, 3),
            "Option price": np.concatenate([price_now, half_life, intrinsic]),
            "Series": np.repeat(
                [
                    "Today ({} days)".format(days),
                    "Halfway ({} days)".format(max(days // 2, 1)),
                    "At expiry (intrinsic)",
                ],
                len(grid),
            ),
        }
    )

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("**Option price vs underlying**")
            profile_chart = (
                alt.Chart(profile)
                .mark_line(strokeWidth=2)
                .encode(
                    x=alt.X("Underlying price:Q", scale=alt.Scale(zero=False), title="Underlying price ($)"),
                    y=alt.Y("Option price:Q", title="Option price ($)"),
                    color=alt.Color("Series:N", title=None, legend=alt.Legend(orient="top")),
                    tooltip=[
                        alt.Tooltip("Underlying price:Q", format="$.2f"),
                        alt.Tooltip("Option price:Q", format="$.2f"),
                        "Series:N",
                    ],
                )
            )
            st.altair_chart((profile_chart + reference_rules(spot, strike)).properties(height=320))
            st.caption(
                "The gap between today's curve and the intrinsic line is time value - "
                "the part of the premium that decays away."
            )

    with right:
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
                .mark_line(strokeWidth=2, color="#1f77b4")
                .encode(
                    x=alt.X("Underlying price:Q", scale=alt.Scale(zero=False), title="Underlying price ($)"),
                    y=alt.Y("Delta:Q", title="Delta", axis=alt.Axis(titleColor="#1f77b4")),
                    tooltip=[alt.Tooltip("Underlying price:Q", format="$.2f"), alt.Tooltip("Delta:Q", format=".3f")],
                )
            )
            gamma_line = (
                alt.Chart(risk)
                .mark_line(strokeWidth=2, color="#ff7f0e", strokeDash=[5, 3])
                .encode(
                    x=alt.X("Underlying price:Q", scale=alt.Scale(zero=False)),
                    y=alt.Y("Gamma:Q", title="Gamma", axis=alt.Axis(titleColor="#ff7f0e")),
                    tooltip=[alt.Tooltip("Underlying price:Q", format="$.2f"), alt.Tooltip("Gamma:Q", format=".4f")],
                )
            )
            st.altair_chart(
                alt.layer(delta_line, gamma_line).resolve_scale(y="independent").properties(height=320)
            )
            st.caption(
                "Blue: delta (left axis). Orange dashed: gamma (right axis). Gamma peaks "
                "near the strike, which is where delta is changing fastest."
            )

    # --- spot x vol heatmap ------------------------------------------------
    with st.container(border=True):
        st.markdown("**Spot x volatility heatmap**")

        spot_axis = np.linspace(spot * (1 - spot_range / 100.0), spot * (1 + spot_range / 100.0), grid_steps)
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

        spot_order = [f"{v:,.0f}" for v in sorted(spot_axis, reverse=True)]
        vol_order = [f"{v:.0f}%" for v in vol_axis]

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
                scale=alt.Scale(scheme="blues"),
                legend=alt.Legend(title="Option price ($)", format="$.2f"),
            ),
            tooltip=[
                alt.Tooltip("Spot:Q", format="$.2f"),
                alt.Tooltip("Vol:Q", format=".1f", title="Vol (%)"),
                alt.Tooltip("Price:Q", format="$.2f"),
            ],
        )
        midpoint = float(heat["Price"].max() + heat["Price"].min()) / 2.0
        labels = base.mark_text(fontSize=13, fontWeight="bold").encode(
            text=alt.Text("Price:Q", format=".2f"),
            color=alt.condition(
                alt.datum.Price > midpoint, alt.value("white"), alt.value("#1a1a1a")
            ),
        )
        st.altair_chart((cells + labels).properties(height=60 + 46 * grid_steps))
        st.caption(
            "Each cell reprices the same {} at a different spot and volatility, holding "
            "{} days to expiry fixed. Reading across a row isolates vega; reading down a "
            "column isolates delta and gamma.".format(option_type, days)
        )

    # --- time decay --------------------------------------------------------
    with st.container(border=True):
        st.markdown("**Time decay**")
        day_axis = np.arange(days, 0, -1)
        decay = pd.DataFrame(
            {
                "Days to expiry": day_axis,
                "Option price": bs_price(spot, strike, day_axis / DAYS_PER_YEAR, r, sigma, q, option_type),
            }
        )
        decay_chart = (
            alt.Chart(decay)
            .mark_line(strokeWidth=2, color="#2ca02c")
            .encode(
                x=alt.X("Days to expiry:Q", scale=alt.Scale(reverse=True), title="Days to expiry"),
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


# ================================================== 4. Volatility smile (V2) ==

with tab_smile:
    st.subheader("Volatility smile")
    st.caption(
        "Black-Scholes assumes one volatility for every strike. Real option chains "
        "do not agree: solve for implied vol strike by strike and the result curves."
    )

    @st.cache_data(ttl=900, show_spinner=False)
    def load_expiries(symbol: str) -> list[str]:
        import yfinance as yf

        return list(yf.Ticker(symbol).options)

    @st.cache_data(ttl=900, show_spinner=False)
    def load_spot_price(symbol: str) -> float:
        import yfinance as yf

        info = yf.Ticker(symbol).fast_info
        for key in ("last_price", "lastPrice", "previous_close"):
            try:
                value = info[key]
            except (KeyError, TypeError):
                continue
            if value:
                return float(value)
        history = yf.Ticker(symbol).history(period="5d")
        return float(history["Close"].iloc[-1])

    @st.cache_data(ttl=900, show_spinner=False)
    def load_chain(symbol: str, expiry: str) -> pd.DataFrame:
        import yfinance as yf

        chain = yf.Ticker(symbol).option_chain(expiry)
        columns = [
            "strike",
            "bid",
            "ask",
            "lastPrice",
            "lastTradeDate",
            "volume",
            "openInterest",
            "impliedVolatility",
        ]
        frames = []
        for kind, table in (("call", chain.calls), ("put", chain.puts)):
            part = table[columns].copy()
            part["type"] = kind
            frames.append(part)
        return pd.concat(frames, ignore_index=True)

    controls = st.container(horizontal=True, vertical_alignment="bottom")
    with controls:
        symbol = st.text_input("Ticker", value="AAPL", max_chars=12, width=200).strip().upper()
        fetch = st.button("Load option chain", icon=":material/download:", type="primary")

    if fetch:
        st.session_state["smile_symbol"] = symbol

    active_symbol = st.session_state.get("smile_symbol")

    if not active_symbol:
        st.info(
            "Enter a ticker and load its option chain to see the smile. "
            "Requires an internet connection (data via Yahoo Finance).",
            icon=":material/info:",
        )
    else:
        try:
            with st.spinner("Fetching {} option chain...".format(active_symbol)):
                expiries = load_expiries(active_symbol)
                live_spot = load_spot_price(active_symbol)
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

            expiry_date = pd.Timestamp(expiry)
            days_left = max((expiry_date - pd.Timestamp.now().normalize()).days, 1)
            T_live = days_left / DAYS_PER_YEAR

            try:
                chain = load_chain(active_symbol, expiry)
            except Exception as exc:  # noqa: BLE001
                chain = pd.DataFrame()
                st.error("Could not load the chain: {}".format(exc), icon=":material/error:")

            if not chain.empty:
                chain = chain.copy()
                # A two-sided quote is the better input, but the free Yahoo feed
                # often returns no bid/ask outside market hours. Fall back to the
                # last traded price and label which one was used, because a stale
                # print produces a stale implied vol.
                quoted = (chain["bid"] > 0) & (chain["ask"] > chain["bid"])
                chain["mid"] = np.where(
                    quoted, (chain["bid"] + chain["ask"]) / 2.0, chain["lastPrice"]
                )
                chain["source"] = np.where(quoted, "bid/ask mid", "last traded")

                # Yahoo leaves volume and open interest blank on untraded strikes.
                chain["volume"] = chain["volume"].fillna(0)
                chain["openInterest"] = chain["openInterest"].fillna(0)

                chain = chain[chain["mid"] > 0]
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
                            row.mid, live_spot, row.strike, T_live, r, q, row.type
                        )
                    except ImpliedVolError:
                        continue
                    contract_vega = vega(live_spot, row.strike, T_live, r, res.sigma, q) / 100.0
                    if contract_vega < VEGA_RELIABILITY_FLOOR:
                        continue  # price carries no volatility information
                    solved.append(
                        {
                            "Strike": float(row.strike),
                            "Type": row.type,
                            "Price": float(row.mid),
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
                        st.metric("ATM implied vol", "{:.1f}%".format(atm["Implied vol (%)"]), border=True)
                        st.metric("Strikes solved", str(len(smile)), border=True)

                    with st.container(border=True):
                        st.markdown("**Implied volatility by strike**")
                        points = (
                            alt.Chart(smile)
                            .mark_circle(size=70, opacity=0.8)
                            .encode(
                                x=alt.X("Strike:Q", scale=alt.Scale(zero=False), title="Strike ($)"),
                                y=alt.Y(
                                    "Implied vol (%):Q",
                                    scale=alt.Scale(zero=False),
                                    title="Implied volatility (%)",
                                ),
                                color=alt.Color("Type:N", title=None, legend=alt.Legend(orient="top")),
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
                            .transform_loess("Strike", "Implied vol (%)", groupby=["Type"], bandwidth=0.45)
                            .mark_line(strokeWidth=2)
                            .encode(
                                x=alt.X("Strike:Q", scale=alt.Scale(zero=False)),
                                y=alt.Y("Implied vol (%):Q", scale=alt.Scale(zero=False)),
                                color=alt.Color("Type:N", legend=None),
                            )
                        )
                        spot_rule = (
                            alt.Chart(pd.DataFrame({"x": [live_spot]}))
                            .mark_rule(color="#8c8c8c", strokeDash=[4, 4])
                            .encode(x="x:Q", tooltip=alt.Tooltip("x:Q", format="$.2f", title="Spot"))
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
