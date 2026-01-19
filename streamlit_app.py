import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import random
import json
from datetime import datetime, timedelta
from scipy.stats import linregress

# --- 1. PERSISTENCE & GLOBAL STATE ---
DEFAULTS = {
    'atr_mult': 1.5,
    'trail_mult': 3.0,
    'regime_mode': "Adaptive (Hybrid)",
    'universe': "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,NFLX,META,AMD,ADBE,LLY,AVGO,COST,CRM,JPM,V,MA,PG,WMT,GE,JNJ,CAH,UNH,ABBV,PEP,KO,MCD,LIN,QCOM",
    'vol_target': 15.0,
    'mc_sims': 1000,
    'start_date': datetime(2020, 1, 1).strftime('%Y-%m-%d'),
    't_cost': 0.1,
    'adx_threshold': 20.0,
    'kelly_fraction': 0.5,
    'allow_fractional': True,
    'stop_buffer': 2.5,
    'pos_size_pct': 10.0 
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.set_page_config(page_title="💸 John's Experiment: Ubos Pera Edition 💸", layout="wide")

# --- 2. SIDEBAR COMMAND CENTER ---
with st.sidebar:
    st.header("⚙️ Strategy Settings")
    st.session_state.universe = st.text_input("Universe", st.session_state.universe)
    TICKERS = [t.strip() for t in st.session_state.universe.split(',')]
    BENCHMARK = st.selectbox("Benchmark", ["SPY", "QQQ", "DIA", "IWM"], index=1)
    PORTFOLIO_VALUE_GBP = st.number_input("Account Balance (£)", min_value=100, value=1400)
    
    st.divider()
    st.header("🎮 Hybrid Regime Logic")
    st.session_state.regime_mode = st.radio("Strategy Mode", ["Adaptive (Hybrid)", "Momentum Only", "Mean Reversion Only"])
    st.session_state.adx_threshold = st.slider("ADX Trend/Range Cutoff", 10.0, 35.0, float(st.session_state.adx_threshold))
    
    st.divider()
    st.header("🛡️ Risk & Trailing Stops")
    st.session_state.pos_size_pct = st.slider("Allocation per Stock (%)", 5.0, 33.3, float(st.session_state.pos_size_pct), 0.5)
    st.session_state.atr_mult = st.slider("Initial ATR Stop Multiplier", 1.0, 5.0, float(st.session_state.atr_mult), 0.1)
    st.session_state.trail_mult = st.slider("Trailing ATR Multiplier", 2.0, 6.0, float(st.session_state.trail_mult), 0.1)
    st.session_state.kelly_fraction = st.slider("Kelly Fraction (Leverage)", 0.1, 1.0, float(st.session_state.kelly_fraction), 0.05)
    st.session_state.stop_buffer = st.slider("Stop Warning Buffer (%)", 1.0, 10.0, float(st.session_state.stop_buffer), 0.5)
    st.session_state.allow_fractional = st.checkbox("Enable Fractional Shares", value=st.session_state.allow_fractional)
    
    st.divider()
    st.header("💾 Config Persistence")
    current_config = {k: v for k, v in st.session_state.items() if k in DEFAULTS}
    st.download_button("📩 Download Config", data=json.dumps(current_config, default=str), file_name="alpha_config_v58.json")
    
    uploaded_file = st.file_uploader("📂 Upload Config", type="json")
    if uploaded_file is not None:
        st.session_state.update(json.load(uploaded_file)); st.rerun()
    
    st.session_state.start_date = st.date_input("Start Date", pd.to_datetime(st.session_state.start_date))

# --- 3. CORE ANALYTICS ENGINE ---
def calculate_rsi(series, n=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=n).mean()
    rs = gain / (loss.replace(0, 0.00001))
    return 100 - (100 / (1 + rs))

def get_slope(series):
    if len(series) < 20: return 0
    y, x = series.values, np.arange(len(series))
    slope, _, _, _, _ = linregress(x, y)
    return slope

def calculate_adx(df, n=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    up, down = h.diff(), -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0).rolling(n).mean()
    minus_dm = down.where((down > up) & (down > 0), 0).rolling(n).mean()
    plus_di, minus_di = 100 * (plus_dm/atr), 100 * (minus_dm/atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di.replace(0, 0.00001))
    return dx.rolling(n).mean()

def calculate_hurst(ts):
    if len(ts) < 50: return 0.5
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

@st.cache_data(ttl=86400)
def fetch_data_v58(u_str, bench, start):
    t_list = list(set([t.strip() for t in u_str.split(',')] + [bench, "GBPUSD=X"]))
    return yf.download(t_list, start=pd.to_datetime(start)-timedelta(days=365), auto_adjust=True)

raw = fetch_data_v58(st.session_state.universe, BENCHMARK, st.session_state.start_date)
prices = raw['Close'].ffill().bfill()
bench_adx = calculate_adx(raw.xs(BENCHMARK, axis=1, level=1))

@st.cache_data(ttl=3600)
def run_v58_backtest(u_str, bench_name, start_dt, mode, vol_t, t_c, adx_t):
    tickers = [t.strip() for t in u_str.split(',')]
    m_prices = prices[tickers].resample('ME').last()
    mom_12_1 = (m_prices.shift(1)/m_prices.shift(12))-1
    rsi_vals = prices[tickers].apply(calculate_rsi)
    
    b_rets, dates, trade_log, raw_trades = [], [], [], []
    active_pos, s_rets_list = {}, []
    bench_p, sma200 = prices[bench_name], prices[bench_name].rolling(200).mean()

    for i in range(13, len(m_prices)):
        dt = m_prices.index[i]
        if dt.date() < pd.to_datetime(start_dt).date(): continue
        p_c, s200_c, adx_c = bench_p.asof(dt), sma200.asof(dt), bench_adx.asof(dt)
        h = calculate_hurst(bench_p.loc[:dt].tail(100).values)
        b_rets.append((bench_p.asof(dt)/bench_p.asof(m_prices.index[i-1]))-1)
        
        is_trending = (h > 0.50 and adx_c > adx_t)
        picks, current_strat = [], "Cash"
        
        if (mode == "Adaptive (Hybrid)" and is_trending) or mode == "Momentum Only":
            if p_c > s200_c:
                current_strat = "Momentum"
                potential = mom_12_1.iloc[i].dropna()
                slopes = prices[tickers].loc[:dt].tail(60).apply(get_slope)
                valid = potential[slopes > 0].index
                picks = list(potential.loc[valid].sort_values(ascending=False).head(3).index)
        elif (mode == "Adaptive (Hybrid)" and not is_trending) or mode == "Mean Reversion Only":
            current_strat = "Mean Reversion"
            c_rsi = rsi_vals.loc[:dt].iloc[-1]
            picks = list(c_rsi[c_rsi < 35].sort_values().head(3).index)

        for t in list(active_pos.keys()):
            if t not in picks:
                ret = (float(m_prices.loc[dt, t])/active_pos.pop(t) - 1)
                raw_trades.append(ret); trade_log.append({"Ticker": t, "Mode": current_strat, "Exit": dt.strftime('%Y-%m'), "Return": f"{ret:.1%}"})
        for t in picks:
            if t not in active_pos: active_pos[t] = float(m_prices.iloc[i-1][t])
        
        v_h = bench_p.pct_change().loc[:dt].tail(20).std() * np.sqrt(252)
        exp = min(1.0, (vol_t/100)/v_h) if v_h > 0 else 1.0
        m_ret = ((m_prices.loc[dt, picks]/m_prices.iloc[i-1][picks])-1).mean() if picks else 0
        s_rets_list.append((m_ret * exp) - (t_c/100))
        dates.append(dt)
        
    return s_rets_list, b_rets, dates, (picks if 'picks' in locals() else []), trade_log, raw_trades, h, adx_c, current_strat, mom_12_1

s_raw, b_raw, dts, final_picks, ledger, r_trades, last_h, last_adx, active_strat, m12 = run_v58_backtest(
    st.session_state.universe, BENCHMARK, st.session_state.start_date, 
    st.session_state.regime_mode, st.session_state.vol_target, 0.1, st.session_state.adx_threshold
)

# --- 4. DASHBOARD UI ---
st.title("💸 John's Experiment: Ubos Pera Edition 💸")
s_rets = pd.Series(s_raw, index=dts)

if not s_rets.empty:
    wins, losses = [r for r in r_trades if r > 0], [r for r in r_trades if r <= 0]
    wr = len(wins)/len(r_trades) if r_trades else 0
    pf = abs((len(wins)*np.mean(wins))/(len(losses)*np.mean(losses))) if losses else 0
    
    # NEW: Sharpe Ratio calculation
    excess_rets = s_rets - (0.02 / 12) # Assuming 2% risk-free rate
    sharpe = (excess_rets.mean() * 12) / (s_rets.std() * np.sqrt(12)) if len(s_rets) > 1 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Profit Factor", f"{pf:.2f}", f"Win Rate: {wr:.1%}")
    c2.metric("Sharpe Ratio", f"{sharpe:.2f}") # RESTORED
    c3.metric("Active Strategy", active_strat)
    c4.metric("Market Hurst", f"{last_h:.2f}", "Trending" if last_h > 0.5 else "Ranging")
    c5.metric("ADX Strength", f"{last_adx:.1f}", "Strong" if last_adx > st.session_state.adx_threshold else "Weak")

    t1, t2, t3, t4, t5 = st.tabs(["🚀 Live Orders", "📊 Metrics", "📉 Stress Tests", "🔔 Sentinel", "📜 History"])
    
    with t1:
        if final_picks:
            st.header(f"🛒 Order Console (£ GBP Balance: £{PORTFOLIO_VALUE_GBP:,.2f})")
            trade_rows = []; fx = raw['Close']["GBPUSD=X"].iloc[-1]
            total_risk_gbp = 0; total_invested_gbp = 0
            
            for s in final_picks:
                h_p, l_p, c_p = raw['High'][s], raw['Low'][s], raw['Close'][s]
                tr = pd.concat([h_p-l_p, abs(h_p-c_p.shift(1)), abs(l_p-c_p.shift(1))], axis=1).max(axis=1)
                p_usd = float(prices[s].iloc[-1])
                highest_high = h_p.tail(14).max()
                
                # Sizing logic preserved
                alloc_gbp = PORTFOLIO_VALUE_GBP * (st.session_state.pos_size_pct / 100)
                raw_shares = (alloc_gbp * fx * st.session_state.kelly_fraction) / p_usd
                shares = round(raw_shares, 4) if st.session_state.allow_fractional else int(raw_shares)
                
                # Exit logic
                atr_14 = tr.tail(14).mean()
                stop_p_usd = p_usd - (atr_14 * st.session_state.atr_mult)
                trail_stop_usd = highest_high - (atr_14 * st.session_state.trail_mult)
                effective_stop = max(stop_p_usd, trail_stop_usd)
                
                # Risk Metrics
                risk_dist_pct = ((p_usd - effective_stop) / p_usd) * 100
                risk_amt_gbp = ((p_usd - effective_stop) * shares) / fx
                total_risk_gbp += risk_amt_gbp
                total_invested_gbp += (shares * p_usd) / fx

                # Trade Efficiency
                entry_est = c_p.tail(14).iloc[0]
                potential_move = max(0.01, highest_high - entry_est)
                captured_move = effective_stop - entry_est
                efficiency = (captured_move / potential_move) * 100
                
                # Confidence Rank RESTORED
                conf_rank = m12.iloc[-1].rank(pct=True)[s]
                
                trade_rows.append({
                    "Ticker": s, 
                    "Confidence": f"{(conf_rank * 100):.0f}%", # RESTORED
                    "Price ($)": p_usd,
                    "Active Exit ($)": effective_stop,
                    "Risk %": round(risk_dist_pct, 2),
                    "Risk (£)": round(risk_amt_gbp, 2),
                    "Efficiency (%)": round(efficiency, 1),
                    "Shares": shares
                })
            
            r_col1, r_col2, r_col3 = st.columns(3)
            r_col1.metric("Total Risk Exposure (£)", f"£{total_risk_gbp:,.2f}")
            r_col2.metric("Portfolio Exposure (%)", f"{(total_risk_gbp/PORTFOLIO_VALUE_GBP):.2%}")
            r_col3.metric("Diversification Status", "Active")
            
            df_display = pd.DataFrame(trade_rows)
            def highlight_danger(row):
                if row['Risk %'] < st.session_state.stop_buffer:
                    return ['background-color: #ff4b4b; color: white'] * len(row)
                return [''] * len(row)
            
            st.dataframe(df_display.style.apply(highlight_danger, axis=1).format({
                "Price ($)": "${:,.2f}", "Active Exit ($)": "${:,.2f}", 
                "Risk %": "{:.2f}%", "Risk (£)": "£{:,.2f}", "Efficiency (%)": "{:.1f}%"
            }), use_container_width=True)
        else: st.warning("NO TRADES: Criteria not met.")

    with t2:
        st.line_chart(pd.DataFrame({"Hybrid Equity": (1+s_rets).cumprod(), BENCHMARK: (1+pd.Series(b_raw, index=dts)).cumprod()}))

    with t3:
        if st.button("▶️ Run Monte Carlo"):
            with st.spinner("Processing..."):
                paths, dds = [], []
                for _ in range(st.session_state.mc_sims):
                    sh = random.sample(list(s_rets), len(s_rets))
                    path = np.cumprod(1 + np.array(sh)); paths.append(path)
                    dds.append((path / np.maximum.accumulate(path) - 1).min())
                st.session_state.mc_results = {"paths": paths, "dds": dds}
        if "mc_results" in st.session_state and st.session_state.mc_results:
            st.line_chart(pd.DataFrame(st.session_state.mc_results["paths"]).T)

    with t4:
        st.header("🔔 Universe Sentinel")
        u_sent = []
        for t in TICKERS:
            try:
                cal = yf.Ticker(t).calendar
                e_date = cal['Earnings Date'][0] if isinstance(cal, dict) and 'Earnings Date' in cal else "N/A"
                u_sent.append({"Ticker": t, "Next Earnings": e_date})
            except: u_sent.append({"Ticker": t, "Next Earnings": "Error"})
        st.table(pd.DataFrame(u_sent).sort_values("Next Earnings"))

    with t5:
        st.dataframe(pd.DataFrame(ledger).sort_index(ascending=False), use_container_width=True)
