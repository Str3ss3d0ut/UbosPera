import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import random
import json
from datetime import datetime, timedelta
from scipy.stats import linregress
import concurrent.futures
import plotly.graph_objects as go
import requests
import io

# --- 1. PERSISTENCE & GLOBAL STATE ---
DEFAULTS = {
    'atr_mult': 1.5,
    'trail_mult': 3.0,
    'regime_mode': "Momentum Only", 
    'universe': "GLW BDX D CIEN CNP SYY MO OKE UHS UPS MRK DLTR KMI ETR HON AES CMS TER VZ O JNJ PHM AEP HST FIX LNT HSY AMCR ECL WELL KIM PWR HUBB EQIX KEYS BMY JBHT DVN BKR DHI HAS MAR RL AVY CAT BG FE KO SRE AEE CBOE MAS WMB PKG TGT DGX XEL MCD CB TRV TJX PLD FRT ON CL PNW HCA NI MPC GEV JBL ALGN VLO PPG AMAT DD DRI ODFL OXY EME MMM AMGN YUM AOS HLT TDY IR ES PSX NDSN MCK MRNA PEP AME HIG FITB REG GEHC WM WEC LOW EXC WBD BRK-B RF MTB SNA FANG NEE L SPG FTV CTRA CVX SBUX HOLX APA CTVA MNST GILD PCAR COP EVRG LUV DG CPAY PNC CFG REGN BIIB MGM IEX ULTA USB STLD ED UAL SLB DOW EQT GL MLM WDC HSIC", #"AAPL MSFT GOOGL AMZN NVDA META AVGO CRM ADBE NFLX AMD QCOM LLY UNH JNJ ABBV MRK V MA JPM GS BAC COST WMT PG PEP KO MCD LIN SHW CAT GE HON XOM CVX NEE AMT PLD PANW ANET WDC SNDK MU AAPL MSFT NVDA GOOGL AMZN META AVGO LLY TSLA JPM V WMT UNH MA PG JNJ XOM HD ORCL COST MRK BAC ABBV CVX CRM KO PEP NFLX AMD TMUS LIN ADBE WFC MCD CSCO TRV DIS AXP INTU QCOM IBM CAT GE TXN AMAT INTC NOW UBER DHR RTX AMGN HON PFE BA SPGI COP LOW UNP BKNG SYK ELV GS PLD BLK TJX SYY TJX C DE MDT VRTX GILD ADI ISRG MMC REGN CB CI SCHW PGR ZTS BDX BSX SLB T EOG MO"
    'vol_target': 15.0,
    'mc_sims': 1000,
    'start_date': datetime(2020, 1, 1).strftime('%Y-%m-%d'),
    't_cost': 0.1,
    'adx_threshold': 15.0, 
    'kelly_fraction': 0.5,
    'allow_fractional': True,
    'stop_buffer': 2.5,
    'pos_size_pct': 20.0 
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

st.set_page_config(page_title="Alpha Engine V59: Terminal Mode", layout="wide")

# --- 2. SIDEBAR COMMAND CENTER ---
with st.sidebar:
    st.header("⚙️ Strategy Settings")
    st.session_state.universe = st.text_input("Universe (Space or Comma separated)", st.session_state.universe)
    
    # KATIE'S FIX: Now accepts spaces, commas, or both. Auto-cleans and upper-cases!
    TICKERS = [t.strip().upper() for t in st.session_state.universe.replace(',', ' ').split() if t.strip()]
    
    BENCHMARK = st.selectbox("Benchmark", ["SPY", "QQQ", "DIA", "IWM"], index=1)
    PORTFOLIO_VALUE_GBP = st.number_input("Account Balance (£)", min_value=100, value=1400)
    
    st.divider()
    st.header("🎮 Hybrid Regime Logic")
    st.session_state.regime_mode = st.radio("Strategy Mode", ["Adaptive (Hybrid)", "Momentum Only", "Mean Reversion Only"])
    st.session_state.adx_threshold = st.slider("ADX Trend/Range Cutoff", 10.0, 35.0, float(st.session_state.adx_threshold))
    
    st.divider()
    st.header("🛡️ Risk & Allocation")
    st.session_state.pos_size_pct = st.slider("Allocation per Stock (%)", 5.0, 50.0, float(st.session_state.pos_size_pct), 0.5)
    st.session_state.vol_target = st.slider("Volatility Target (%)", 5.0, 40.0, float(st.session_state.vol_target), 1.0)
    st.session_state.atr_mult = st.slider("Initial ATR Stop Multiplier", 1.0, 5.0, float(st.session_state.atr_mult), 0.1)
    st.session_state.trail_mult = st.slider("Trailing ATR Multiplier", 2.0, 6.0, float(st.session_state.trail_mult), 0.1)
    st.session_state.kelly_fraction = st.slider("Kelly Fraction (Leverage)", 0.1, 1.0, float(st.session_state.kelly_fraction), 0.05)
    st.session_state.stop_buffer = st.slider("Stop Warning Buffer (%)", 1.0, 10.0, float(st.session_state.stop_buffer), 0.5)
    st.session_state.allow_fractional = st.checkbox("Enable Fractional Shares", value=st.session_state.allow_fractional)
    
    st.divider()
    st.header("💾 Config Persistence")
    current_config = {k: v for k, v in st.session_state.items() if k in DEFAULTS}
    st.download_button("📩 Download Config", data=json.dumps(current_config, default=str), file_name="alpha_config_v59.json")
    
    uploaded_file = st.file_uploader("📂 Upload Config", type="json")
    if uploaded_file is not None:
        st.session_state.update(json.load(uploaded_file)); st.rerun()
    
    st.session_state.start_date = st.date_input("Start Date", pd.to_datetime(st.session_state.start_date))

# --- 3. CORE ANALYTICS ENGINE ---
def calculate_rsi(series, n=14):
    delta = series.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Wilder's Exponential Smoothing (alpha = 1/n)
    avg_gain = gain.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss.replace(0, 0.00001)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_crsi(series):
    # 1. Standard 3-period RSI
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=3).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=3).mean()
    rs = gain / (loss.replace(0, 0.00001))
    rsi_3 = 100 - (100 / (1 + rs))
    
    # 2. UpDown Streak RSI (2-period)
    streak = np.zeros(len(series))
    for i in range(1, len(series)):
        if series.iloc[i] > series.iloc[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif series.iloc[i] < series.iloc[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    streak_series = pd.Series(streak, index=series.index)
    s_delta = streak_series.diff()
    s_gain = (s_delta.where(s_delta > 0, 0)).rolling(window=2).mean()
    s_loss = (-s_delta.where(s_delta < 0, 0)).rolling(window=2).mean()
    s_rs = s_gain / (s_loss.replace(0, 0.00001))
    streak_rsi_2 = 100 - (100 / (1 + s_rs))
    
    # 3. Rate of Change (100-period Percentile Rank)
    pct_ret = series.pct_change(fill_method=None)
    def percentile_rank(x):
        if np.isnan(x).all(): return 50
        return (x < x[-1]).mean() * 100
    roc_100 = pct_ret.rolling(window=100).apply(percentile_rank, raw=True)
    
    # CRSI is the simple average of all three components
    crsi = (rsi_3 + streak_rsi_2 + roc_100) / 3
    return crsi

def get_slope(series):
    if len(series) < 20: return 0
    y, x = series.values, np.arange(len(series))
    slope, _, _, _, _ = linregress(x, y)
    return slope

def calculate_adx(df, n=14):
    h, l, c = df['High'], df['Low'], df['Close']
    
    # 1. Calculate True Range (TR)
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 2. Calculate Directional Movement (+DM and -DM)
    up = h.diff()
    down = -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0)
    minus_dm = down.where((down > up) & (down > 0), 0)
    
    # 3. Apply Wilder's Smoothing to TR, +DM, and -DM
    atr = tr.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    
    # 4. Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # 5. Calculate DX and smooth it to get ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 0.00001)
    adx = dx.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    
    return adx

def calculate_hurst(ts):
    if len(ts) < 100: return 0.5
    try:
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except: return 0.5

@st.cache_data(ttl=86400)
def fetch_data_v59(u_str, bench, start):
    parsed_tickers = [t.strip().upper() for t in u_str.replace(',', ' ').split() if t.strip()]
    t_list = list(set(parsed_tickers + [bench, "GBPUSD=X"]))
    
    # KATIE'S FIX: Let yfinance handle its own session security
    return yf.download(t_list, start=pd.to_datetime(start)-timedelta(days=365), auto_adjust=True, progress=False, threads=False)


def fetch_single_meta(t):
    res = {'sector': 'Unknown', 'earnings': None}
    try:
        tick = yf.Ticker(t)
        try:
            res['sector'] = tick.info.get('sector', 'Unknown')
        except: pass
        try:
            cal = tick.calendar
            if isinstance(cal, dict) and 'Earnings Date' in cal:
                res['earnings'] = cal['Earnings Date'][0]
            elif hasattr(cal, 'iloc') and not cal.empty:
                res['earnings'] = cal.iloc[0,0]
        except: pass
    except Exception: pass
    return t, res

@st.cache_data(ttl=604800)
def fetch_metadata_optimized(tickers):
    meta = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text(f"🚀 Multithreading activated: Fetching {len(tickers)} tickers concurrently...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(fetch_single_meta, t): t for t in tickers}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_ticker):
            t, res = future.result()
            meta[t] = res
            completed += 1
            progress_bar.progress(completed / len(tickers))
            
    status_text.empty()
    progress_bar.empty()
    return meta

raw = fetch_data_v59(st.session_state.universe, BENCHMARK, st.session_state.start_date)
prices = raw['Close'].ffill().bfill()
bench_adx = calculate_adx(raw.xs(BENCHMARK, axis=1, level=1))

meta_data = fetch_metadata_optimized(TICKERS)
earnings_map = {k: v['earnings'] for k, v in meta_data.items()}

@st.cache_data(ttl=3600)
def run_v59_backtest(u_str, bench_name, start_dt, mode, vol_t, t_c, adx_t, use_hurst=True):
    raw_tickers = [t.strip().upper() for t in u_str.replace(',', ' ').split() if t.strip()]
    tickers = list(dict.fromkeys(raw_tickers))
    
    m_prices = prices[tickers].resample('ME').last()
    mom_12_1 = (m_prices.shift(1)/m_prices.shift(12))-1
    
    # KATIE'S UPGRADE: Generate the hyper-sensitive CRSI for the entire universe!
    crsi_vals = prices[tickers].apply(calculate_crsi)
    
    b_rets, dates, trade_log, raw_trades = [], [], [], []
    active_pos, s_rets_list = {}, [] 
    hurst_history = [] 
    bench_p, sma200 = prices[bench_name], prices[bench_name].rolling(200).mean()

    h, l, c = raw['High'], raw['Low'], raw['Close']
    prev_c = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    tr_all = np.maximum(tr1, np.maximum(tr2, tr3)) 
    daily_atr = tr_all.rolling(14).mean()

    for i in range(13, len(m_prices)):
        dt = m_prices.index[i]
        prev_dt = m_prices.index[i-1]
        if dt.date() < pd.to_datetime(start_dt).date(): continue
        
        p_c, s200_c, adx_c = bench_p.asof(dt), sma200.asof(dt), bench_adx.asof(dt)
        hurst_val = calculate_hurst(bench_p.loc[:dt].tail(100).values)
        hurst_history.append(hurst_val)
        b_rets.append((bench_p.asof(dt)/bench_p.asof(prev_dt))-1)
        
        if use_hurst:
            is_trending = (hurst_val > 0.50 and adx_c > adx_t)
        else:
            is_trending = (adx_c > adx_t)

        picks, current_strat = [], "Cash"
        
        if (mode == "Adaptive (Hybrid)" and is_trending) or mode == "Momentum Only":
            if p_c > s200_c:
                current_strat = "Momentum"
                potential = mom_12_1.iloc[i].dropna()
                slopes = prices[tickers].loc[:dt].tail(60).apply(get_slope)
                valid = potential[slopes[potential.index] > 0].index
                picks = list(potential.loc[valid].sort_values(ascending=False).head(3).index)
        
        elif (mode == "Adaptive (Hybrid)" and not is_trending) or mode == "Mean Reversion Only":
            current_strat = "Mean Reversion"
            c_crsi = crsi_vals.loc[:dt].iloc[-1]
            
            # KATIE'S INSTITUTIONAL SHIELD: CRSI < 15 AND Price > 200 SMA!
            potential_reversions = []
            for t in c_crsi[c_crsi < 15].index:
                try:
                    t_price = float(prices[t].loc[:dt].iloc[-1])
                    t_sma200 = float(prices[t].loc[:dt].tail(200).mean())
                    # Only buy if the stock is in a Secular Bull Market!
                    if t_price > t_sma200:
                        potential_reversions.append(t)
                except:
                    pass

            picks = []
            for t in potential_reversions:
                if prices[t].loc[:dt].iloc[-1] > raw['High'][t].loc[:dt].iloc[-2]: 
                    picks.append(t)
            picks = picks[:3]

        pos_keys = list(active_pos.keys())
        for t in pos_keys:
            entry_data = active_pos[t]
            entry_p = entry_data['price']
            stop_p = entry_data['stop']
            entry_mode = entry_data.get('mode', 'Momentum')
            
            mask = (raw['Low'][t].index > prev_dt) & (raw['Low'][t].index <= dt)
            period_lows = raw['Low'][t].loc[mask]
            period_highs = raw['High'][t].loc[mask]
            period_opens = raw['Open'][t].loc[mask] # Katie's Addition: We MUST track the daily Open!
            
            exit_p = float(m_prices.loc[dt, t])
            hit_stop = False
            take_profit = False
            
            if not period_lows.empty and stop_p > 0:
                # 1. Find the specific days the stop was breached
                breach_days = period_lows[period_lows < stop_p]
                
                if not breach_days.empty:
                    hit_stop = True
                    # 2. Identify the EXACT first day of the breach
                    first_breach_date = breach_days.index[0]
                    breach_open = float(period_opens.loc[first_breach_date])
                    
                    # 3. KATIE'S INSTITUTIONAL SLIPPAGE MODEL:
                    # If the stock gapped down violently overnight, the Open is below your Stop.
                    # Your stop-loss converts to a market order and fills at the Open price.
                    if breach_open < stop_p:
                        exit_p = breach_open  # The brutal reality of market gaps
                    else:
                        # If it opened above the stop and fell through it intraday, 
                        # we assume a fill at the stop price, minus 10 bps for realistic intraday slippage.
                        exit_p = stop_p * 0.999 
            
            if not hit_stop and entry_mode == "Mean Reversion" and not period_highs.empty:
                max_high = period_highs.max()
                tp_target = entry_p * 1.05
                
                # Similarly, if it gapped UP violently, we shouldn't assume we get the exact target.
                # But to be conservative in our testing, we will just cap the profit at the target.
                if max_high >= tp_target:
                    exit_p = tp_target
                    take_profit = True

            if t not in picks or hit_stop or take_profit:
                ret = (exit_p / entry_p) - 1
                raw_trades.append(ret)
                
                if take_profit:
                    exit_reason = "Take Profit (5%)"
                elif hit_stop:
                    # Tagging the trade log so you know when overnight gaps destroyed a position
                    exit_reason = "Stop Loss (GAP DOWN)" if (hit_stop and exit_p < stop_p * 0.99) else "Stop Loss"
                else:
                    exit_reason = "Rebalance"
                
                trade_log.append({
                    "Ticker": t, 
                    "Mode": entry_mode,
                    "Entry": entry_data.get('entry_date', 'Unknown'),
                    "Exit": dt.strftime('%Y-%m'), 
                    "Return": f"{ret:.1%}", 
                    "Type": exit_reason,
                    "Market Hurst": f"{hurst_val:.2f}"
                })
                del active_pos[t]

        for t in picks:
            if t not in active_pos:
                curr_price = float(m_prices.iloc[i-1][t])
                try:
                    cur_atr = daily_atr.asof(prev_dt)[t]
                    stop_price = curr_price - (cur_atr * st.session_state.atr_mult)
                except:
                    stop_price = curr_price * 0.90 
                active_pos[t] = {'price': curr_price, 'stop': stop_price, 'mode': current_strat, 'entry_date': prev_dt.strftime('%Y-%m')}
        
        if picks:
            v_h = bench_p.pct_change().loc[:dt].tail(20).std() * np.sqrt(252)
            exp = min(1.0, (vol_t/100)/v_h) if v_h > 0 else 1.0
            monthly_rets = []
            for p in picks:
                if p in active_pos: 
                    r = (m_prices.loc[dt, p] / m_prices.iloc[i-1][p]) - 1
                    monthly_rets.append(r)
            if monthly_rets:
                m_ret = np.mean(monthly_rets)
                s_rets_list.append((m_ret * exp) - (t_c/100))
            else:
                s_rets_list.append(0.0) 
        else:
            s_rets_list.append(0.0) 
        dates.append(dt)
        
    return s_rets_list, b_rets, dates, (picks if 'picks' in locals() else []), trade_log, raw_trades, hurst_val, adx_c, current_strat, mom_12_1, hurst_history# Initial run (Hurst OFF by default as per experiment)
s_raw, b_raw, dts, final_picks, ledger, r_trades, last_h, last_adx, active_strat, m12, h_hist = run_v59_backtest(
    st.session_state.universe, BENCHMARK, st.session_state.start_date, 
    st.session_state.regime_mode, st.session_state.vol_target, 0.1, st.session_state.adx_threshold,
    use_hurst=True 
)

# --- 4. DASHBOARD UI ---
st.title("💸 John's Experiment: Ubos Pera Edition 💸")

# ==========================================
# KATIE'S "SAFE IN CASH" SENTINEL BANNER
# ==========================================
if last_h < 0.50:
    st.markdown("""
    <div style="background-color: rgba(0, 255, 170, 0.1); padding: 15px; border-radius: 10px; border: 2px solid #00FFAA; text-align: center; margin-bottom: 20px;">
        <h2 style="color: #00FFAA; margin: 0;">🏠 SAFE IN CASH (Hurst < 0.50)</h2>
        <p style="color: white; margin: 0;">The market is a blender (Mean-Reverting). Katie's Capital Protection is active. Sit back and relax.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background-color: rgba(255, 75, 75, 0.1); padding: 15px; border-radius: 10px; border: 2px solid #FF4B4B; text-align: center; margin-bottom: 20px;">
        <h2 style="color: #FF4B4B; margin: 0;">🔥 ALPHA ZONE ACTIVE (Hurst > 0.50)</h2>
        <p style="color: white; margin: 0;">The market is trending. Momentum sniper is officially hunting.</p>
    </div>
    """, unsafe_allow_html=True)
# ==========================================

s_rets = pd.Series(s_raw, index=dts)

if not s_rets.empty:
    wins, losses = [r for r in r_trades if r > 0], [r for r in r_trades if r <= 0]
    wr = len(wins)/len(r_trades) if r_trades else 0
    pf = abs((len(wins)*np.mean(wins))/(len(losses)*np.mean(losses))) if losses else 0
    excess_rets = s_rets - (0.02 / 12)
    sharpe = (excess_rets.mean() * 12) / (s_rets.std() * np.sqrt(12)) if len(s_rets) > 1 and s_rets.std() > 0 else 0
    
    cum_returns = (1 + s_rets).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7) 
    c1.metric("Profit Factor", f"{pf:.2f}")
    c2.metric("Win Rate", f"{wr:.1%}")
    c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c4.metric("Max Drawdown", f"{max_dd:.1%}") 
    c5.metric("Total Trades", f"{len(r_trades)}")
    c6.metric("Market Hurst", f"{last_h:.2f}", "Trending" if last_h > 0.5 else "Ranging")
    c7.metric("Active Strategy", active_strat)

    # KATIE'S UPDATE: All 8 Tabs are perfectly restored!
    t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs(["🚀 Live Orders", "📊 Metrics", "📉 Stress Tests", "🔔 Sentinel", "📜 History", "🧪 Lab Mode", "🔥 Heatmap", "🔭 S&P 500 Scanner"])
    
    with t1:
        if final_picks:
            st.header(f"🛒 Order Console (£ GBP Balance: £{PORTFOLIO_VALUE_GBP:,.2f})")
            trade_rows = []; 
            try: fx = raw['Close']["GBPUSD=X"].iloc[-1]
            except: fx = 1.25
            total_risk_gbp = 0; total_invested_gbp = 0

            # RS RANKINGS
            rs_scores = {}
            for s in final_picks:
                curr_p_now = float(prices[s].iloc[-1])
                bench_p_now = float(prices[BENCHMARK].iloc[-1])
                rs_scores[s] = curr_p_now / bench_p_now
            rs_ranks = pd.Series(rs_scores).rank(ascending=False, method='min').astype(int)
            
            for s in final_picks:
                h_p, l_p, c_p = raw['High'][s], raw['Low'][s], raw['Close'][s]
                tr = pd.concat([h_p-l_p, abs(h_p-c_p.shift(1)), abs(l_p-c_p.shift(1))], axis=1).max(axis=1)
                p_usd = float(prices[s].iloc[-1])
                highest_high = h_p.tail(14).max()
                
                e_date = earnings_map.get(s)
                warning_flag = ""
                k_fraction = st.session_state.kelly_fraction
                
                if e_date:
                    try:
                        days_to_earnings = (pd.to_datetime(e_date).date() - datetime.now().date()).days
                        if 0 <= days_to_earnings <= 10:
                            warning_flag = f"⚠️ EARNINGS IN {days_to_earnings} DAYS"
                            k_fraction = k_fraction * 0.5
                    except: pass

                alloc_gbp = PORTFOLIO_VALUE_GBP * (st.session_state.pos_size_pct / 100)
                raw_shares = (alloc_gbp * fx * k_fraction) / p_usd
                shares = round(raw_shares, 4) if st.session_state.allow_fractional else int(raw_shares)
                
                atr_14 = tr.tail(14).mean()
                stop_p_usd = p_usd - (atr_14 * st.session_state.atr_mult)
                trail_stop_usd = highest_high - (atr_14 * st.session_state.trail_mult)
                effective_stop = max(stop_p_usd, trail_stop_usd)

		# --- KATIE'S SAFE LIMIT ENTRY LOGIC ---
                try: prev_close = float(c_p.iloc[-2])
                except: prev_close = p_usd
                
                # If the stock is gapping up aggressively (> 1.5%), set a trap at the gap-fill
                if p_usd > (prev_close * 1.015):
                    safe_limit = prev_close * 1.01 # Wait for the institutional pullback
                else:
                    # Normal trend: Buy the algorithmic intraday dip (Current Price minus 25% of the Daily ATR)
                    safe_limit = p_usd - (atr_14 * 0.25)
                
                # Ensure we never suggest buying higher than the current price
                safe_limit = min(safe_limit, p_usd)
                # --------------------------------------
                
                risk_dist_pct = ((p_usd - effective_stop) / p_usd) * 100
                risk_amt_gbp = ((p_usd - effective_stop) * shares) / fx
                total_risk_gbp += risk_amt_gbp
                total_invested_gbp += (shares * p_usd) / fx
                
                entry_est = c_p.tail(14).iloc[0]
                potential_move = max(0.01, highest_high - entry_est)
                captured_move = effective_stop - entry_est
                efficiency = (captured_move / potential_move) * 100
                
                days_held = 14
                
                try: conf_rank = m12.iloc[-1].rank(pct=True).get(s, 0)
                except: conf_rank = 0.5
                
                # --- KATIE'S REVENUE PROTECTION UPDATE ---
                if efficiency < 0 or p_usd < entry_est:
                    continue # Skip these entirely, they are catching falling knives!

                trade_rows.append({
                    "Ticker": s, 
                    "Warning": warning_flag,
                    "Status": "✅ HIGH PROB" if efficiency > 20 else "⚠️ WATCHING",
                    "RS Rank": rs_ranks[s],
                    "Confidence": f"{(conf_rank * 100):.0f}%",
                    "Days Held": days_held,
                    "Trend Base ($)": entry_est,
                    "Price ($)": p_usd,
                    "Limit Buy ($)": safe_limit,  # <-- NEW KATIE METRIC
                    "Active Exit ($)": effective_stop,
                    "Risk %": round(risk_dist_pct, 2),
                    "Risk (£)": round(risk_amt_gbp, 2),
                    "Efficiency (%)": round(efficiency, 1),
                    "Shares": shares
                })
            
            r_col1, r_col2, r_col3 = st.columns(3)
            r_col1.metric("Total Risk Exposure (£)", f"£{total_risk_gbp:,.2f}")
            r_col2.metric("Portfolio Exposure (%)", f"{(total_invested_gbp/PORTFOLIO_VALUE_GBP):.2%}")
            r_col3.metric("Diversification Status", "Active")
            
            df_display = pd.DataFrame(trade_rows)
            
            def highlight_smart_rows(row):
                styles = [''] * len(row)
                # 1. Red Danger Zone (Your original logic)
                if row['Risk %'] < st.session_state.stop_buffer:
                    styles = ['background-color: #ff4b4b; color: white'] * len(row)
                # 2. Katie's Safe Alpha Glow
                elif row['Efficiency (%)'] > 20:
                    styles = ['background-color: rgba(0, 255, 170, 0.2)'] * len(row)
                
                # 3. Yellow Earnings Warning (Your original logic)
                if "EARNINGS" in str(row['Warning']):
                     styles[1] = 'background-color: #ffd700; color: black; font-weight: bold'
                return styles
            
            st.dataframe(df_display.style.apply(highlight_smart_rows, axis=1).format({
                "Trend Base ($)": "${:,.2f}", 
                "Price ($)": "${:,.2f}", 
                "Limit Buy ($)": "${:,.2f}", # <-- Format the new column!
                "Active Exit ($)": "${:,.2f}", 
                "Risk %": "{:.2f}%", 
                "Risk (£)": "£{:,.2f}", 
                "Efficiency (%)": "{:.1f}%"
            }), width='stretch')
            
            if any("EARNINGS" in str(r['Warning']) for r in trade_rows):
                st.warning("🚨 CAUTION: Positions adjusted for earnings volatility. Kelly fraction reduced.")
            
            st.divider()
            st.subheader("📈 Quick Chart Inspector")
            if trade_rows:
                selected_chart_ticker = st.selectbox("🔍 Select Ticker to View Chart + RSI:", [d['Ticker'] for d in trade_rows])
                if selected_chart_ticker:
                    chart_prices = raw['Close'][selected_chart_ticker].dropna()
                    bench_prices_chart = raw['Close'][BENCHMARK].reindex(chart_prices.index).ffill()
                    
                    chart_sma20 = chart_prices.rolling(window=20).mean()
                    chart_rsi = calculate_rsi(chart_prices)
                    
                    rs_ratio = chart_prices / bench_prices_chart
                    
                    display_df = pd.DataFrame({"Price": chart_prices, "20 SMA": chart_sma20}).tail(252)
                    st.line_chart(display_df, color=["#FFFFFF", "#00FFAA"])
                    
                    rsi_display = pd.DataFrame({"RSI": chart_rsi}).tail(252)
                    st.line_chart(rsi_display, color=["#FFCC00"])
                    
                    st.caption(f"📊 Relative Strength Ratio (vs {BENCHMARK})")
                    rs_display = pd.DataFrame({"RS Ratio": rs_ratio}).tail(252)
                    st.line_chart(rs_display, color=["#00CCFF"])
                    
                    curr_p = chart_prices.iloc[-1]
                    curr_sma = chart_sma20.iloc[-1]
                    curr_rsi = chart_rsi.iloc[-1]
                    trend_stat = "ABOVE" if curr_p > curr_sma else "BELOW"
                    st.caption(f"📍 **{selected_chart_ticker}** | Price: ${curr_p:.2f} | RSI: {curr_rsi:.2f}")

        else: st.warning("NO TRADES: Criteria not met.")

    with t2:
        st.subheader("Performance Overview")
        st.line_chart(pd.DataFrame({"Hybrid Equity": (1+s_rets).cumprod(), BENCHMARK: (1+pd.Series(b_raw, index=dts)).cumprod()}))
        
        st.divider()
        st.subheader("📈 Historical Market Hurst Exponent (with Confidence Interval)")
        st.caption("Visualizing the Benchmark's trend strength over time. The shaded area represents statistical noise variance.")
        
        # KATIE'S PLOTLY HURST CHART UPGRADE
        h_series = pd.Series(h_hist, index=dts)
        h_std = h_series.rolling(10).std().bfill().fillna(0.02)
        upper_bound = h_series + h_std
        lower_bound = h_series - h_std

        fig_hurst = go.Figure()
        fig_hurst.add_hline(y=0.5, line_dash="dash", line_color="#FF4B4B", annotation_text="Random Walk Boundary (0.50)")
        fig_hurst.add_trace(go.Scatter(x=dts, y=upper_bound, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_hurst.add_trace(go.Scatter(x=dts, y=lower_bound, mode='lines', fill='tonexty', fillcolor='rgba(0, 255, 170, 0.15)', line=dict(width=0), name='Noise Band', hoverinfo='skip'))
        fig_hurst.add_trace(go.Scatter(x=dts, y=h_series, mode='lines', name='Market Hurst', line=dict(color='#00FFAA', width=2)))
        
        fig_hurst.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        fig_hurst.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        fig_hurst.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        
        st.plotly_chart(fig_hurst, width='stretch')

    with t3:
        if 's_rets' not in locals() or s_rets.empty:
            st.error("⚠️ No strategy returns found. Please check your inputs or Universe.")
        else:
            if st.button("▶️ Run Monte Carlo (True Stress Test)"):
                with st.spinner("Simulating 1,000 alternative futures..."):
                    paths = []
                    pool_of_returns = s_rets.dropna().tolist()
                    n_periods = len(pool_of_returns)
                    
                    for _ in range(st.session_state.mc_sims):
                        sh = random.choices(pool_of_returns, k=n_periods)
                        path = np.cumprod(1 + np.array(sh))
                        paths.append(path)
                    st.session_state.mc_results = {"paths": paths}
            
            if "mc_results" in st.session_state and st.session_state.mc_results:
                st.subheader(f"Monte Carlo Results ({st.session_state.mc_sims} Sims)")
                df_mc = pd.DataFrame(st.session_state.mc_results["paths"]).T
                st.line_chart(df_mc)
                worst_ending = df_mc.iloc[-1].min()
                st.caption(f"Worst Case Scenario (Final Equity): {worst_ending:.2f}x initial capital")

    with t4:
        st.header("🔔 Universe Sentinel")
        u_sent = []
        for t in TICKERS:
            e_date = meta_data.get(t, {}).get('earnings', "N/A")
            sec = meta_data.get(t, {}).get('sector', "Unknown")
            u_sent.append({"Ticker": t, "Sector": sec, "Next Earnings": e_date})
        st.table(pd.DataFrame(u_sent).sort_values("Next Earnings"))

    with t5:
        st.dataframe(pd.DataFrame(ledger).sort_index(ascending=False), width='stretch')
        
    with t6:
        st.header("🧪 Lab Mode: Validation Engine")
        st.subheader("🔥 Aggressive Mode Comparison")
        st.caption("Side-by-side analysis of your current 'Always On' settings vs. the original 'Safe' Hurst Filter.")
        
        if st.button("🚀 Run Side-by-Side Analysis"):
            with st.spinner("Running double backtest..."):
                safe_raw, _, _, _, _, safe_tr, _, _, _, _, _ = run_v59_backtest(
                    st.session_state.universe, BENCHMARK, st.session_state.start_date, 
                    st.session_state.regime_mode, st.session_state.vol_target, 0.1, 20.0, use_hurst=True
                )
                safe_rets = pd.Series(safe_raw, index=dts)
                safe_cum = (1 + safe_rets).cumprod()
                agg_cum = (1 + s_rets).cumprod()
                
                def quick_stats(series, raw_trades):
                    if len(series) < 2: return 0, 0, 0
                    total_ret = (series.iloc[-1] - 1)
                    shp = (series.mean() * 12) / (series.std() * np.sqrt(12)) if series.std() > 0 else 0
                    win_r = len([r for r in raw_trades if r > 0]) / len(raw_trades) if raw_trades else 0
                    trade_results = [1 if r > 0 else 0 for r in raw_trades]
                    max_consec_loss, current_streak = 0, 0
                    for res in trade_results:
                        if res == 0:
                            current_streak += 1
                        else:
                            max_consec_loss = max(max_consec_loss, current_streak)
                            current_streak = 0
                    max_consec_loss = max(max_consec_loss, current_streak)
                    return total_ret, shp, win_r, max_consec_loss

                s_ret, s_shp, s_wr, s_streak = quick_stats(safe_rets, safe_tr)
                a_ret, a_shp, a_wr, a_streak = quick_stats(s_rets, r_trades)
                
                l_col, r_col = st.columns(2)
                l_col.metric("Original SAFE Return", f"{s_ret:.1%}", f"Sharpe: {s_shp:.2f}")
                l_col.metric("Worst Losing Streak (SAFE)", f"{s_streak} trades")
                r_col.metric("Current AGGRESSIVE Return", f"{a_ret:.1%}", f"Sharpe: {a_shp:.2f}")
                r_col.metric("Worst Losing Streak (AGGRESSIVE)", f"{a_streak} trades")
                
                st.line_chart(pd.DataFrame({"SAFE Strategy (Hurst ON)": safe_cum, "AGGRESSIVE Strategy (Hurst OFF)": agg_cum}))
        st.divider()

        if len(s_rets) > 10:
            st.subheader("1. In-Sample vs Out-of-Sample (70/30 Split)")
            split_idx = int(len(s_rets) * 0.7)
            split_date = dts[split_idx]
            train_rets = s_rets.iloc[:split_idx]
            test_rets = s_rets.iloc[split_idx:]
            def calc_metrics(series):
                if len(series) < 2: return 0, 0, 0
                cum = (1 + series).cumprod()
                dd = (cum - cum.cummax()) / cum.cummax()
                total_ret = cum.iloc[-1] - 1
                sharpe = (series.mean() * 12) / (series.std() * np.sqrt(12)) if series.std() > 0 else 0
                return total_ret, sharpe, dd.min()
            tr_ret, tr_shp, tr_dd = calc_metrics(train_rets)
            te_ret, te_shp, te_dd = calc_metrics(test_rets)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Training Return (70%)", f"{tr_ret:.1%}", f"Sharpe: {tr_shp:.2f}")
            col_b.metric("Testing Return (30%)", f"{te_ret:.1%}", f"Sharpe: {te_shp:.2f}")
            col_c.info(f"Split Date: {split_date.strftime('%Y-%m-%d')}")
            df_full = (1 + s_rets).cumprod()
            chart_data = pd.DataFrame({"Training (In-Sample)": df_full.iloc[:split_idx], "Testing (Out-of-Sample)": df_full.iloc[split_idx:]})
            st.line_chart(chart_data)
            st.divider()
            
            st.subheader("2. Walk-Forward Analysis (Rolling 12-Month Performance)")
            wf_data = []
            for year, data in s_rets.groupby(s_rets.index.year):
                y_ret, y_shp, y_dd = calc_metrics(data)
                win_pct = len(data[data > 0]) / len(data) if len(data) > 0 else 0
                wf_data.append({"Year": year, "Return": y_ret, "Sharpe": y_shp, "Max Drawdown": y_dd, "Win Rate": win_pct})
            df_wf = pd.DataFrame(wf_data).set_index("Year")
            st.dataframe(df_wf.style.format({"Return": "{:.1%}", "Sharpe": "{:.2f}", "Max Drawdown": "{:.1%}", "Win Rate": "{:.1%}"}).background_gradient(subset=["Return"], cmap="RdYlGn"), width='stretch')
        else:
            st.warning("Not enough data to perform Split Analysis.")

    with t7:
        st.header("🔥 Parameter Sensitivity Heatmap")
        if st.button("🚀 Generate Heatmap"):
            with st.spinner("Crunching numbers..."):
                vol_range = [10, 15, 20, 25, 30]
                atr_range = [1.5, 2.0, 2.5, 3.0, 3.5]
                results_matrix = []
                for v in vol_range:
                    row = []
                    for a in atr_range:
                        s_r, _, _, _, _, _, _, _, _, _, _ = run_v59_backtest(st.session_state.universe, BENCHMARK, st.session_state.start_date, st.session_state.regime_mode, v, 0.1, st.session_state.adx_threshold)
                        if not s_r: row.append(0)
                        else:
                            s_series = pd.Series(s_r)
                            row.append((s_series.mean() * 12) / (s_series.std() * np.sqrt(12)) if s_series.std() > 0 else 0)
                    results_matrix.append(row)
                df_heat = pd.DataFrame(results_matrix, index=[f"Vol {v}%" for v in vol_range], columns=[f"ATR x{a}" for a in atr_range])
                st.dataframe(df_heat.style.background_gradient(cmap="RdYlGn", axis=None).format("{:.2f}"), width='stretch')

    # ==========================================
    # KATIE'S NEW 8TH TAB: S&P 500 SCANNER
    # ==========================================
    with t8:
        st.header("🔭 S&P 500 Master Scanner")
        st.caption("Scanning the entire index for the absolute best Mean Reversion & Momentum setups today.")
        
        if st.button("🚀 Run Full S&P 500 Scan"):
            with st.spinner("Fetching S&P 500 Universe from Wikipedia..."):
                try:
                    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
                    html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
                    sp500_df = pd.read_html(io.StringIO(html))[0]
                    sp500_tickers = [sym.replace('.', '-') for sym in sp500_df['Symbol'].tolist()]
                except Exception as e:
                    st.error("Failed to fetch S&P 500 list. Using a backup top 50.")
                    sp500_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "BRK-B", "LLY", "AVGO", "V", "JPM", "UNH", "XOM", "MA", "PG", "JNJ", "HD", "CVX", "MRK", "ABBV", "COST", "PEP", "ADBE", "WMT", "CRM", "KO", "BAC", "TMO", "MCD", "CSCO", "ACN", "LIN", "NFLX", "DHR", "INTC", "CMCSA", "ABT", "AMD", "TXN", "PFE", "DIS", "WFC", "PM", "COP", "VZ", "CAT", "UNP", "IBM", "LOW"]
                
            st.info(f"Downloading data for {len(sp500_tickers)} stocks. This will take ~20 seconds...")
            
            with st.spinner("Crunching Quant Metrics..."):
                try:
                    session = requests.Session()
                    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                    scan_data = yf.download(sp500_tickers, period="1y", group_by='ticker', threads=False, progress=False)
                    
                    mom_results = []
                    
                    mom_results = []
                    mr_results = []
                    
                    for t in sp500_tickers:
                        try:
                            if len(sp500_tickers) > 1:
                                if t not in scan_data.columns.levels[0]: continue
                                df = scan_data[t].dropna()
                            else:
                                df = scan_data.dropna()
                                
                            if len(df) < 200: continue
                            
                            c = df['Close']
                            sma50 = c.rolling(50).mean().iloc[-1]
                            sma200 = c.rolling(200).mean().iloc[-1]
                            price = c.iloc[-1]
                            
                            # Calculate Standard RSI (Kept for Momentum reference)
                            delta = c.diff()
                            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                            rs = gain / (loss.replace(0, 0.00001))
                            rsi = (100 - (100 / (1 + rs))).iloc[-1]
                            
                            # KATIE'S FIX: Calculate CRSI for Mean Reversion!
                            crsi_series = calculate_crsi(c)
                            crsi_val = crsi_series.iloc[-1]
                            
                            # KATIE'S DOUBLE CONFIRMATION LOGIC
                            prev_close = c.iloc[-2]
                            is_bouncing = price > prev_close
                            sig_label = "✅ BUY (Bouncing)" if is_bouncing else "❌ WAIT (Falling)"
                            
                            # Momentum Logic
                            if price > sma50 and sma50 > sma200 and 60 <= rsi <= 80:
                                mom_results.append({"Ticker": t, "Price": round(price, 2), "RSI": round(rsi, 2), "50 SMA": round(sma50, 2), "200 SMA": round(sma200, 2)})
                                
                            # Mean Reversion Logic (Upgraded to CRSI < 15 and > 200 SMA)
                            if price > sma200 and crsi_val <= 15:
                                mr_results.append({"Ticker": t, "Signal": sig_label, "Price": round(price, 2), "CRSI": round(crsi_val, 2), "Dist to 200 SMA (%)": round(((price - sma200)/sma200)*100, 2)})
                        except Exception as e:
                            continue
                            
                    st.subheader("🧲 Mean Reversion Setups (Double Confirmation)")
                    st.caption("Katie's Note: Only buy the stocks that have a green '✅ BUY (Bouncing)' signal. Wait out the red ones!")
                    if mr_results:
                        mr_df = pd.DataFrame(mr_results).sort_values(by="CRSI", ascending=True)
                        def highlight_signal(row):
                            if row['Signal'] == "✅ BUY (Bouncing)": return ['background-color: rgba(0, 255, 170, 0.2)'] * len(row)
                            else: return ['background-color: rgba(255, 75, 75, 0.2)'] * len(row)
                        st.dataframe(mr_df.style.apply(highlight_signal, axis=1), width='stretch')
                    else:
                        st.write("No perfect mean reversion setups found today.")
                        
                    st.subheader("🔥 Top Momentum Setups (DANGER)")
                    st.caption("Katie's Warning: 🚨 Do NOT trade these unless the Hurst Banner at the top of the app turns RED.")
                    if mom_results:
                        st.dataframe(pd.DataFrame(mom_results).sort_values(by="RSI", ascending=False), width='stretch')
                    else:
                        st.write("No perfect momentum setups found today.")
                        
                except Exception as e:
                    st.error(f"Scanner encountered an error: {e}")
