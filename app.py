
import os
import json
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ============================================================
# INDEX SIGNAL DASHBOARD — PAPER TRADING ONLY
# Data source: Yahoo Finance via yfinance. Not an NSE-certified
# feed. Configure futures tickers for actual volume confirmation.
# ============================================================

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "signal_journal.db")
IST = ZoneInfo("Asia/Kolkata")

INDEXES = {
    "NIFTY": {"spot": "^NSEI", "futures_env": "NIFTY_FUTURES_TICKER"},
    "BANK NIFTY": {"spot": "^NSEBANK", "futures_env": "BANKNIFTY_FUTURES_TICKER"},
    "SENSEX": {"spot": "^BSESN", "futures_env": "SENSEX_FUTURES_TICKER"},
}

st.set_page_config(page_title="Index Signal Dashboard", page_icon="📈", layout="wide")
st.title("📈 Index Signal Dashboard")
st.caption("NIFTY 50 · BANK NIFTY · SENSEX | Multi-timeframe analysis | Paper trading only")

# -------------------------- DATABASE -------------------------

def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT, symbol TEXT, side TEXT, score INTEGER,
            entry REAL, stop REAL, target1 REAL, target2 REAL,
            status TEXT, outcome REAL, reason TEXT, strategy TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id INTEGER, opened_at TEXT, symbol TEXT, side TEXT,
            entry REAL, stop REAL, target1 REAL, target2 REAL,
            quantity INTEGER, status TEXT, exit_price REAL,
            closed_at TEXT, pnl REAL, notes TEXT, target1_hit INTEGER DEFAULT 0
        )
    """)
    # Safe migration for an existing journal database created by an earlier version.
    paper_columns = [row[1] for row in con.execute("PRAGMA table_info(paper_trades)").fetchall()]
    if "target1_hit" not in paper_columns:
        con.execute("ALTER TABLE paper_trades ADD COLUMN target1_hit INTEGER DEFAULT 0")
    con.commit()
    return con

CON = db()

def save_signal(sig):
    cur = CON.cursor()
    cur.execute("""
        INSERT INTO signals
        (created_at,symbol,side,score,entry,stop,target1,target2,status,reason,strategy)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        sig["created_at"], sig["symbol"], sig["side"], sig["score"],
        sig["entry"], sig["stop"], sig["target1"], sig["target2"],
        "GENERATED", sig["reason"], sig["strategy"]
    ))
    CON.commit()
    return cur.lastrowid

def save_paper_trade(signal_id, sig, qty):
    CON.execute("""
        INSERT INTO paper_trades
        (signal_id,opened_at,symbol,side,entry,stop,target1,target2,quantity,status,notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        signal_id, sig["created_at"], sig["symbol"], sig["side"],
        sig["entry"], sig["stop"], sig["target1"], sig["target2"],
        qty, "OPEN", sig["reason"]
    ))
    CON.commit()

# -------------------------- DATA ------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def fetch_ohlcv(ticker, interval, period):
    try:
        df = yf.download(
            ticker, period=period, interval=interval,
            auto_adjust=False, progress=False, threads=False
        )
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
        needed = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in needed):
            return pd.DataFrame()
        if "volume" not in df.columns:
            df["volume"] = np.nan
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df = df.dropna(subset=["open", "high", "low", "close"])
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(IST)
        return df
    except Exception:
        return pd.DataFrame()

def get_market_data(name, use_futures_volume=True):
    spot_ticker = INDEXES[name]["spot"]
    spot5 = fetch_ohlcv(spot_ticker, "5m", "5d")
    spot15 = fetch_ohlcv(spot_ticker, "15m", "1mo")
    spot60 = fetch_ohlcv(spot_ticker, "60m", "3mo")
    daily = fetch_ohlcv(spot_ticker, "1d", "3mo")
    futures_ticker = os.getenv(INDEXES[name]["futures_env"], "").strip()
    fut5 = fetch_ohlcv(futures_ticker, "5m", "5d") if use_futures_volume and futures_ticker else pd.DataFrame()
    return {"5m": spot5, "15m": spot15, "1h": spot60, "daily": daily, "futures": fut5,
            "futures_ticker": futures_ticker}

# ---------------------- INDICATORS ---------------------------

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    delta = s.diff()
    up = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def add_indicators(df):
    if df.empty:
        return df
    x = df.copy()
    for n in [9, 20, 50]:
        x[f"ema{n}"] = ema(x["close"], n)
    x["rsi"] = rsi(x["close"])
    x["atr"] = atr(x)
    x["vol_sma20"] = x["volume"].rolling(20).mean()
    x["rvol"] = x["volume"] / x["vol_sma20"].replace(0, np.nan)
    typical = (x["high"] + x["low"] + x["close"]) / 3
    # Intraday VWAP, reset each local trading day.
    day = pd.Series(x.index.date, index=x.index)
    pv = typical * x["volume"]
    x["vwap"] = pv.groupby(day).cumsum() / x["volume"].groupby(day).cumsum().replace(0, np.nan)
    return x

def pivots(daily):
    if len(daily) < 2:
        return {}
    prev = daily.iloc[-2]
    h, l, c = float(prev.high), float(prev.low), float(prev.close)
    p = (h + l + c) / 3
    return {
        "PDH": h, "PDL": l, "PDC": c, "P": p,
        "R1": 2*p-l, "S1": 2*p-h,
        "R2": p+(h-l), "S2": p-(h-l),
        "R3": h+2*(p-l), "S3": l-2*(h-p),
    }

def cpr_levels(daily):
    if len(daily) < 2:
        return {}
    d = daily.iloc[-2]
    p = (float(d.high) + float(d.low) + float(d.close)) / 3
    bc = (float(d.high) + float(d.low)) / 2
    tc = 2*p-bc
    return {"CPR_BOTTOM": min(bc,tc), "CPR_PIVOT": p, "CPR_TOP": max(bc,tc)}

# ---------------------- STRUCTURE / PA ------------------------

def swing_points(df, left=2, right=2):
    x = df.copy()
    x["swing_high"] = False
    x["swing_low"] = False
    if len(x) < left + right + 3:
        return x
    highs, lows = x["high"].values, x["low"].values
    sh, sl = np.zeros(len(x), dtype=bool), np.zeros(len(x), dtype=bool)
    for i in range(left, len(x)-right):
        sh[i] = highs[i] == np.max(highs[i-left:i+right+1])
        sl[i] = lows[i] == np.min(lows[i-left:i+right+1])
    x["swing_high"], x["swing_low"] = sh, sl
    return x

def market_structure(df):
    x = swing_points(df).dropna(subset=["close"])
    highs = x.loc[x.swing_high, "high"].tail(2)
    lows = x.loc[x.swing_low, "low"].tail(2)
    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs.iloc[-1] > highs.iloc[-2]
        hl = lows.iloc[-1] > lows.iloc[-2]
        lh = highs.iloc[-1] < highs.iloc[-2]
        ll = lows.iloc[-1] < lows.iloc[-2]
        if hh and hl: trend = "BULLISH"
        elif lh and ll: trend = "BEARISH"
        else: trend = "MIXED"
    else:
        trend = "INSUFFICIENT DATA"
    last = x.iloc[-1]
    prior_highs = x.loc[x.swing_high, "high"]
    prior_lows = x.loc[x.swing_low, "low"]
    last_sh = float(prior_highs.iloc[-1]) if len(prior_highs) else np.nan
    last_sl = float(prior_lows.iloc[-1]) if len(prior_lows) else np.nan
    bos_up = pd.notna(last_sh) and float(last.close) > last_sh
    bos_down = pd.notna(last_sl) and float(last.close) < last_sl
    return {
        "trend": trend, "last_swing_high": last_sh, "last_swing_low": last_sl,
        "bos_up": bool(bos_up), "bos_down": bool(bos_down),
        "structure_frame": x
    }

def candle_patterns(df):
    if len(df) < 2:
        return {"bull_engulf": False, "bear_engulf": False, "bull_reject": False, "bear_reject": False}
    a, b = df.iloc[-2], df.iloc[-1]
    body = abs(float(b.close)-float(b.open))
    rng = max(float(b.high)-float(b.low), 1e-9)
    lower = min(float(b.open),float(b.close))-float(b.low)
    upper = float(b.high)-max(float(b.open),float(b.close))
    bull_engulf = (a.close < a.open and b.close > b.open and b.close >= a.open and b.open <= a.close)
    bear_engulf = (a.close > a.open and b.close < b.open and b.open >= a.close and b.close <= a.open)
    bull_reject = lower/rng >= 0.45 and b.close > b.open and body/rng >= 0.15
    bear_reject = upper/rng >= 0.45 and b.close < b.open and body/rng >= 0.15
    return {"bull_engulf": bool(bull_engulf), "bear_engulf": bool(bear_engulf),
            "bull_reject": bool(bull_reject), "bear_reject": bool(bear_reject)}

def detect_liquidity_sweep(df, lookback=20):
    if len(df) < lookback+2:
        return False, False
    prev = df.iloc[-lookback-1:-1]
    cur = df.iloc[-1]
    buy_side_sweep = cur.high > prev.high.max() and cur.close < prev.high.max()
    sell_side_sweep = cur.low < prev.low.min() and cur.close > prev.low.min()
    return bool(sell_side_sweep), bool(buy_side_sweep)

def recent_zones(df, lookback=80):
    x = df.tail(lookback)
    if x.empty:
        return [], []
    sw = swing_points(x)
    supports = sw.loc[sw.swing_low, "low"].tail(3).astype(float).tolist()
    resistances = sw.loc[sw.swing_high, "high"].tail(3).astype(float).tolist()
    return supports, resistances


def fibonacci_retracement_1h(df, trend):
    """
    Uses the latest confirmed 1H swing pair.
    Bullish impulse: swing low -> subsequent swing high.
    Bearish impulse: swing high -> subsequent swing low.
    Only 50% and 61.8% retracement levels are used.
    """
    if len(df) < 15:
        return {"direction": "UNAVAILABLE", "fib50": np.nan, "fib618": np.nan,
                "zone_low": np.nan, "zone_high": np.nan, "anchor_low": np.nan,
                "anchor_high": np.nan}

    sw = swing_points(df.tail(180), left=2, right=2)
    highs = [(idx, float(row.high)) for idx, row in sw[sw.swing_high].iterrows()]
    lows = [(idx, float(row.low)) for idx, row in sw[sw.swing_low].iterrows()]

    if trend == "BULLISH":
        pairs = [(lo, hi) for lo in lows for hi in highs if lo[0] < hi[0]]
        if not pairs:
            return {"direction": "UNAVAILABLE", "fib50": np.nan, "fib618": np.nan,
                    "zone_low": np.nan, "zone_high": np.nan, "anchor_low": np.nan,
                    "anchor_high": np.nan}
        low_anchor, high_anchor = max(pairs, key=lambda pair: pair[1][0])
        lo, hi = low_anchor[1], high_anchor[1]
        if hi <= lo:
            return {"direction": "UNAVAILABLE", "fib50": np.nan, "fib618": np.nan,
                    "zone_low": np.nan, "zone_high": np.nan, "anchor_low": lo,
                    "anchor_high": hi}
        f50 = hi - (hi-lo)*0.50
        f618 = hi - (hi-lo)*0.618
        direction = "BULLISH IMPULSE"
    elif trend == "BEARISH":
        pairs = [(hi, lo) for hi in highs for lo in lows if hi[0] < lo[0]]
        if not pairs:
            return {"direction": "UNAVAILABLE", "fib50": np.nan, "fib618": np.nan,
                    "zone_low": np.nan, "zone_high": np.nan, "anchor_low": np.nan,
                    "anchor_high": np.nan}
        high_anchor, low_anchor = max(pairs, key=lambda pair: pair[1][0])
        hi, lo = high_anchor[1], low_anchor[1]
        if hi <= lo:
            return {"direction": "UNAVAILABLE", "fib50": np.nan, "fib618": np.nan,
                    "zone_low": np.nan, "zone_high": np.nan, "anchor_low": lo,
                    "anchor_high": hi}
        f50 = lo + (hi-lo)*0.50
        f618 = lo + (hi-lo)*0.618
        direction = "BEARISH IMPULSE"
    else:
        return {"direction": "WAITING FOR 1H TREND", "fib50": np.nan, "fib618": np.nan,
                "zone_low": np.nan, "zone_high": np.nan, "anchor_low": np.nan,
                "anchor_high": np.nan}

    return {
        "direction": direction, "fib50": float(f50), "fib618": float(f618),
        "zone_low": float(min(f50, f618)), "zone_high": float(max(f50, f618)),
        "anchor_low": float(lo), "anchor_high": float(hi)
    }

def daily_trend_context(daily_df):
    """Higher-timeframe context from daily EMA alignment and confirmed swing structure."""
    if daily_df.empty or len(daily_df) < 25:
        return {"trend": "INSUFFICIENT DATA", "ema_bull": False, "ema_bear": False,
                "structure": "INSUFFICIENT DATA", "fib": fibonacci_retracement_1h(pd.DataFrame(), "MIXED")}
    x = add_indicators(daily_df)
    last = x.iloc[-1]
    ema_bull = bool(last.close > last.ema20 and last.ema20 > last.ema50)
    ema_bear = bool(last.close < last.ema20 and last.ema20 < last.ema50)
    structure = market_structure(x)["trend"]
    if ema_bull and structure == "BULLISH":
        trend = "BULLISH"
    elif ema_bear and structure == "BEARISH":
        trend = "BEARISH"
    elif ema_bull:
        trend = "BULLISH (EMA)"
    elif ema_bear:
        trend = "BEARISH (EMA)"
    else:
        trend = "SIDEWAYS / MIXED"
    fib_trend = "BULLISH" if ema_bull else "BEARISH" if ema_bear else structure
    fib = fibonacci_retracement_1h(x, fib_trend)
    return {
        "trend": trend, "ema_bull": ema_bull, "ema_bear": ema_bear,
        "structure": structure, "fib": fib,
        "close": safe_num(last.close), "ema20": safe_num(last.ema20),
        "ema50": safe_num(last.ema50), "last_time": str(x.index[-1])
    }


def fvg_status(df):
    if len(df) < 3:
        return "Not enough data"
    a, c = df.iloc[-3], df.iloc[-1]
    if c.low > a.high:
        return "Bullish FVG"
    if c.high < a.low:
        return "Bearish FVG"
    return "No fresh FVG"

def order_block_hint(df):
    if len(df) < 5:
        return "Not enough data"
    x = df.tail(5)
    last = x.iloc[-1]
    prev = x.iloc[-2]
    if last.close > x.high.iloc[:-1].max() and prev.close < prev.open:
        return "Possible bullish order-block origin"
    if last.close < x.low.iloc[:-1].min() and prev.close > prev.open:
        return "Possible bearish order-block origin"
    return "No clear order-block displacement"

# ----------------------- SIGNAL ENGINE -----------------------

def safe_num(v):
    try:
        return float(v) if pd.notna(v) else np.nan
    except Exception:
        return np.nan

def analyze_symbol(symbol, data, volume_required=True):
    d5, d15, d1h, daily = data["5m"], data["15m"], data["1h"], data["daily"]
    if min(len(d5), len(d15), len(d1h), len(daily)) < 25:
        return {"side":"NO TRADE", "score":0, "reason":["Insufficient OHLC history."],
                "warnings":["Check data source, market hours and ticker availability."]}
    x5, x15, x1h = add_indicators(d5), add_indicators(d15), add_indicators(d1h)
    m5, m15, m1h = market_structure(x5), market_structure(x15), market_structure(x1h)
    c5, c15, c1h = x5.iloc[-1], x15.iloc[-1], x1h.iloc[-1]
    daily_context = daily_trend_context(daily)
    daily_fib = daily_context.get("fib", {})
    daily_fib_tol = max((safe_num(daily_context.get("close", np.nan)) or 0) * 0.001, 1)
    daily_fib_touch = (
        pd.notna(daily_fib.get("zone_low", np.nan))
        and pd.notna(daily_fib.get("zone_high", np.nan))
        and float(c15.close) >= daily_fib["zone_low"] - daily_fib_tol
        and float(c15.close) <= daily_fib["zone_high"] + daily_fib_tol
    )
    pat = candle_patterns(x5)
    bull_sweep, bear_sweep = detect_liquidity_sweep(x5)
    levels = pivots(daily)
    cpr = cpr_levels(daily)
    supports, resistances = recent_zones(x15)
    price = float(c5.close)

    # Volume is valid only when real corresponding futures volume is configured.
    fut = data.get("futures", pd.DataFrame())
    volume_source = "Futures volume" if not fut.empty else "Unavailable"
    vol_ok = False
    rvol = np.nan
    if not fut.empty and len(fut) >= 21:
        fv = fut.copy()
        fv["vol_sma20"] = fv.volume.rolling(20).mean()
        fv["rvol"] = fv.volume / fv.vol_sma20.replace(0,np.nan)
        rvol = safe_num(fv.rvol.iloc[-1])
        vol_ok = pd.notna(rvol) and rvol >= 1.2

    h1_bull = c1h.close > c1h.ema20 and c1h.ema20 > c1h.ema50
    h1_bear = c1h.close < c1h.ema20 and c1h.ema20 < c1h.ema50
    m15_bull = c15.close > c15.ema9 and c15.ema9 > c15.ema20 and c15.rsi >= 50
    m15_bear = c15.close < c15.ema9 and c15.ema9 < c15.ema20 and c15.rsi <= 50
    m5_bull = c5.close > c5.ema9 and c5.ema9 > c5.ema20
    m5_bear = c5.close < c5.ema9 and c5.ema9 < c5.ema20
    vwap_bull = pd.notna(c5.vwap) and c5.close > c5.vwap
    vwap_bear = pd.notna(c5.vwap) and c5.close < c5.vwap
    bull_trigger = pat["bull_engulf"] or pat["bull_reject"] or m5["bos_up"] or bull_sweep
    bear_trigger = pat["bear_engulf"] or pat["bear_reject"] or m5["bos_down"] or bear_sweep

    # Dynamic level proximity, calculated before scoring and reversal classification.
    all_levels = {**levels, **cpr}
    nearest_support = max(
        [v for v in supports + [levels.get("PDL",np.nan), levels.get("S1",np.nan),
                                cpr.get("CPR_BOTTOM",np.nan)] if pd.notna(v) and v <= price],
        default=np.nan
    )
    nearest_res = min(
        [v for v in resistances + [levels.get("PDH",np.nan), levels.get("R1",np.nan),
                                   cpr.get("CPR_TOP",np.nan)] if pd.notna(v) and v >= price],
        default=np.nan
    )
    level_tolerance = max((safe_num(c5.atr) if pd.notna(safe_num(c5.atr)) else 0) * 0.35, 1)
    near_support = pd.notna(nearest_support) and (price-nearest_support) <= level_tolerance
    near_res = pd.notna(nearest_res) and (nearest_res-price) <= level_tolerance

    bull_points = 0
    bear_points = 0
    bull_reasons, bear_reasons = [], []

    # 1H trend and structure: 25 points
    if h1_bull:
        bull_points += 25
        bull_reasons.append("1H trend is bullish: price > EMA20 > EMA50.")
    if h1_bear:
        bear_points += 25
        bear_reasons.append("1H trend is bearish: price < EMA20 < EMA50.")

    # 15m setup and structure: 20 points
    if m15_bull:
        bull_points += 10
        bull_reasons.append("15m EMA momentum and RSI support the bullish setup.")
    if m15_bear:
        bear_points += 10
        bear_reasons.append("15m EMA momentum and RSI support the bearish setup.")
    if m15["trend"] == "BULLISH":
        bull_points += 10
        bull_reasons.append("15m structure shows higher highs and higher lows.")
    if m15["trend"] == "BEARISH":
        bear_points += 10
        bear_reasons.append("15m structure shows lower highs and lower lows.")

    # 5m price-action entry: 20 points
    if m5_bull:
        bull_points += 5
        bull_reasons.append("5m EMA momentum is bullish.")
    if m5_bear:
        bear_points += 5
        bear_reasons.append("5m EMA momentum is bearish.")
    if bull_trigger:
        bull_points += 15
        bull_reasons.append("5m bullish trigger detected: BOS, rejection, engulfing or sell-side liquidity sweep.")
    if bear_trigger:
        bear_points += 15
        bear_reasons.append("5m bearish trigger detected: BOS, rejection, engulfing or buy-side liquidity sweep.")

    # Support/resistance confluence: 10 points
    sr_bull = near_support and (bull_sweep or pat["bull_reject"] or pat["bull_engulf"])
    sr_bear = near_res and (bear_sweep or pat["bear_reject"] or pat["bear_engulf"])
    if sr_bull:
        bull_points += 10
        bull_reasons.append("Bullish rejection/sweep is occurring near a support zone.")
    if sr_bear:
        bear_points += 10
        bear_reasons.append("Bearish rejection/sweep is occurring near a resistance zone.")

    # Volume: 10 points, only from configured futures volume.
    if vol_ok:
        if c5.close >= c5.open:
            bull_points += 10
            bull_reasons.append(f"Futures RVOL is {rvol:.2f}x its 20-bar average.")
        else:
            bear_points += 10
            bear_reasons.append(f"Futures RVOL is {rvol:.2f}x its 20-bar average.")

    # VWAP + RSI confirmation: 5 points
    if vwap_bull and safe_num(c15.rsi) >= 50:
        bull_points += 5
        bull_reasons.append("Price is above VWAP and 15m RSI is at/above 50.")
    if vwap_bear and safe_num(c15.rsi) <= 50:
        bear_points += 5
        bear_reasons.append("Price is below VWAP and 15m RSI is at/below 50.")

    # Fibonacci 50% / 61.8% confluence: 10 points.
    fib = fibonacci_retracement_1h(x1h, m1h["trend"])
    atr15_for_fib = safe_num(c15.atr)
    fib_tol = max((atr15_for_fib if pd.notna(atr15_for_fib) else 0) * 0.25, price * 0.0003)
    fib_touch = (
        pd.notna(fib["zone_low"]) and pd.notna(fib["zone_high"])
        and (float(c15.close) >= fib["zone_low"] - fib_tol)
        and (float(c15.close) <= fib["zone_high"] + fib_tol)
    )
    if fib_touch and fib["direction"] == "BULLISH IMPULSE":
        bull_points += 10
        bull_reasons.append(
            f"15m price is testing the 1H Fibonacci 50%-61.8% pullback zone "
            f"({fib['zone_low']:.2f}–{fib['zone_high']:.2f})."
        )
    elif fib_touch and fib["direction"] == "BEARISH IMPULSE":
        bear_points += 10
        bear_reasons.append(
            f"15m price is testing the 1H Fibonacci 50%-61.8% pullback zone "
            f"({fib['zone_low']:.2f}–{fib['zone_high']:.2f})."
        )

    warnings = []
    # Daily trend is higher-timeframe context, not an extra score bucket.
    # Keep the 100-point intraday score unchanged and clearly flag counter-trend setups.
    if daily_context.get("trend", "").startswith("BULLISH"):
        if h1_bull:
            bull_reasons.append("Daily trend also supports the bullish 1H direction.")
        if h1_bear:
            warnings.append("Counter-trend caution: 1H setup is bearish while the daily trend is bullish.")
    elif daily_context.get("trend", "").startswith("BEARISH"):
        if h1_bear:
            bear_reasons.append("Daily trend also supports the bearish 1H direction.")
        if h1_bull:
            warnings.append("Counter-trend caution: 1H setup is bullish while the daily trend is bearish.")
    if daily_fib_touch:
        warnings.append("Price is also testing the DAILY 50%-61.8% Fibonacci retracement zone; check daily support/resistance before entry.")
    if fut.empty:
        warnings.append("No futures ticker configured: actual traded volume confirmation is unavailable. Signals are downgraded.")
    elif pd.isna(rvol):
        warnings.append("Futures volume history is insufficient to calculate RVOL.")
    if m1h["trend"] == "MIXED":
        warnings.append("1H structure is mixed; avoid aggressive trend-following entries.")
    if m15["trend"] == "MIXED":
        warnings.append("15m structure is mixed/choppy.")
    if not h1_bull and not h1_bear:
        warnings.append("1H EMA trend filter is not aligned.")
    if not m15_bull and not m15_bear:
        warnings.append("15m setup confirmation is missing.")

    # Conservative final gate: 1H bias + 15m alignment + 5m trigger.
    side, score, reasons = "NO TRADE", max(bull_points,bear_points), []
    if h1_bull and m15_bull and bull_trigger and m5_bull and bull_points >= 65:
        if volume_required and not vol_ok:
            side = "REVERSAL WATCH" if bull_sweep or pat["bull_reject"] else "NO TRADE"
            reasons = bull_reasons + ["Volume confirmation is not available/confirmed; no confirmed BUY entry."]
        else:
            side, score, reasons = "BUY", bull_points, bull_reasons
    elif h1_bear and m15_bear and bear_trigger and m5_bear and bear_points >= 65:
        if volume_required and not vol_ok:
            side = "REVERSAL WATCH" if bear_sweep or pat["bear_reject"] else "NO TRADE"
            reasons = bear_reasons + ["Volume confirmation is not available/confirmed; no confirmed SELL entry."]
        else:
            side, score, reasons = "SELL", bear_points, bear_reasons
    elif (bull_sweep or pat["bull_reject"]) and near_support:
        side, reasons = "REVERSAL WATCH", ["Price is testing a nearby support zone and 5m bullish rejection/sweep is present.", "Wait for a 5m bullish CHoCH/BOS and 15m confirmation."]
    elif (bear_sweep or pat["bear_reject"]) and near_res:
        side, reasons = "REVERSAL WATCH", ["Price is testing a nearby resistance zone and 5m bearish rejection/sweep is present.", "Wait for a 5m bearish CHoCH/BOS and 15m confirmation."]
    else:
        reasons = ["No complete 1H → 15m → 5m alignment.", "Wait for structure confirmation and a valid risk/reward setup."]

    # Entry/SL/targets based on recent swing and ATR buffer.
    entry = price
    atrv = safe_num(c5.atr)
    swing_low, swing_high = m5["last_swing_low"], m5["last_swing_high"]
    stop = t1 = t2 = np.nan
    if side == "BUY" and pd.notna(atrv):
        candidate = swing_low if pd.notna(swing_low) else price-atrv
        stop = min(candidate, price-0.15*atrv) - 0.10*atrv
        risk = price-stop
        t1, t2 = price+2*risk, price+3*risk
        if pd.notna(nearest_res) and nearest_res > price and nearest_res < t1:
            t1 = nearest_res
        if t1 <= price or (t1-price)/risk < 2:
            side = "NO TRADE"; reasons.append("Nearest resistance does not allow minimum 1:2 reward/risk.")
    elif side == "SELL" and pd.notna(atrv):
        candidate = swing_high if pd.notna(swing_high) else price+atrv
        stop = max(candidate, price+0.15*atrv) + 0.10*atrv
        risk = stop-price
        t1, t2 = price-2*risk, price-3*risk
        if pd.notna(nearest_support) and nearest_support < price and nearest_support > t1:
            t1 = nearest_support
        if t1 >= price or (price-t1)/risk < 2:
            side = "NO TRADE"; reasons.append("Nearest support does not allow minimum 1:2 reward/risk.")

    if side in ["NO TRADE","REVERSAL WATCH"]:
        entry = stop = t1 = t2 = np.nan

    return {
        "symbol":symbol, "side":side, "score":int(min(score,100)), "price":price,
        "entry":entry, "stop":stop, "target1":t1, "target2":t2,
        "reasons":reasons, "warnings":warnings, "levels":all_levels,
        "supports":supports, "resistances":resistances,
        "trend1h":m1h["trend"], "trend15m":m15["trend"],
        "bos5_up":m5["bos_up"], "bos5_down":m5["bos_down"],
        "rsi15":safe_num(c15.rsi), "rvol":rvol, "volume_source":volume_source,
        "vwap":safe_num(c5.vwap), "fib":fib, "fib_touch":bool(fib_touch),
        "daily_context":daily_context, "daily_fib_touch":bool(daily_fib_touch),
        "fvg":fvg_status(x15),
        "order_block":order_block_hint(x15), "last_time":str(x5.index[-1]),
        "data":data, "strategy":"MTF price action + structure"
    }

# ----------------------- PAPER TRADING ------------------------

def open_trade_from_signal(sig, qty, signal_id):
    if signal_id is None:
        return
    save_paper_trade(signal_id, {
        **sig, "created_at":datetime.now(IST).isoformat()
    }, qty)

def update_open_trades(latest_prices, trailing_levels):
    rows = CON.execute(
        "SELECT id,symbol,side,entry,stop,target1,target2,quantity,target1_hit "
        "FROM paper_trades WHERE status='OPEN'"
    ).fetchall()
    for tid,symbol,side,entry,stop,t1,t2,qty,t1_hit in rows:
        px = latest_prices.get(symbol)
        if px is None or pd.isna(px):
            continue
        trail = trailing_levels.get(symbol, np.nan)
        exit_reason, exit_px = None, None
        if side == "BUY":
            if px <= stop:
                exit_reason, exit_px = ("BREAKEVEN / TRAIL STOP" if t1_hit else "STOP LOSS"), stop
            elif px >= t2:
                exit_reason, exit_px = "TARGET 2", t2
            elif px >= t1 and not t1_hit:
                new_stop = max(float(entry), float(trail)) if pd.notna(trail) else float(entry)
                CON.execute("UPDATE paper_trades SET target1_hit=1,stop=? WHERE id=?", (new_stop,tid))
            elif t1_hit and pd.notna(trail) and float(trail) > float(stop) and float(trail) < float(px):
                CON.execute("UPDATE paper_trades SET stop=? WHERE id=?", (float(trail),tid))
        else:
            if px >= stop:
                exit_reason, exit_px = ("BREAKEVEN / TRAIL STOP" if t1_hit else "STOP LOSS"), stop
            elif px <= t2:
                exit_reason, exit_px = "TARGET 2", t2
            elif px <= t1 and not t1_hit:
                new_stop = min(float(entry), float(trail)) if pd.notna(trail) else float(entry)
                CON.execute("UPDATE paper_trades SET target1_hit=1,stop=? WHERE id=?", (new_stop,tid))
            elif t1_hit and pd.notna(trail) and float(trail) < float(stop) and float(trail) > float(px):
                CON.execute("UPDATE paper_trades SET stop=? WHERE id=?", (float(trail),tid))
        if exit_reason:
            pnl = (exit_px-entry)*qty if side=="BUY" else (entry-exit_px)*qty
            CON.execute(
                "UPDATE paper_trades SET status=?,exit_price=?,closed_at=?,pnl=? WHERE id=?",
                (exit_reason,exit_px,datetime.now(IST).isoformat(),pnl,tid)
            )
    CON.commit()

# -------------------------- UI -------------------------------

with st.sidebar:
    st.header("Settings")
    st.caption("Paper trading only. No broker connection.")
    volume_required = st.checkbox("Require futures volume for confirmed signals", value=True)
    use_futures = st.checkbox("Load configured futures volume tickers", value=True)
    qty = st.number_input("Paper quantity (index units)", min_value=1, value=1, step=1)
    refresh = st.selectbox("Refresh interval", [30, 60, 120, 300], index=1, format_func=lambda x:f"{x} seconds")
    st.info("Set futures ticker environment variables in deployment secrets. Yahoo Finance symbols and data availability may change.")

st_autorefresh = None
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=refresh*1000, key="market_refresh")
except Exception:
    st.caption("Auto-refresh package unavailable; use the refresh button.")
if st.button("🔄 Refresh market data"):
    st.cache_data.clear()
    st.rerun()

st.warning("Data source: Yahoo Finance (yfinance), not an official exchange feed. Verify prices with your broker before relying on any analysis. Signals are educational and paper-trading only.")

signals = {}
market_data = {}
for name in INDEXES:
    with st.spinner(f"Loading {name}..."):
        md = get_market_data(name, use_futures)
        market_data[name] = md
        signals[name] = analyze_symbol(name, md, volume_required)

# Journal every distinct 5m candle's generated status once per app session.
if "journaled_candles" not in st.session_state:
    st.session_state.journaled_candles = {}
if "signal_ids" not in st.session_state:
    st.session_state.signal_ids = {}
for _name, _sig in signals.items():
    _candle_key = f"{_name}|{_sig.get('last_time')}|{_sig.get('side')}"
    if _sig.get("last_time") and st.session_state.journaled_candles.get(_name) != _candle_key:
        _sid = save_signal({
            "created_at": datetime.now(IST).isoformat(), "symbol":_name,
            "side":_sig["side"], "score":_sig["score"], "entry":_sig.get("entry"),
            "stop":_sig.get("stop"), "target1":_sig.get("target1"), "target2":_sig.get("target2"),
            "reason":"; ".join(_sig.get("reasons",[])), "strategy":_sig.get("strategy","MTF")
        })
        st.session_state.signal_ids[_name] = _sid
        st.session_state.journaled_candles[_name] = _candle_key

_trailing_levels = {}
for _name, _sig in signals.items():
    _frame = _sig.get("data", {}).get("5m", pd.DataFrame())
    _ms = market_structure(add_indicators(_frame)) if not _frame.empty else {}
    _field = "last_swing_low" if _sig["side"] == "BUY" else "last_swing_high"
    _trailing_levels[_name] = _ms.get(_field, np.nan)
update_open_trades({k:v.get("price",np.nan) for k,v in signals.items()}, _trailing_levels)

tabs = st.tabs(["Live Dashboard", "Levels & Structure", "Paper Trading", "Signal Journal", "Backtesting", "Data & Setup"])

with tabs[0]:
    st.subheader("Market overview")
    cols = st.columns(3)
    for col, name in zip(cols, INDEXES):
        sig = signals[name]
        with col:
            st.metric(name, f"{sig.get('price',np.nan):,.2f}" if pd.notna(sig.get("price",np.nan)) else "Unavailable")
            side = sig["side"]
            if side == "BUY": st.success(f"BUY · {sig['score']}/100")
            elif side == "SELL": st.error(f"SELL · {sig['score']}/100")
            elif side == "REVERSAL WATCH": st.warning(f"REVERSAL WATCH · {sig['score']}/100")
            else: st.info(f"NO TRADE · {sig['score']}/100")
            st.caption(f"1H: {sig.get('trend1h','—')} | 15m: {sig.get('trend15m','—')}")
            st.caption(f"Last candle: {sig.get('last_time','—')}")
            if sig["side"] in ["BUY","SELL"]:
                st.write("**Trade plan**")
                st.write(f"Entry: {sig['entry']:,.2f}")
                st.write(f"SL: {sig['stop']:,.2f}")
                st.write(f"T1: {sig['target1']:,.2f}")
                st.write(f"T2: {sig['target2']:,.2f}")
                if st.button(f"Open paper {name}", key=f"paper_{name}"):
                    open_trade_from_signal(sig, qty, st.session_state.signal_ids.get(name))
                    st.success("Paper trade recorded.")
            st.write("**Why this signal?**")
            for reason in sig.get("reasons",[]): st.write(f"• {reason}")
            for warning in sig.get("warnings",[]): st.caption(f"⚠️ {warning}")

    st.divider()
    st.subheader("Signal status")
    summary = pd.DataFrame([{
        "Index":n, "Signal":signals[n]["side"], "Score":signals[n]["score"],
        "Last price":signals[n].get("price"), "RSI 15m":signals[n].get("rsi15"),
        "Futures RVOL":signals[n].get("rvol"), "VWAP 5m":signals[n].get("vwap"),
        "Fib 1H":("IN ZONE" if signals[n].get("fib_touch") else "Outside / unavailable"),
        "Daily trend":signals[n].get("daily_context",{}).get("trend","—"),
        "Fib Daily":("IN ZONE" if signals[n].get("daily_fib_touch") else "Outside / unavailable"),
        "Last candle":signals[n].get("last_time")
    } for n in INDEXES])
    st.dataframe(summary, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Previous day levels, pivots and market structure")
    selected = st.selectbox("Select index", list(INDEXES), key="levels_symbol")
    sig = signals[selected]
    st.metric("Current spot price", f"{sig.get('price',np.nan):,.2f}" if pd.notna(sig.get("price",np.nan)) else "Unavailable")
    lev = sig.get("levels",{})
    if lev:
        st.dataframe(pd.DataFrame([{"Level":k,"Price":v} for k,v in lev.items()]), use_container_width=True, hide_index=True)
    st.write("**Recent 15m support zones**", [round(v,2) for v in sig.get("supports",[])])
    st.write("**Recent 15m resistance zones**", [round(v,2) for v in sig.get("resistances",[])])
    st.write(f"**1H structure:** {sig.get('trend1h')} | **15m structure:** {sig.get('trend15m')}")
    st.write(f"**FVG:** {sig.get('fvg')} | **Order-block heuristic:** {sig.get('order_block')}")
    fib = sig.get("fib", {})
    st.markdown("### 1H Fibonacci retracement")
    if pd.notna(fib.get("fib50", np.nan)) and pd.notna(fib.get("fib618", np.nan)):
        st.write(f"Impulse: **{fib.get('direction')}**")
        fib_cols = st.columns(2)
        fib_cols[0].metric("50% retracement", f"{fib['fib50']:,.2f}")
        fib_cols[1].metric("61.8% retracement", f"{fib['fib618']:,.2f}")
        st.write(f"Pullback zone: **{fib['zone_low']:,.2f} – {fib['zone_high']:,.2f}**")
        st.write(f"1H swing anchors: low **{fib['anchor_low']:,.2f}** · high **{fib['anchor_high']:,.2f}**")
        if sig.get("fib_touch"):
            st.success("Price is currently testing the 50%–61.8% Fibonacci zone. This is confluence only, not an entry by itself.")
        else:
            st.caption("Price is outside the Fibonacci pullback zone.")
    else:
        st.info("Fibonacci levels unavailable: waiting for a confirmed 1H directional swing.")
    st.caption("Fibonacci uses confirmed 1H pivots (2 bars on each side). The 50% and 61.8% levels are confluence zones, never standalone signals.")

    daily_ctx = sig.get("daily_context", {})
    daily_fib = daily_ctx.get("fib", {})
    st.markdown("### Daily trend — bigger picture")
    daily_trend = daily_ctx.get("trend", "UNAVAILABLE")
    if daily_trend.startswith("BULLISH"):
        st.success(f"Daily trend: {daily_trend}")
    elif daily_trend.startswith("BEARISH"):
        st.error(f"Daily trend: {daily_trend}")
    else:
        st.info(f"Daily trend: {daily_trend}")
    st.write(f"Daily structure: **{daily_ctx.get('structure', '—')}**")
    if pd.notna(daily_ctx.get("close", np.nan)):
        st.write(
            f"Daily close: **{daily_ctx['close']:,.2f}** · "
            f"EMA20: **{daily_ctx['ema20']:,.2f}** · EMA50: **{daily_ctx['ema50']:,.2f}**"
        )
    if pd.notna(daily_fib.get("fib50", np.nan)) and pd.notna(daily_fib.get("fib618", np.nan)):
        dcols = st.columns(2)
        dcols[0].metric("Daily Fib 50%", f"{daily_fib['fib50']:,.2f}")
        dcols[1].metric("Daily Fib 61.8%", f"{daily_fib['fib618']:,.2f}")
        st.write(
            f"Daily pullback zone: **{daily_fib['zone_low']:,.2f} – "
            f"{daily_fib['zone_high']:,.2f}**"
        )
        if sig.get("daily_fib_touch"):
            st.warning("Price is testing the DAILY Fibonacci zone. Treat it as a major higher-timeframe reaction area, not an entry trigger.")
    else:
        st.info("Daily Fibonacci unavailable: waiting for a confirmed daily impulse swing.")
    st.caption("Daily trend/Fibonacci are context filters. The 5m price-action trigger still controls entry; counter-trend setups are explicitly flagged.")
    st.caption("Swing levels use a confirmed 2-left/2-right pivot method; levels confirm with delay and are not predictive.")

with tabs[2]:
    st.subheader("Paper trading book")
    open_df = pd.read_sql_query("SELECT * FROM paper_trades WHERE status='OPEN' ORDER BY id DESC", CON)
    closed_df = pd.read_sql_query("SELECT * FROM paper_trades WHERE status!='OPEN' ORDER BY id DESC", CON)
    st.write("**Open positions**")
    st.dataframe(open_df, use_container_width=True, hide_index=True)
    st.write("**Closed positions**")
    st.dataframe(closed_df, use_container_width=True, hide_index=True)
    if not open_df.empty:
        st.caption("At T1, the paper stop moves to breakeven and then trails behind the latest confirmed 5m swing level. Positions are checked only when the app refreshes; this is not broker-grade execution simulation.")

with tabs[3]:
    st.subheader("Signal journal")
    journal = pd.read_sql_query("SELECT * FROM signals ORDER BY id DESC LIMIT 500", CON)
    st.dataframe(journal, use_container_width=True, hide_index=True)
    if not journal.empty:
        st.download_button("Download signal journal CSV", journal.to_csv(index=False), "signal_journal.csv", "text/csv")

with tabs[4]:
    st.subheader("Historical backtesting — conservative prototype")
    st.caption("This is a simplified OHLC-based research backtest. It does not model order-book fills, latency, options premiums, or all live signal-engine rules. Validate results independently.")
    bt_symbol = st.selectbox("Index to backtest", list(INDEXES), key="bt_symbol")
    bt_days = st.selectbox("History window", ["5d","1mo","3mo"], index=1)
    if st.button("Run backtest"):
        ticker = INDEXES[bt_symbol]["spot"]
        hist = fetch_ohlcv(ticker, "5m", bt_days)
        if len(hist) < 100:
            st.error("Insufficient 5-minute history for this backtest window.")
        else:
            z = add_indicators(hist)
            # Proxy strategy: EMA trend + VWAP + previous 20-bar breakout + RVOL if available.
            z["prior_high"] = z.high.shift(1).rolling(20).max()
            z["prior_low"] = z.low.shift(1).rolling(20).min()
            z["long_entry"] = (z.close > z.ema20) & (z.ema9 > z.ema20) & (z.close > z.vwap) & (z.close > z.prior_high)
            z["short_entry"] = (z.close < z.ema20) & (z.ema9 < z.ema20) & (z.close < z.vwap) & (z.close < z.prior_low)
            trades, pos = [], None
            for i in range(30, len(z)-1):
                row = z.iloc[i]
                if pos is None:
                    if bool(row.long_entry):
                        risk = max(float(row.atr)*1.0, float(row.close)*0.0005)
                        pos = {"side":"BUY","entry":float(row.close),"stop":float(row.close-risk),
                               "target":float(row.close+2*risk),"time":str(z.index[i])}
                    elif bool(row.short_entry):
                        risk = max(float(row.atr)*1.0, float(row.close)*0.0005)
                        pos = {"side":"SELL","entry":float(row.close),"stop":float(row.close+risk),
                               "target":float(row.close-2*risk),"time":str(z.index[i])}
                else:
                    if pos["side"]=="BUY":
                        hit_sl = row.low <= pos["stop"]
                        hit_tp = row.high >= pos["target"]
                        if hit_sl or hit_tp:
                            # Conservative assumption if both touched in same bar: stop first.
                            out = pos["stop"] if hit_sl else pos["target"]
                            trades.append({**pos,"exit":out,"pnl":out-pos["entry"],"exit_time":str(z.index[i])})
                            pos = None
                    else:
                        hit_sl = row.high >= pos["stop"]
                        hit_tp = row.low <= pos["target"]
                        if hit_sl or hit_tp:
                            out = pos["stop"] if hit_sl else pos["target"]
                            trades.append({**pos,"exit":out,"pnl":pos["entry"]-out,"exit_time":str(z.index[i])})
                            pos = None
            if trades:
                tdf = pd.DataFrame(trades)
                wins = int((tdf.pnl>0).sum())
                st.metric("Trades",len(tdf))
                st.metric("Win rate",f"{wins/len(tdf)*100:.1f}%")
                st.metric("Net points",f"{tdf.pnl.sum():.2f}")
                st.metric("Average points / trade",f"{tdf.pnl.mean():.2f}")
                st.dataframe(tdf,use_container_width=True,hide_index=True)
                st.download_button("Download backtest CSV",tdf.to_csv(index=False),"backtest.csv","text/csv")
            else:
                st.info("No completed trades found in this sample.")

with tabs[5]:
    st.subheader("Data configuration and limitations")
    st.markdown("""
    **Default spot tickers**
    - NIFTY: `^NSEI`
    - BANK NIFTY: `^NSEBANK`
    - SENSEX: `^BSESN`

    **Fibonacci and daily context**
    The app identifies the latest confirmed 1H and daily directional swings and calculates only the 50% and 61.8% retracement levels on each timeframe. The 1H Fibonacci zone contributes up to 10 points to the intraday score. Daily trend and daily Fibonacci are displayed as bigger-picture context and flag counter-trend setups; they do not independently trigger trades.

    **Futures volume configuration**
    Set these environment variables to valid Yahoo Finance futures symbols if available:
    - `NIFTY_FUTURES_TICKER`
    - `BANKNIFTY_FUTURES_TICKER`
    - `SENSEX_FUTURES_TICKER`

    Futures contracts roll over. Update symbols when contracts expire. The app never treats spot-index volume as real traded volume.

    **Deployment**
    1. Upload `app.py` and `requirements.txt` to a GitHub repository.
    2. In Streamlit Community Cloud, set the main file path to `app.py`.
    3. Deploy and monitor the Data & Setup tab.
    4. Add futures ticker environment variables in app Secrets only after verifying the symbols.

    **Important**
    Yahoo Finance is a convenience data source, not an exchange-authorized execution feed. Intraday history, timestamps, symbols and volume coverage can be delayed, incomplete or unavailable. For production-grade signals, use a licensed broker/exchange data API. Never trade solely from this dashboard.
    """)
    st.write("Database file:", DB_PATH)
    st.write("Last app refresh:", datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S %Z"))

st.caption("Paper-trading research tool only. No broker integration, no order placement, and no investment advice.")
