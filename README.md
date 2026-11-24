# Index Signal Dashboard

Paper-trading Streamlit dashboard for NIFTY 50, BANK NIFTY and SENSEX.

## Included
- 1H trend filter, 15m setup, 5m trigger
- EMA 9/20/50, RSI, VWAP, ATR
- PDH, PDL, PDC, pivot levels and CPR
- Swing-based support/resistance, BOS/CHoCH proxies
- Automatic 1H and daily trend context using EMA alignment and swing structure
- Automatic 1H and daily Fibonacci retracement using only 50% and 61.8% levels; 1H zone test contributes up to 10 confluence points; daily levels provide higher-timeframe context
- Candle rejection/engulfing, liquidity sweep, FVG and order-block heuristics
- Futures-volume RVOL when configured
- BUY/SELL/REVERSAL WATCH/NO TRADE with reasons
- Paper-trade ledger, signal journal and CSV export
- Simplified historical breakout backtest

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Upload `app.py` and `requirements.txt` to GitHub. In Streamlit Community Cloud, select `app.py` as the main file.

## Data
Default spot data comes from Yahoo Finance through `yfinance`:
- `^NSEI`
- `^NSEBANK`
- `^BSESN`

This is not an official NSE/BSE feed. Intraday availability can vary. For actual volume confirmation, configure verified futures symbols as environment variables:
- `NIFTY_FUTURES_TICKER`
- `BANKNIFTY_FUTURES_TICKER`
- `SENSEX_FUTURES_TICKER`

Do not use a made-up symbol or assume the futures contract rolls automatically. Update the ticker when the active contract changes. Without futures volume, confirmed signals are downgraded when volume confirmation is required.

## Limitations
- The live signal engine is a rule-based prototype, not a validated profitable strategy.
- BOS/CHoCH, order blocks, FVG and support/resistance are heuristic approximations and need visual validation.
- Backtest tab uses a simplified 5-minute breakout proxy; it does not reproduce every live signal rule or simulate realistic slippage, fees, latency, partial fills or options premiums.
- Paper trades use spot index prices and simplified target/stop checking. They are not broker fills.
- No options chain, open interest, broker connection or live order execution.


## Dashboard visual layout
- Dark market-terminal style overview with three index status cards, a 5-minute candlestick chart, and a signal/entry/SL/targets panel.
- Fibonacci overlays use separate palettes: 1H = cyan/teal; Daily = violet/pink.
- Daily trend, Daily Fib and 1H Fib are shown together as higher-timeframe context.
\n\n## Visual terminal and market status\n- Dark visual trading-terminal layout with index cards, interactive candlestick chart and signal explanation panel.\n- 1H Fibonacci lines use cyan/teal; daily Fibonacci lines use violet/pink.\n- Market status distinguishes weekend/outside-hours, data freshness, and a manually confirmed NSE holiday.\n

## Signal strategy (Option B)
The daily timeframe is used as higher-timeframe context only; it does not block an otherwise valid intraday signal. BUY/SELL qualification is driven by 1H trend, 15-minute setup/structure, and a 5-minute price-action trigger, plus the configured volume and risk/reward gates. When the 1H direction conflicts with the daily trend, the dashboard displays a counter-trend caution. Signals are for paper testing, not financial advice.


## Candle-close and futures-volume safeguards
- Signal calculations and futures RVOL use only completed 5-minute candles. The currently forming 5-minute candle is excluded using its candle-start timestamp plus the full 5-minute interval.
- Futures ticker settings are read from environment variables or Streamlit secrets. The app contains configuration variable names, not preconfigured active futures contracts.
- Configure active contract symbols yourself after verifying the exact Yahoo Finance symbol. Do not assume symbols or contract rolls.
- Futures-volume confirmation is enabled by default only when all three index futures ticker settings are present. If no tickers are configured, the app leaves that gate off and clearly labels futures volume as unavailable; this does not mean volume has been confirmed.


## Reliability, risk controls and forward paper trading
- All signal timeframes (5m, 15m, 60m and daily) are filtered to completed candles before indicators and signal logic run. Daily candles from the current local date are excluded.
- Page 1 includes an interactive Plotly candlestick chart with EMA 9/20/50 and VWAP overlays.
- Paper risk controls include max risk per trade, daily realized-loss stop, maximum concurrent paper positions, and maximum paper entries per day.
- Optional auto-forward paper mode records each qualifying signal once per completed 5-minute candle. It never sends broker orders. It requires the app to remain running/refreshing and is not a substitute for a continuous execution service.
- Performance analytics report closed-trade win rate, net P&L, expectancy, profit factor, average win/loss, max drawdown and cumulative P&L curve. Brokerage, taxes and slippage are not yet included.
- Run automated checks with: `python -m unittest -v test_risk_engine.py`.
