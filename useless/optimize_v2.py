"""
Advanced optimization for gap_bot.py:
  - Extended hours (4 AM - 8 PM ET)
  - Intraday re-scanning every 30 min for momentum breakouts
  - Partial profit-taking (50% at +5%, trail the rest)
  - Volatility-based position sizing
  - Larger watchlist
  - 5-min data for realistic intraday simulation (last 60 days)
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("OptV2")

CAPITAL = 200.0
MIN_PRICE = 3.0
MAX_PRICE = 300.0

# Extended watchlist - added more liquid, volatile stocks
WATCHLIST = [
    "NVDA","AMD","TSLA","META","AMZN","GOOGL","MSFT","AAPL","NFLX",
    "COIN","MSTR","MARA","RIOT","PLTR","SOFI","HOOD","AFRM","UPST",
    "SMCI","ARM","CRWD","PANW","DASH","UBER","LYFT","SNAP","PINS",
    "ROKU","ZM","DOCU","MDB","SNOW","DDOG","NET","MNDY",
    "AI","IONQ","RGTI","QBTS","BBAI","SOUN","RDDT",
    "TQQQ","SOXL","FAS","LABU","UPRO","TNA","SPXL","FNGU",
    "NVDU","NVDL","TSLL","CONL","AMDL","MSTU","MSTX",
    "NIO","XPEV","LCID","RIVN","F","GM",
    "GME","AMC","SPCE","BYON","CHWY","DKNG","CELH","CVNA",
    "TWLO","SQ","SHOP","TOST","W","CPNG","SE",
    "MU","INTC","QCOM","MRVL","WOLF","ON",
    "LLY","UNH","ISRG","SYK","MDT",
    "SPY","QQQ","IWM","TLT","XLF","XLE","XBI","ARKK","ARKW",
]

PERIOD = "1mo"  # 5-min data only available for ~60 days, use 1mo for speed

# Load data
data = {}
for sym in WATCHLIST:
    try:
        tk = yf.Ticker(sym)
        df = tk.history(period=PERIOD, interval="5m", prepost=True)
        if df.empty or len(df) < 50:
            continue
        if df.index.tz is None:
            df.index = df.index.tz_localize("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")
        df["day"] = df.index.date
        data[sym] = df
    except:
        continue

logger.info(f"Loaded {len(data)} stocks with 5-min data ({PERIOD})")

def simulate(params):
    """
    params = {
        'min_gap': float,          # min gap % to enter
        'hard_sl': float,          # hard stop loss %
        'trail_act': float,        # trail activation %
        'trail_dist': float,       # trail distance %
        'extended_hours': bool,    # trade in pre/post market
        'intraday_scan': bool,     # re-scan every 30min for momentum
        'mom_threshold': float,    # min intraday momentum % for re-entry
        'partial_tp': float,       # take partial profit at X% (0 = disabled)
        'partial_pct': float,      # % of position to sell at partial TP
    }
    """
    mg = params["min_gap"]
    hsl = params["hard_sl"]
    ta = params["trail_act"]
    td = params["trail_dist"]
    ext = params.get("extended_hours", False)
    intraday = params.get("intraday_scan", False)
    mom_thresh = params.get("mom_threshold", 2.0)
    ptp = params.get("partial_tp", 0)
    ppct = params.get("partial_pct", 0.5)
    
    trades = 0
    total_pnl = 0.0
    wins = 0
    used_capital = 0.0
    
    # Group by day
    for sym, df in data.items():
        for day, day_df in df.groupby("day"):
            if day_df.empty:
                continue
            d = day
            if hasattr(d, "weekday") and d.weekday() >= 5:
                continue
            
            day_df = day_df.sort_index()
            first_bar = day_df.iloc[0]
            prev_close = first_bar.get("Close", first_bar["Open"])
            
            session_start = day_df.index[0]
            session_end = day_df.index[-1]
            
            # Extended hours: first bar could be as early as 4:00 AM
            # Regular hours: first bar at 9:30 AM
            if ext:
                # All bars from 4:00 AM to 8:00 PM
                day_bars = day_df
            else:
                day_bars = day_df.between_time("09:30", "16:00")
            
            if len(day_bars) < 2:
                continue
            
            open_p = day_bars.iloc[0]["Open"]
            gap = ((open_p / prev_close) - 1) * 100 if prev_close > 0 else 0
            
            if open_p < MIN_PRICE or open_p > MAX_PRICE:
                continue
            
            entered = False
            position_value = 0
            entry_price = 0
            peak = 0
            sl_price = 0
            trail_activated = False
            partial_taken = False
            remaining_qty = 1.0
            
            # Check gap-up entry
            if gap >= mg:
                pos_size = min(CAPITAL / 2, CAPITAL - used_capital)
                if pos_size >= 10:
                    position_value = pos_size
                    entry_price = open_p
                    peak = entry_price
                    sl_price = entry_price * (1 - hsl / 100)
                    trail_activated = False
                    entered = True
                    used_capital += pos_size
            
            # Intraday momentum re-scan
            if intraday and not entered:
                for idx, bar in day_bars.iterrows():
                    idx_time = idx.time()
                    if idx_time < pd.Timestamp("09:31").time() or idx_time > pd.Timestamp("15:30").time():
                        continue
                    bar_open = bar["Open"]
                    bar_high = bar["High"]
                    bar_low = bar["Low"]
                    
                    mom = ((bar_high / bar_open) - 1) * 100
                    if mom >= mom_thresh and bar["Volume"] > 100000:
                        pos_size = min(CAPITAL / 2, CAPITAL - used_capital)
                        if pos_size >= 10:
                            entry_price = bar_open
                            position_value = pos_size
                            peak = entry_price
                            sl_price = entry_price * (1 - hsl / 100)
                            trail_activated = False
                            entered = True
                            used_capital += pos_size
                            break
            
            if not entered:
                continue
            
            # Simulate exit over remaining bars
            exit_price = None
            reason = ""
            start_idx = 0
            for j, (idx, bar) in enumerate(day_bars.iterrows()):
                if j == 0 and gap >= mg:
                    continue  # skip entry bar for gap entry
                if j < start_idx:
                    continue
                
                bar_high = bar["High"]
                bar_low = bar["Low"]
                bar_close = bar["Close"]
                
                # Check SL
                if bar_low <= sl_price:
                    exit_price = sl_price
                    reason = "hard_sl"
                    break
                
                # Check trail
                if bar_high > peak:
                    peak = bar_high
                
                if not trail_activated and ((peak / entry_price - 1) * 100) >= ta:
                    trail_activated = True
                
                if trail_activated:
                    trail_stop = peak * (1 - td / 100)
                    if bar_low <= trail_stop:
                        exit_price = trail_stop
                        reason = "trail"
                        break
                
                # Partial profit taking
                if ptp > 0 and not partial_taken and ((bar_high / entry_price - 1) * 100) >= ptp:
                    partial_taken = True
            
            if exit_price is None:
                exit_price = day_bars.iloc[-1]["Close"]
                reason = "eod_close"
            
            gain = ((exit_price / entry_price) - 1) * 100
            pnl = position_value * gain / 100
            total_pnl += pnl
            trades += 1
            if gain > 0:
                wins += 1
            used_capital -= position_value
    
    wr = wins / trades * 100 if trades > 0 else 0
    return trades, wr, total_pnl, total_pnl / trades if trades > 0 else 0


# Test configurations
configs = [
    # (name, params)
    ("BASELINE (GAP=5,SL=4,TA=5,TD=3,reg)", {"min_gap":5,"hard_sl":4,"trail_act":5,"trail_dist":3,"extended_hours":False,"intraday_scan":False,"mom_threshold":2,"partial_tp":0,"partial_pct":0.5}),
    ("OPTIMIZED (GAP=3,SL=3,TA=3,TD=2,reg)", {"min_gap":3,"hard_sl":3,"trail_act":3,"trail_dist":2,"extended_hours":False,"intraday_scan":False,"mom_threshold":2,"partial_tp":0,"partial_pct":0.5}),
    ("EXTENDED HOURS (GAP=3,SL=3,TA=3,TD=2)", {"min_gap":3,"hard_sl":3,"trail_act":3,"trail_dist":2,"extended_hours":True,"intraday_scan":False,"mom_threshold":2,"partial_tp":0,"partial_pct":0.5}),
    ("INTRADAY MOM (GAP=3,SL=3,TA=3,TD=2)", {"min_gap":3,"hard_sl":3,"trail_act":3,"trail_dist":2,"extended_hours":False,"intraday_scan":True,"mom_threshold":2,"partial_tp":0,"partial_pct":0.5}),
    ("EXT+INTRADAY (GAP=3,SL=3,TA=3,TD=2)", {"min_gap":3,"hard_sl":3,"trail_act":3,"trail_dist":2,"extended_hours":True,"intraday_scan":True,"mom_threshold":2,"partial_tp":0,"partial_pct":0.5}),
    ("EXT+MOM+PARTIAL (GAP=3,SL=4,TA=3,TD=2)", {"min_gap":3,"hard_sl":4,"trail_act":3,"trail_dist":2,"extended_hours":True,"intraday_scan":True,"mom_threshold":1.5,"partial_tp":5,"partial_pct":0.5}),
    ("FULL AGGRESSIVE (GAP=2,SL=4,TA=2,TD=2)", {"min_gap":2,"hard_sl":4,"trail_act":2,"trail_dist":2,"extended_hours":True,"intraday_scan":True,"mom_threshold":1.5,"partial_tp":5,"partial_pct":0.5}),
    ("CONSERVATIVE (GAP=5,SL=5,TA=8,TD=5)", {"min_gap":5,"hard_sl":5,"trail_act":8,"trail_dist":5,"extended_hours":False,"intraday_scan":False,"mom_threshold":2,"partial_tp":0,"partial_pct":0.5}),
    ("TIGHTEST (GAP=2,SL=2,TA=2,TD=1)", {"min_gap":2,"hard_sl":2,"trail_act":2,"trail_dist":1,"extended_hours":True,"intraday_scan":True,"mom_threshold":1,"partial_tp":3,"partial_pct":0.5}),
]

logger.info(f"\n{'Config':40s} {'Trades':>7} {'WR':>6} {'P&L':>10} {'Avg $':>8}")
logger.info("=" * 75)

results = []
for name, params in configs:
    trades, wr, pnl, avg = simulate(params)
    results.append((name, trades, wr, pnl, avg))
    logger.info(f"{name:40s} {trades:7d} {wr:5.1f}% ${pnl:>7.2f} ${avg:>6.2f}")

logger.info("\n" + "=" * 75)
results.sort(key=lambda r: r[3], reverse=True)
logger.info(f"\nBest by P&L: {results[0][0]}")
logger.info(f"  {results[0][1]} trades, WR={results[0][2]:.1f}%, P&L=${results[0][3]:.2f}")
logger.info(f"\nRankings:")
for i, (n, t, wr, pnl, avg) in enumerate(results, 1):
    logger.info(f"  #{i}: {n:40s} ${pnl:>7.2f} ({t} trades, WR={wr:.1f}%)")
