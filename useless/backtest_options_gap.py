"""
backtest_options_gap.py — Historical backtest for options_gap_bot.py strategy.
Simulates buying 0.30-delta OTM options on gap-up days.

Option return approximation:
  - 0.30 delta OTM option on an 8%+ gap-up
  - Approx return = delta * stock_return * gamma_multiplier
  - gamma_multiplier grows as stock moves ITM (typically 1.5x-4x)
  - Hard SL at -30% of premium (option can lose value quickly)
  - Trail activates at +50% gain, trails 25% below peak
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("BacktestOptions")

CAPITAL = 200.0
TRANCHES = 4
TRANCH_SIZE = CAPITAL / TRANCHES  # $50
MAX_POSITIONS = 2

MIN_GAP = 5.0
MIN_PRICE = 10.0
MAX_PRICE = 500.0
MIN_VOL = 1_000_000
MIN_PRE_VOL = 50_000

OPTION_SL_PCT = 0.30
TRAIL_ACTIVATE_PCT = 0.50
TRAIL_DISTANCE_PCT = 0.25

DELTA_TARGET = 0.30
OPTION_PREMIUM_PCT = 0.05  # option premium as % of stock price (typical for 0.30 delta OTM)

WATCHLIST = [
    "NVDA","AMD","TSLA","META","AMZN","GOOGL","MSFT","AAPL","NFLX",
    "COIN","MSTR","SMCI","ARM","CRWD","PANW","DASH","UBER",
    "HOOD","AFRM","UPST","SOFI","MARA","RIOT","CLSK",
    "AI","IONQ","RGTI","QBTS","BBAI","SOUN","RDDT",
    "TQQQ","SOXL","FAS","LABU","UPRO","FNGU",
    "NVDU","NVDL","TSLL","CONL","AMDL","MSTU","MSTX",
    "CELH","CHWY","DKNG","RIVN","CVNA",
]

PERIOD = "2y"

def option_return(stock_move_pct: float, delta: float, gamma_mult: float = 2.0) -> float:
    """Estimate option return from stock move.
    
    For small moves: option_return ≈ delta * stock_move
    For large moves with gamma: option_return ≈ delta * stock_move * gamma_mult
    Gamma effect grows as stock moves deeper ITM.
    """
    abs_move = abs(stock_move_pct)
    # Gamma multiplier increases with move size (more gamma for bigger moves)
    gm = 1.0 + (abs_move / 10.0) * (gamma_mult - 1.0)
    gm = min(gm, 4.0)  # cap at 4x
    est = delta * stock_move_pct * gm
    return est


logger.info("Backtesting options_gap_bot.py on %d stocks, %s", len(WATCHLIST), PERIOD)
logger.info("  Capital: $%.0f (%d x $%.0f tranches)", CAPITAL, TRANCHES, TRANCH_SIZE)
logger.info("  Delta: %.2f | Option premium: ~%.0f%% of stock price", DELTA_TARGET, OPTION_PREMIUM_PCT * 100)
logger.info("  SL: -%.0f%% of premium | Trail: +%.0f%% -> trail %.0f%% below peak",
            OPTION_SL_PCT * 100, TRAIL_ACTIVATE_PCT * 100, TRAIL_DISTANCE_PCT * 100)
logger.info("  Min gap: +%.0f%%", MIN_GAP)
logger.info("=" * 60)

results = []
trades = 0
total_pnl = 0.0
wins = 0

# Test multiple gamma multiplier assumptions
for gamma_mult in [1.5, 2.0, 3.0]:
    trades = 0
    total_pnl = 0.0
    wins = 0
    trade_results = []
    
    for sym in WATCHLIST:
        try:
            tk = yf.Ticker(sym)
            df = tk.history(period=PERIOD)
            if df.empty or len(df) < 20:
                continue

            df["gap_pct"] = (df["Open"] / df["Close"].shift(1) - 1) * 100

            for i in range(1, len(df)):
                row = df.iloc[i]
                open_p = row["Open"]
                high = row["High"]
                low = row["Low"]
                close = row["Close"]
                vol = row["Volume"]
                gap = row["gap_pct"]
                d = row.name
                if isinstance(d, pd.Timestamp):
                    d = d.to_pydatetime()
                if d.weekday() >= 5:
                    continue
                if gap < MIN_GAP or open_p < MIN_PRICE or open_p > MAX_PRICE or vol < MIN_VOL:
                    continue

                trades += 1
                
                # Option premium cost
                premium_per_share = open_p * OPTION_PREMIUM_PCT
                contracts = max(1, int(TRANCH_SIZE / (premium_per_share * 100)))
                cost = contracts * premium_per_share * 100
                
                # Simulate the option P&L
                daily_stock_return = ((close / open_p) - 1) * 100
                intra_stock_high = ((high / open_p) - 1) * 100
                intra_stock_low = ((low / open_p) - 1) * 100
                
                opt_ret = option_return(daily_stock_return, DELTA_TARGET, gamma_mult)
                
                # Simulate trailing stop on the option
                entry_premium = premium_per_share
                peak_premium = entry_premium
                sl_premium = entry_premium * (1 - OPTION_SL_PCT)
                trail_active = False
                exit_premium = None
                reason = ""
                
                # Check intraday for SL
                opt_intra_low = option_return(intra_stock_low, DELTA_TARGET, gamma_mult)
                opt_intra_high = option_return(intra_stock_high, DELTA_TARGET, gamma_mult)
                
                if opt_intra_low <= -OPTION_SL_PCT * 100:
                    exit_premium = sl_premium
                    reason = "hard_sl"
                elif opt_intra_high >= TRAIL_ACTIVATE_PCT * 100:
                    trail_active = True
                    peak_premium = entry_premium * (1 + opt_intra_high / 100)
                    trail_stop = peak_premium * (1 - TRAIL_DISTANCE_PCT)
                    if opt_intra_low <= ((trail_stop / entry_premium) - 1) * 100:
                        exit_premium = trail_stop
                        reason = "trail"
                
                if exit_premium is None:
                    exit_premium = entry_premium * (1 + opt_ret / 100)
                    reason = "eod_close"
                
                gain_pct = ((exit_premium / entry_premium) - 1) * 100
                pnl = cost * gain_pct / 100
                total_pnl += pnl
                if gain_pct > 0:
                    wins += 1
                
                trade_results.append({
                    "date": d.strftime("%Y-%m-%d"), "sym": sym,
                    "gap": round(gap, 1), "entry_stock": round(open_p, 2),
                    "stock_return": round(daily_stock_return, 1),
                    "opt_return": round(gain_pct, 1), "pnl": round(pnl, 2),
                    "reason": reason
                })

        except Exception:
            continue

    if trades == 0:
        continue

    df_r = pd.DataFrame(trade_results)
    wr = wins / trades * 100
    avg_win = df_r[df_r["opt_return"] > 0]["opt_return"].mean() if len(df_r[df_r["opt_return"] > 0]) > 0 else 0
    avg_loss = df_r[df_r["opt_return"] <= 0]["opt_return"].mean() if len(df_r[df_r["opt_return"] <= 0]) > 0 else 0
    
    logger.info(f"\nGamma multiplier: {gamma_mult}x")
    logger.info(f"  Trades: {trades} | WR: {wr:.1f}%")
    logger.info(f"  Avg win: +{avg_win:.1f}% | Avg loss: {avg_loss:.1f}%")
    logger.info(f"  Max win: +{df_r['opt_return'].max():.1f}% | Max loss: {df_r['opt_return'].min():.1f}%")
    logger.info(f"  Total P&L: ${total_pnl:.2f} | Avg/trade: ${total_pnl/trades:.2f}")

    total_days_range = (pd.to_datetime(df_r["date"]).max() - pd.to_datetime(df_r["date"]).min()).days
    trading_days = total_days_range * 5 / 7
    hrs = trading_days * 6.5
    logger.info(f"  Per hour: ${total_pnl/hrs:.2f}" if hrs > 0 else "")
    logger.info(f"  Per trading day: ${total_pnl/trading_days:.2f}" if trading_days > 0 else "")

    mc = []
    for _ in range(10000):
        s = df_r["opt_return"].sample(n=len(df_r), replace=True)
        mc.append(s.sum())
    mc_s = pd.Series(mc)
    logger.info(f"  MC P(profit): {(mc_s > 0).mean()*100:.1f}% | Median: {mc_s.median():.1f}%")

    if gamma_mult == 2.0:
        df_r.to_csv("/tmp/options_gap_backtest.csv", index=False)
        top5 = df_r.nlargest(5, "opt_return")
        worst5 = df_r.nsmallest(5, "opt_return")
        logger.info(f"\nTop 5 (γ=2x):")
        for _, r in top5.iterrows():
            logger.info(f"  +{r['opt_return']:.0f}%  ${r['pnl']:.2f}  {r['sym']:6s}  {r['date']}  ({r['reason']})")
        logger.info(f"\nWorst 5 (γ=2x):")
        for _, r in worst5.iterrows():
            logger.info(f"  {r['opt_return']:.0f}%  ${r['pnl']:.2f}  {r['sym']:6s}  {r['date']}  ({r['reason']})")

logger.info("\n" + "=" * 60)
logger.info("Options backtest complete. Results saved to /tmp/options_gap_backtest.csv")
