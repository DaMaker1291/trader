#!/usr/bin/env python3
"""
Weekend Sandbox — Train & Backtest the momentum strategy.

Downloads historical data, sweeps exit parameters (TP/SL/trail),
reports optimal config, and retrains ML models.

Usage:
    python3 sandbox.py                  # full run (2y data, may take 20-30 min)
    python3 sandbox.py --quick          # quick: last 90 days
    python3 sandbox.py --train-only     # train ML models only
    python3 sandbox.py --backtest-only  # backtest only
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | Sandbox | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Sandbox")

UNIVERSE: list[str] = [
    "SPY","QQQ","IWM","DIA","VTI","VOO",
    "XLF","XLK","XLV","XLE","XLI","XLP","XLU","XLY","XLB","XLRE",
    "ARKK","ARKQ","ARKW","ARKG","ARKF",
    "UPRO","TQQQ","SOXL","FAS","LABU","SPXL","TECL","FNGU",
    "UDOW","URTY","YINN","CURE","RETL","DPST","JNUG","NUGT",
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA",
    "PLTR","SOUN","IONQ","RGTI","ARM","CRWD","PANW","ZS",
    "DDOG","MDB","SNOW","NET","AI","BBAI",
    "COIN","MSTR","RIOT","CLSK","IBIT",
    "AMD","INTC","MU","AVGO","QCOM","MRVL","AMAT","LRCX",
    "KLAC","NXPI","STM","TSM","ASML","ON",
    "LLY","UNH","JNJ","PFE","ABBV","VRTX","MRNA",
    "REGN","GILD","AMGN","BMY",
    "JPM","GS","BAC","MS","V","MA","AXP","SCHW","BLK",
    "WMT","COST","HD","LOW","NFLX","DIS","SBUX","MCD",
    "CMG","TSCO","TJX","TGT",
    "CRM","NOW","ADBE","ORCL","INTU","UBER","DASH",
    "SNAP","PINS","RBLX","ZM","WDAY","TEAM",
    "LMT","RTX","NOC","GD","BA","GE","CAT","DE",
    "ENPH","SEDG","FSLR","USO","GDX","SLV","GLD",
    "HOOD","CHWY","DKNG","RIVN","AMC","GME","CVNA",
    "CELH","AFRM","SOFI","UPST",
    "ISRG","SYK","MDT","BSX","ABT","DHR","TMO",
    "CMCSA","T","VZ",
    "FDX","UPS",
    "KRE","KBE","HBAN","KEY","ZION","FITB","RF",
    "PLD","AMT","EQIX","DLR",
    "EEM","VWO","FXI","EWJ","EWZ",
    "TLT","IEF","AGG","BND","LQD","HYG",
]

TP_GRID = [1.0, 1.5, 2.0, 3.0]
SL_GRID = [1.0, 1.5, 2.0, 2.5]
TRAIL_ACT_GRID = [0.5, 1.0]
TRAIL_DIST_GRID = [0.8, 1.0, 1.5]
CACHE_FILE = "sandbox_data.pkl"


def download_data(period: str = "2y") -> dict[str, pd.DataFrame]:
    if os.path.exists(CACHE_FILE):
        logger.info("Loading cached data from %s ...", CACHE_FILE)
        return pd.read_pickle(CACHE_FILE)

    logger.info("Downloading %s data for %d symbols ...", period, len(UNIVERSE))
    all_data: dict[str, pd.DataFrame] = {}
    batch_size = 15
    for i in range(0, len(UNIVERSE), batch_size):
        batch = UNIVERSE[i:i + batch_size]
        try:
            data = yf.download(batch, period=period, progress=False, auto_adjust=True, group_by="ticker")
            for sym in batch:
                try:
                    sd = data[sym] if isinstance(data.columns, pd.MultiIndex) and sym in data else data
                    if sd.empty or "Close" not in sd.columns:
                        continue
                    df = pd.DataFrame({
                        "open": sd["Open"], "high": sd["High"],
                        "low": sd["Low"], "close": sd["Close"],
                        "volume": sd["Volume"],
                    }).dropna()
                    if len(df) >= 20:
                        all_data[sym] = df
                except Exception:
                    continue
        except Exception as e:
            logger.warning("Batch download failed: %s", e)
        time.sleep(1.5)
    logger.info("Downloaded %d symbols successfully", len(all_data))
    pd.to_pickle(all_data, CACHE_FILE)
    logger.info("Cached to %s", CACHE_FILE)
    return all_data


def simulate_trade(
    entry: float, shares: float, df: pd.DataFrame, date_idx: int,
    tp_pct: float, sl_pct: float, trail_act_pct: float, trail_dist_pct: float,
) -> tuple[float, float, str | None]:
    day = df.iloc[date_idx]
    o, hi, lo, c = day["open"], day["high"], day["low"], day["close"]
    hi, lo = float(hi), float(lo)
    exit_px: float | None = None
    reason: str | None = None
    high_water = entry

    hard_sl = entry * (1 - sl_pct / 100)
    tp = entry * (1 + tp_pct / 100)

    if lo <= hard_sl:
        return shares * hard_sl, hard_sl, "stop_loss"

    high_water = max(high_water, hi)
    if high_water >= entry * (1 + trail_act_pct / 100):
        trail = high_water * (1 - trail_dist_pct / 100)
        if lo <= trail:
            pnl = shares * trail
            return pnl, trail, "trailing_stop"

    if hi >= tp:
        pnl = shares * tp
        return pnl, tp, "take_profit"

    return 0.0, 0.0, None  # no exit


def run_backtest(
    data: dict[str, pd.DataFrame],
    tp_pct: float, sl_pct: float,
    trail_act_pct: float, trail_dist_pct: float,
    capital: float = 100.0,
) -> dict[str, Any]:
    common_dates = sorted(set.intersection(
        *[set(df.index) for df in data.values() if len(df) > 0]
    ))
    if not common_dates:
        return {"total_return_pct": 0, "num_trades": 0, "win_rate_pct": 0}

    trades: list[dict] = []
    position: tuple[str, float, int, float, float] | None = None

    for dt in common_dates:
        if position is not None:
            sym, entry, date_idx, shares, high_water = position
            if sym in data:
                df = data[sym]
                pos_idx = list(df.index).index(dt) if dt in df.index else -1
                if pos_idx > 0 and pos_idx == date_idx + 1:
                    pnl, exit_px, reason = simulate_trade(
                        entry, shares, df, pos_idx,
                        tp_pct, sl_pct, trail_act_pct, trail_dist_pct,
                    )
                    if reason is not None:
                        capital += pnl
                        pnl_pct = (exit_px - entry) / entry * 100
                        trades.append({
                            "symbol": sym,
                            "entry_px": round(entry, 2),
                            "exit_px": round(exit_px, 2),
                            "pnl_pct": round(pnl_pct, 2),
                            "reason": reason,
                        })
                        position = None

        if position is not None:
            continue

        candidates: list[tuple[str, float, float]] = []
        for sym, df in data.items():
            if dt not in df.index:
                continue
            idx = list(df.index).index(dt)
            if idx == 0:
                continue
            prev_close = float(df.iloc[idx - 1]["close"])
            cur_close = float(df.iloc[idx]["close"])
            if prev_close <= 0:
                continue
            chg = (cur_close - prev_close) / prev_close * 100
            if 1.0 <= chg <= 12.0:
                candidates.append((sym, chg, cur_close))

        if not candidates:
            continue
        candidates.sort(key=lambda x: x[1], reverse=True)
        sym, chg, price = candidates[0]
        shares = capital / price
        capital = 0
        pos_idx = list(data[sym].index).index(dt)
        position = (sym, price, pos_idx, shares, price)

    if position is not None:
        sym, entry, _, shares, _ = position
        if sym in data and common_dates[-1] in data[sym].index:
            final = float(data[sym].loc[common_dates[-1], "close"])
            capital += shares * final

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0

    returns = [t["pnl_pct"] for t in trades]
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0

    total_ret = (capital - 100) / 100 * 100

    return {
        "params": {"tp": tp_pct, "sl": sl_pct, "trail_act": trail_act_pct, "trail_dist": trail_dist_pct},
        "total_return_pct": round(total_ret, 2),
        "num_trades": len(trades),
        "win_rate_pct": round(win_rate, 1),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "sharpe": round(sharpe, 3),
    }


def sweep(data: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(TP_GRID) * len(SL_GRID) * len(TRAIL_ACT_GRID) * len(TRAIL_DIST_GRID)
    count = 0
    for tp in TP_GRID:
        for sl in SL_GRID:
            for ta in TRAIL_ACT_GRID:
                for td in TRAIL_DIST_GRID:
                    count += 1
                    logger.info("[%d/%d] TP=%.1f%% SL=%.1f%% TrAct=%.1f%% TrDist=%.1f%%",
                                count, total, tp, sl, ta, td)
                    r = run_backtest(data, tp, sl, ta, td)
                    results.append(r)
                    best = max(results, key=lambda x: x["total_return_pct"])
                    logger.info("  -> %.2f%% | Best: %.2f%% (TP=%.1f SL=%.1f TrAct=%.1f TrDist=%.1f)",
                                r["total_return_pct"], best["total_return_pct"],
                                best["params"]["tp"], best["params"]["sl"],
                                best["params"]["trail_act"], best["params"]["trail_dist"])
    return results


def train_models(data: dict[str, pd.DataFrame]) -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    features: list[list[float]] = []
    labels: list[int] = []

    for sym, df in data.items():
        c = df["close"].values
        h = df["high"].values
        l_ = df["low"].values
        v = df["volume"].values
        for i in range(20, len(c) - 1):
            r1 = (c[i] / c[i - 1] - 1) * 100
            r5 = (c[i] / c[i - 5] - 1) * 100 if i >= 5 else 0
            r10 = (c[i] / c[i - 10] - 1) * 100 if i >= 10 else 0
            vola = np.std(c[i - 10:i]) / c[i] * 100 if i >= 10 else 0
            vr = v[i] / np.mean(v[i - 10:i]) if i >= 10 and np.mean(v[i - 10:i]) > 0 else 1
            hl = (h[i] - l_[i]) / c[i] * 100
            features.append([r1, r5, r10, vola, vr, hl])
            labels.append(1 if c[i + 1] > c[i] else 0)

    if len(features) < 100:
        logger.warning("Not enough samples: %d", len(features))
        return

    X = np.array(features)
    y = np.array(labels)

    os.makedirs("models", exist_ok=True)

    # Train RF
    try:
        from trading_engine import MLPredictor, MLConfig
        cfg = MLConfig()
        predictor = MLPredictor(cfg)
        for i in range(len(X)):
            predictor.feature_window.append(X[i])
            predictor.label_window.append(y[i])
        predictor.train()
        logger.info("MLPredictor RF trained on %d samples", len(X))
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import joblib
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
        rf.fit(Xs, y)
        acc = rf.score(Xs, y)
        joblib.dump(rf, "models/rf_model.joblib")
        joblib.dump(scaler, "models/scaler.joblib")
        logger.info("Standalone RF trained: %d samples, accuracy=%.4f", len(X), acc)

    # Train LSTM
    try:
        from trading_engine import LSTMPredictor
        lstm = LSTMPredictor(
            input_size=6, hidden_size=32, output_size=1,
            seq_length=10, learning_rate=0.001,
        )
        X_seq, y_seq = [], []
        for i in range(10, len(X)):
            X_seq.append(X[i - 10:i])
            y_seq.append(y[i])
        if len(X_seq) > 100:
            X_seq_a = np.array(X_seq, dtype=np.float32)
            y_seq_a = np.array(y_seq, dtype=np.float32)
            lstm.train((X_seq_a, y_seq_a), epochs=3, batch_size=32)
            lstm.save("models/lstm_model.pt")
            logger.info("LSTM trained on %d sequences", len(X_seq))
    except Exception as e:
        logger.warning("LSTM training skipped: %s", e)

    np.save("models/feature_window.npy", X)
    np.save("models/label_window.npy", y)
    logger.info("Training data saved to models/")


def print_report(results: list[dict[str, Any]]) -> None:
    results.sort(key=lambda r: r["total_return_pct"], reverse=True)
    print("\n" + "=" * 90)
    print("  RANKED BACKTEST RESULTS")
    print("=" * 90)
    hdr = f"{'Rank':<5} {'TP%':<6} {'SL%':<6} {'TrAct':<7} {'TrDist':<7} {'Return%':<9} {'Trades':<7} {'Win%':<7} {'AvgW':<7} {'AvgL':<7} {'Sharpe':<7}"
    print(hdr)
    print("-" * 90)
    for rank, r in enumerate(results[:15], 1):
        p = r["params"]
        print(f"{rank:<5} {p['tp']:<6} {p['sl']:<6} {p['trail_act']:<7} {p['trail_dist']:<7} "
              f"{r['total_return_pct']:<9} {r['num_trades']:<7} {r['win_rate_pct']:<7} "
              f"{r['avg_win_pct']:<7} {r['avg_loss_pct']:<7} {r['sharpe']:<7}")
    print("-" * 90)

    best_tr = results[0]
    best_wr = max(results, key=lambda r: r["win_rate_pct"])
    best_sh = max(results, key=lambda r: r["sharpe"])

    print(f"\n  BEST TOTAL RETURN: {best_tr['total_return_pct']}%"
          f"  TP={best_tr['params']['tp']}%  SL={best_tr['params']['sl']}%"
          f"  TrAct={best_tr['params']['trail_act']}%  TrDist={best_tr['params']['trail_dist']}%")
    print(f"  BEST WIN RATE:    {best_wr['win_rate_pct']}%"
          f"  TP={best_wr['params']['tp']}%  SL={best_wr['params']['sl']}%"
          f"  TrAct={best_wr['params']['trail_act']}%  TrDist={best_wr['params']['trail_dist']}%")
    print(f"  BEST SHARPE:      {best_sh['sharpe']}"
          f"  TP={best_sh['params']['tp']}%  SL={best_sh['params']['sl']}%"
          f"  TrAct={best_sh['params']['trail_act']}%  TrDist={best_sh['params']['trail_dist']}%")

    print(f"\n  AVG return across {len(results)} sets: "
          f"{np.mean([r['total_return_pct'] for r in results]):.2f}%")
    print("=" * 90)

    best = results[0]
    print(f"\n  RECOMMENDED PARAMS for trading_engine.py:")
    print(f"    TP={best['params']['tp']}% -> entry * {1 + best['params']['tp']/100:.3f}")
    print(f"    SL={best['params']['sl']}% -> entry * {1 - best['params']['sl']/100:.3f}")
    print(f"    Trail activation={best['params']['trail_act']}% gain")
    print(f"    Trail distance={best['params']['trail_dist']}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekend Sandbox")
    parser.add_argument("--quick", action="store_true", help="Quick: 90 days")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--backtest-only", action="store_true")
    parser.add_argument("--period", default="2y")
    args = parser.parse_args()

    period = "3mo" if args.quick else args.period
    print("\n SANDBOX MODE ")
    print("=" * 60)
    print(f"  Period:  {period}")
    print(f"  Symbols: {len(UNIVERSE)}")
    print(f"  Params:  {len(TP_GRID) * len(SL_GRID) * len(TRAIL_ACT_GRID) * len(TRAIL_DIST_GRID)} combos")
    print("=" * 60)

    data = download_data(period)

    if not args.train_only:
        results = sweep(data)
        print_report(results)
        with open("sandbox_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to sandbox_results.json")

    if not args.backtest_only:
        logger.info("Training ML models on %d symbols ...", len(data))
        train_models(data)

    print("\n SANDBOX COMPLETE\n")


if __name__ == "__main__":
    main()
