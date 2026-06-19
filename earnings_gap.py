"""
Earnings Gap Bot v2 — Trailing Stop Edition.
Buys gap-ups at 9:31 AM ET with:
  - Hard SL at -4%
  - Trailing stop activates at +8%, trails at 5% distance
  - This lets winners run to +20%, +50%, +100%+ while locking in gains

Target: $24/day ($1/hr) on $200 capital.

Usage:
  export APCA_API_KEY_ID=your_key
  export APCA_API_SECRET_KEY=your_secret
  python3 earnings_gap.py [--sim]

Schedule:
  9:00 AM ET  → initial scan
  9:25 AM ET  → final scan, select best setup
  9:31 AM ET  → enter position (market order + bracket)
"""
import asyncio, json, time, os, sys, logging
from datetime import datetime, timezone, timedelta, date
from typing import Optional, List
import yfinance as yf
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("GapBot")

# ── Config ──────────────────────────────────────────────────────────────
CAPITAL = 200.0
HARD_SL = 4.0              # % — always exit if this is hit
TRAIL_ACTIVATE = 8.0       # % gain to activate trailing stop
TRAIL_DIST = 5.0           # % below peak to trail
MIN_GAP = 8.0              # min pre-market gap %
MIN_PRE_VOL = 100_000      # min pre-market volume
REL_VOL_MIN = 2.0          # min relative volume vs 30-day avg
MIN_PRICE = 5.0
MAX_PRICE = 200.0
MAX_POSITIONS = 1          # trade 1 at a time with $200
ENTER_AFTER_SEC = 90       # enter 90s after 9:30 (wait for initial volatility)

WATCHLIST = [
    "NVDA","AMD","TSLA","META","AMZN","GOOGL","MSFT","AAPL","NFLX",
    "COIN","MSTR","MARA","RIOT","PLTR","SOFI","HOOD","AFRM","UPST",
    "SMCI","ARM","CRWD","PANW","DASH","UBER","LYFT","SNAP","PINS",
    "ROKU","ZM","DOCU","MDB","SNOW","DDOG","NET","CFLT","MNDY",
    "AI","IONQ","RGTI","QBTS","BBAI","SOUN","RDDT",
    "TQQQ","SOXL","FAS","LABU","UPRO","TNA","SPXL","FNGU",
    "NVDU","NVDL","TSLL","CONL","AMDL","MSTU","MSTX",
]

# ── Alpaca Setup ────────────────────────────────────────────────────────
def get_alpaca_client():
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        return None, "no keys"
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(key, secret, paper=True)
        acct = client.get_account()
        logger.info("✅ Alpaca paper: $%.2f equity", float(acct.equity))
        return client, "ok"
    except Exception as e:
        return None, str(e)


# ── Pre-Market Scanner ──────────────────────────────────────────────────
def get_avg_vol(sym: str) -> float:
    """30-day average volume via yfinance."""
    try:
        tk = yf.Ticker(sym)
        hist = tk.history(period="2mo")
        if hist.empty: return 0
        return float(hist["Volume"].tail(30).mean())
    except: return 0


async def scan_premarket() -> list:
    """Scan WATCHLIST for gap-ups. Returns sorted list of dicts."""
    results = []
    logger.info("🔍 Scanning %d stocks for gap-ups...", len(WATCHLIST))

    for sym in WATCHLIST:
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(period="5d", interval="5m")
            if hist.empty: continue

            today_d = date.today()
            before_today = hist[hist.index.date < today_d]
            if before_today.empty: continue
            prev_close = before_today.iloc[-1]["Close"]

            today_bars = hist[hist.index.date == today_d]
            if today_bars.empty: continue

            latest = today_bars.iloc[-1]
            price = latest["Close"]
            gap = ((price - prev_close) / prev_close) * 100
            pre_vol = int(today_bars["Volume"].sum())

            if gap < MIN_GAP or pre_vol < MIN_PRE_VOL: continue
            if price < MIN_PRICE or price > MAX_PRICE: continue

            avg_vol = get_avg_vol(sym)
            rel_vol = pre_vol / avg_vol if avg_vol > 0 else 0

            results.append({
                "sym": sym, "gap": round(gap, 1), "vol": pre_vol,
                "rel_vol": round(rel_vol, 1), "price": round(price, 2),
                "avg_vol": int(avg_vol),
            })
            logger.info("  ✅ %s: +%.1f%%  vol=%d  rel=%.1fx  $%.2f",
                        sym, gap, pre_vol, rel_vol, price)

            await asyncio.sleep(0.05)  # rate limit
        except Exception as e:
            logger.debug("  %s skip: %s", sym, e)

    # Score: weighted combination of gap %, volume, relative volume
    for r in results:
        r["score"] = r["gap"] * 0.3 + (r["vol"] / 1e6) * 0.3 + r["rel_vol"] * 0.4

    results.sort(key=lambda x: x["score"], reverse=True)
    logger.info("   → %d qualifying (best: %s +%.1f%% score=%.1f)",
                len(results), results[0]["sym"] if results else "NONE",
                results[0]["gap"] if results else 0,
                results[0]["score"] if results else 0)
    return results


# ── Execution ───────────────────────────────────────────────────────────
async def execute(client, signal: dict):
    """Buy at open with trailing bracket stop."""
    sym = signal["sym"]
    price = signal["price"]
    gap = signal["gap"]
    score = signal["score"]

    qty = int(CAPITAL / price)
    if qty < 1:
        logger.warning("❌ Can't afford 1 share of %s ($%.2f)", sym, price)
        return False

    entry_cost = round(qty * price, 2)
    sl_price = round(price * (1 - HARD_SL/100), 2)
    trail_trigger = round(price * (1 + TRAIL_ACTIVATE/100), 2)
    sl_loss = round(entry_cost * HARD_SL / 100, 2)
    tp_target = round(entry_cost * 1.12, 2)  # 12% for reference

    logger.info("\n" + "=" * 55)
    logger.info("🎯 ENTERING: %s  gap=+%.1f%%  score=%.1f", sym, gap, score)
    logger.info("   %d shares × $%.2f = $%.2f (%.0f%% of capital)",
                qty, price, entry_cost, entry_cost/CAPITAL*100)
    logger.info("   Hard SL: $%.2f (-%.0f%%) = -$%.2f risk",
                sl_price, HARD_SL, sl_loss)
    logger.info("   Trail activates at +%.0f%% ($%.2f), trails at %.0f%% dist",
                TRAIL_ACTIVATE, trail_trigger, TRAIL_DIST)
    logger.info("   Target 12%%: $%.2f | 20%%: $%.2f | 50%%: $%.2f",
                tp_target, round(entry_cost*1.2,2), round(entry_cost*1.5,2))
    logger.info("=" * 55)

    if client is None:
        logger.info("🔷 SIM — logging trade only")
        # Log trade result to file for tracking
        trade_log = {
            "time": datetime.now(timezone.utc).isoformat(),
            "sym": sym, "qty": qty, "entry": price, "entry_cost": entry_cost,
            "sl_pct": HARD_SL, "trail_act": TRAIL_ACTIVATE, "trail_dist": TRAIL_DIST,
            "gap": gap, "score": score,
            "status": "entered",
        }
        with open("/tmp/gap_trades.json", "a") as f:
            f.write(json.dumps(trade_log) + "\n")
        return True

    try:
        # Market buy
        from alpaca.trading.requests import MarketOrderRequest, StopLimitOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

        buy = MarketOrderRequest(
            symbol=sym, qty=qty, side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY)
        client.submit_order(buy)
        logger.info("🟢 Buy order submitted: %d %s", qty, sym)

        # Bracket: TP limit at 12% + trailing stop (we'll manage the trail via polling)
        # Alpaca doesn't natively support trailing stop brackets, so we set
        # a wide limit TP and manage the trailing stop ourselves via polling
        tp_price = round(price * (1 + 50/100), 2)  # 50% wide TP (prevent capping)
        bracket = LimitOrderRequest(
            symbol=sym, qty=qty, side=OrderSide.SELL,
            limit_price=tp_price, time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLimitOrderRequest(
                stop_price=sl_price,
                limit_price=round(sl_price * 0.98, 2),
            ),
        )
        client.submit_order(bracket)
        logger.info("📋 Bracket: hard SL=$%.2f, wide TP=$%.2f (+50%%)", sl_price, tp_price)
        logger.info("📋 Trail will be managed by polling loop (checking every 30s)")
        return True

    except Exception as e:
        logger.error("❌ Order failed: %s", e)
        return False


async def trail_monitor(client, sym: str, qty: int, entry_price: float):
    """Poll position and trail the stop as price moves up."""
    logger.info("📡 Trail monitor started for %s (entry=$%.2f, qty=%d)", sym, entry_price, qty)
    peak = entry_price
    hard_sl = entry_price * (1 - HARD_SL/100)
    trail_triggered = False

    while True:
        try:
            pos = client.get_position(sym)
            cur = float(pos.current_price)
            gain = ((cur - entry_price) / entry_price) * 100

            if cur > peak:
                peak = cur
                if gain >= TRAIL_ACTIVATE and not trail_triggered:
                    trail_triggered = True
                    trail_stop = peak * (1 - TRAIL_DIST/100)
                    logger.info("🔔 Trail ACTIVATED at +%.1f%% ($%.2f). Trail stop=$%.2f",
                                gain, cur, trail_stop)

            if trail_triggered:
                trail_stop = peak * (1 - TRAIL_DIST/100)
                if cur <= trail_stop:
                    logger.info("🚩 Trail STOP hit at $%.2f (+%.1f%%)", cur,
                                ((cur - entry_price)/entry_price)*100)
                    client.close_position(sym)
                    logger.info("✅ Position closed via trailing stop")
                    return

            if gain <= -HARD_SL:
                logger.info("🚩 Hard SL should have caught this. Checking position...")
                if pos:  # SL didn't fire? close manually
                    client.close_position(sym)

        except Exception as e:
            # Position might already be closed
            if "position" in str(e).lower():
                logger.info("📌 Position %s closed (no longer held)", sym)
                return
            logger.debug("Trail monitor: %s", e)

        await asyncio.sleep(30)


# ── Main Loop ───────────────────────────────────────────────────────────
async def main():
    SIM = "--sim" in sys.argv
    client = None if SIM else get_alpaca_client()[0]
    if client is None and not SIM:
        logger.warning("⚠️ No Alpaca keys found. Run with --sim or export keys.")
        logger.warning("   export APCA_API_KEY_ID=... APCA_API_SECRET_KEY=...")
        SIM = True
    mode = "SIMULATION" if SIM else "ALPACA PAPER"
    logger.info("=" * 55)
    logger.info("🚀 GAP BOT v2 — Trailing Stop Edition")
    logger.info("   $%.0f → target $24/day | TP: trailing | SL: -%.0f%%", CAPITAL, HARD_SL)
    logger.info("   Trail: activate at +%.0f%%, trail %.0f%% below peak", TRAIL_ACTIVATE, TRAIL_DIST)
    logger.info("   Min gap: +%.0f%% | Min pre-vol: %d | Watchlist: %d", MIN_GAP, MIN_PRE_VOL, len(WATCHLIST))
    logger.info("   Mode: %s", mode)
    logger.info("=" * 55)

    today_state = {}  # tracks scan/executed/reported state per day

    while True:
        et = datetime.now(timezone.utc)
        # ET offset
        dst = date.today().month in range(3, 11)  # approximate
        et = et + timedelta(hours=-4 if dst else -5)
        h, m = et.hour, et.minute

        day_key = et.date().isoformat()
        if day_key not in today_state:
            today_state[day_key] = set()
        state = today_state[day_key]

        # 9:00 AM — initial scan
        if h == 9 and m == 0 and "scan" not in state:
            state.add("scan")
            logger.info("\n📡 9:00 AM SCAN at %s ET", et.strftime("%H:%M"))
            signals = await scan_premarket()
            if signals:
                state.add("has_signal")
                state.add(("signals", signals))
            else:
                logger.info("🚫 No setups. Try again tomorrow.")
                await asyncio.sleep(60)

        # 9:25 AM — re-scan and confirm
        if h == 9 and m == 25 and "rescan" not in state and "has_signal" in state:
            state.add("rescan")
            logger.info("\n📡 9:25 AM CONFIRMATION SCAN")
            signals = await scan_premarket()
            if signals:
                state.add(("signals", signals))
                best = signals[0]
                logger.info("\n🏆 BEST SETUP: %s +%.1f%% vol=%d score=%.1f $%.2f",
                            best["sym"], best["gap"], best["vol"], best["score"], best["price"])
            else:
                logger.info("🚫 Setup faded. No trade today.")

        # 9:31 AM — execute (90s after open)
        if h == 9 and m >= 31 and "executed" not in state and "has_signal" in state:
            # Check we're at least 90s after 9:30
            seconds_after_open = (h - 9) * 3600 + m * 60 - 30 * 60
            if seconds_after_open >= 90:
                state.add("executed")
                signals = None
                for s in state:
                    if isinstance(s, tuple) and s[0] == "signals":
                        signals = s[1]
                        break
                if signals and len(signals) > 0:
                    best = signals[0]
                    ok = await execute(client, best)
                    if ok and client:
                        # Start trail monitor in background
                        qty = int(CAPITAL / best["price"])
                        asyncio.create_task(trail_monitor(client, best["sym"], qty, best["price"]))
                        logger.info("⏳ Trail monitor running. Checking back at 4 PM.")

        # 4:00 PM — report
        if h >= 16 and "reported" not in state:
            state.add("reported")
            if client:
                try:
                    acct = client.get_account()
                    pnl = float(acct.equity) - CAPITAL
                    logger.info("\n📊 END OF DAY: $%.2f | Day P&L: $%.2f (%.1f%%)",
                                float(acct.equity), pnl, pnl/CAPITAL*100)
                    pos = client.get_all_positions()
                    if pos:
                        for p in pos:
                            logger.info("   Holding: %s %s shares P&L=$%.2f",
                                        p.symbol, p.qty, float(p.unrealized_pl))
                    else:
                        logger.info("   No positions (flat)")
                except Exception as e:
                    logger.info("\n📊 Day complete (account check: %s)", e)
            logger.info("⏰ Next scan tomorrow 9:00 AM ET.")
            await asyncio.sleep(3600)

        # Clean old days
        for k in list(today_state.keys()):
            if k < day_key:
                del today_state[k]

        await asyncio.sleep(15)


if __name__ == "__main__":
    asyncio.run(main())
