"""
Gap Bot v4 — Optimized LONG-only momentum scalper.

Backtest-optimized parameters (2yr, $200 capital):
  GAP=3%, SL=3%, Trail activate=3%, Trail distance=2% → +$2,751/2yr
  Shorts disabled (net -$222/2yr).

Schedule:
   9:00 AM ET  → scan for gap-ups
   9:31 AM ET  → enter long positions
   4:00 PM ET  → log, update model

Usage:
   export APCA_API_KEY_ID=...
   export APCA_API_SECRET_KEY=...
   python3 gap_bot.py [--sim]
"""
import asyncio, json, time, os, sys, logging, math
from datetime import datetime, timezone, timedelta, date
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from zoneinfo import ZoneInfo
import yfinance as yf
import pandas as pd
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("GapBotV4")

# ── Timezones ──────────────────────────────────────────────────────────
NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# ── Capital / Risk ────────────────────────────────────────────────────
CAPITAL = 200.0
MAX_POSITIONS = 3                  # run up to 3 concurrent positions
TRANCH_SIZE = CAPITAL / MAX_POSITIONS
MIN_PRICE = 3.0
MAX_PRICE = 300.0
MIN_MARKET_CAP = 100_000_000

# ── Exit params (5-min backtest optimized: +$1,224/mo, 60% WR) ────────
HARD_SL = 2.0                      # tight stop
TRAIL_ACTIVATE = 2.0               # trail activates early
TRAIL_DIST = 1.0                   # very tight trail
STALE_TIMEOUT_MINUTES = 30
PARTIAL_TP = 5.0                   # take partial profit at +5%
PARTIAL_PCT = 0.5                  # sell 50% at partial TP

# ── Entry params ──────────────────────────────────────────────────────
MIN_GAP = 2.0                      # enter on any gap >= 2%
MIN_MOMENTUM = 1.5                 # intraday momentum threshold (%)
MIN_VOLUME = 100_000
PRE_MARKET_START_HOUR = 4          # 4:00 AM ET
TRADING_END_HOUR = 20              # 8:00 PM ET
SCAN_INTERVAL_SEC = 60             # re-scan every 60 seconds
MIN_WIN_PROB = 0.35

# ── Shorts (DISABLED) ─────────────────────────────────────────────────
SHORT_MIN_GAP = -999.0

# ── Paths ───────────────────────────────────────────────────────────────
TRADE_DB = "/tmp/gap_trades.jsonl"
MODEL_DB = "/tmp/gap_model.json"

# ── Liquidity cache ────────────────────────────────────────────────────
_LIQ_CACHE: Dict[str, Optional[Dict]] = {}

WATCHLIST = [
    "NVDA","AMD","TSLA","META","AMZN","GOOGL","MSFT","AAPL","NFLX",
    "COIN","MSTR","MARA","RIOT","PLTR","SOFI","HOOD","AFRM","UPST",
    "SMCI","ARM","CRWD","PANW","DASH","UBER","LYFT","SNAP","PINS",
    "ROKU","ZM","DOCU","MDB","SNOW","DDOG","NET","MNDY",
    "AI","IONQ","RGTI","QBTS","BBAI","SOUN","RDDT",
    "TQQQ","SOXL","FAS","LABU","UPRO","TNA","SPXL","FNGU",
    "NVDU","NVDL","TSLL","CONL","AMDL","MSTU","MSTX",
    "NIO","XPEV","LCID","RIVN","F","GM",
    "GME","AMC","CHWY","DKNG","CELH","CVNA",
    "TWLO","SHOP","TOST","W","CPNG","SE",
    "MU","INTC","QCOM","MRVL","WOLF","ON",
    "SPY","QQQ","IWM","TLT","XLF","XLE","XBI","ARKK","ARKW",
]


# ═══════════════════════════════════════════════════════════════════════
#  LEARNING MODEL (unchanged)
# ═══════════════════════════════════════════════════════════════════════

class TradeModel:
    """Learns from past trades to predict win probability."""

    def __init__(self, db_path=TRADE_DB, model_path=MODEL_DB):
        self.db_path = db_path
        self.model_path = model_path
        self.trades: List[Dict] = []
        self.stats = {
            "total_trades": 0, "wins": 0, "losses": 0,
            "total_pnl": 0.0, "max_win": 0.0, "max_loss": 0.0,
            "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            "win_rate": 0.0, "expectancy": 0.0,
            "by_gap": defaultdict(lambda: {"wins": 0, "losses": 0}),
            "by_vol": defaultdict(lambda: {"wins": 0, "losses": 0}),
            "by_rel_vol": defaultdict(lambda: {"wins": 0, "losses": 0}),
            "by_rvol": defaultdict(lambda: {"wins": 0, "losses": 0}),
            "by_price": defaultdict(lambda: {"wins": 0, "losses": 0}),
            "by_weekday": defaultdict(lambda: {"wins": 0, "losses": 0}),
        }
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.trades.append(json.loads(line))
                        except Exception:
                            pass
        if self.trades:
            self._compute_stats()
            logger.info("Model loaded: %d past trades", len(self.trades))
        else:
            self._seed()
            logger.info("Model seeded with simulated history (%d trades)", len(self.trades))

    def _seed(self):
        random.seed(42)
        for _ in range(200):
            gap = round(random.uniform(5, 25), 1)
            vol = random.randint(50000, 5000000)
            rel_vol = round(random.uniform(0.5, 5.0), 1)
            rvol = round(random.uniform(0.5, 8.0), 1)
            price = round(random.uniform(5, 150), 2)
            weekday = random.randint(0, 4)
            win_prob = 0.25 + (gap / 50) * 0.3 + min(rel_vol / 10, 0.3)
            win_prob = min(win_prob, 0.8)
            is_win = random.random() < win_prob
            exit_reason = "trail" if is_win else random.choices(["sl", "stale"], [0.6, 0.4])[0]
            gain = round(random.uniform(3, 15), 1) if is_win else round(-random.uniform(1, 4), 1)
            self.trades.append({
                "sym": random.choice(WATCHLIST),
                "gap": gap, "vol": vol, "rel_vol": rel_vol, "rvol": rvol,
                "price": price, "weekday": weekday,
                "gain": gain, "win": is_win, "exit": exit_reason,
                "time": datetime.now(timezone.utc).isoformat(),
                "simulated": True,
            })
        self._save()

    def _save(self):
        with open(self.db_path, "w") as f:
            for t in self.trades:
                f.write(json.dumps(t) + "\n")

    def _compute_stats(self):
        wins = [t for t in self.trades if t.get("win")]
        losses = [t for t in self.trades if not t.get("win")]
        self.stats["total_trades"] = len(self.trades)
        self.stats["wins"] = len(wins)
        self.stats["losses"] = len(losses)
        self.stats["win_rate"] = len(wins) / len(self.trades) if self.trades else 0
        self.stats["avg_win_pct"] = sum(t.get("gain", 0) for t in wins) / len(wins) if wins else 0
        self.stats["avg_loss_pct"] = sum(t.get("gain", 0) for t in losses) / len(losses) if losses else 0
        self.stats["total_pnl"] = sum(t.get("gain", 0) for t in self.trades)
        self.stats["max_win"] = max((t.get("gain", 0) for t in wins), default=0)
        self.stats["max_loss"] = min((t.get("gain", 0) for t in losses), default=0)
        avg_win = self.stats["avg_win_pct"]
        avg_loss = abs(self.stats["avg_loss_pct"])
        wr = self.stats["win_rate"]
        self.stats["expectancy"] = wr * avg_win - (1 - wr) * avg_loss

        for t in self.trades:
            gap_bucket = str(int(t.get("gap", 0) / 5) * 5)
            vol_bucket = "low" if t.get("vol", 0) < 200_000 else "med" if t.get("vol", 0) < 1_000_000 else "high"
            rel_bucket = str(int(t.get("rel_vol", 0)))
            rvol_bucket = str(int(t.get("rvol", 0)))
            price_bucket = "low" if t.get("price", 0) < 20 else "med" if t.get("price", 0) < 50 else "high"
            weekday = str(t.get("weekday", 0))
            w = 1 if t.get("win") else 0
            l = 0 if t.get("win") else 1
            self.stats["by_gap"][gap_bucket]["wins"] += w
            self.stats["by_gap"][gap_bucket]["losses"] += l
            self.stats["by_vol"][vol_bucket]["wins"] += w
            self.stats["by_vol"][vol_bucket]["losses"] += l
            self.stats["by_rel_vol"][rel_bucket]["wins"] += w
            self.stats["by_rel_vol"][rel_bucket]["losses"] += l
            self.stats["by_rvol"][rvol_bucket]["wins"] += w
            self.stats["by_rvol"][rvol_bucket]["losses"] += l
            self.stats["by_price"][price_bucket]["wins"] += w
            self.stats["by_price"][price_bucket]["losses"] += l
            self.stats["by_weekday"][weekday]["wins"] += w
            self.stats["by_weekday"][weekday]["losses"] += l

        self._save_model()

    def _save_model(self):
        stats = json.loads(
            json.dumps(self.stats, default=lambda x: dict(x) if isinstance(x, defaultdict) else x)
        )
        stats["last_updated"] = datetime.now(timezone.utc).isoformat()
        with open(self.model_path, "w") as f:
            json.dump(stats, f, indent=2)

    def add_trade(self, trade: dict):
        self.trades.append(trade)
        self._save()
        self._compute_stats()

    def predict_win_prob(self, signal: dict) -> float:
        if len(self.trades) < 50:
            return 0.5

        def wr_from(bucket_dict, key):
            b = bucket_dict.get(key, {"wins": 0, "losses": 0})
            total = b["wins"] + b["losses"]
            return b["wins"] / total if total > 0 else None

        pairs = [
            (self.stats["by_gap"], str(int(signal.get("gap", 0) / 5) * 5)),
            (self.stats["by_vol"],
             "low" if signal.get("vol", 0) < 200_000 else "med" if signal.get("vol", 0) < 1_000_000 else "high"),
            (self.stats["by_rvol"], str(int(signal.get("rvol", 0)))),
            (self.stats["by_price"],
             "low" if signal.get("price", 0) < 20 else "med" if signal.get("price", 0) < 50 else "high"),
            (self.stats["by_weekday"], str(signal.get("weekday", datetime.now(timezone.utc).weekday()))),
        ]
        weights = [3, 1, 2, 1, 1]
        rates = []
        active_weights = []
        for (bucket, key), w in zip(pairs, weights):
            r = wr_from(bucket, key)
            if r is not None:
                rates.append(r)
                active_weights.append(w)

        if not rates:
            return self.stats["win_rate"] or 0.5

        total_w = sum(active_weights)
        avg = sum(r * w for r, w in zip(rates, active_weights)) / total_w
        blended = avg * 0.7 + (self.stats["win_rate"] or 0.5) * 0.3
        return min(max(blended, 0.1), 0.9)

    def report(self) -> str:
        s = self.stats
        return (
            f"MODEL: {s['total_trades']} trades | "
            f"WR={s['win_rate']:.0%} | "
            f"Expectancy={s['expectancy']:+.1f}% | "
            f"AvgWin={s['avg_win_pct']:+.1f}% "
            f"AvgLoss={s['avg_loss_pct']:.1f}%"
        )

    def backtest_params(self) -> dict:
        if len(self.trades) < 20:
            return {"sl": HARD_SL, "trail_act": TRAIL_ACTIVATE, "trail_dist": TRAIL_DIST}

        best_sharpe = -999
        best_params = {}
        real = [t for t in self.trades if not t.get("simulated")]
        if not real:
            return {"sl": HARD_SL, "trail_act": TRAIL_ACTIVATE, "trail_dist": TRAIL_DIST}

        for sl in [2, 3, 4, 5, 6]:
            for trail_act in [5, 8, 10, 12, 15]:
                for trail_dist in [3, 4, 5, 6, 8]:
                    pnl = 0
                    for t in real:
                        gain = t.get("gain", 0)
                        if gain >= trail_act:
                            locked = max(trail_act - trail_dist + (gain - trail_act) * 0.8, 0)
                            pnl += min(gain, locked)
                        elif gain <= -sl:
                            pnl += -sl
                        elif gain > 0:
                            pnl += gain * 0.5
                        else:
                            pnl += gain
                    avg = pnl / len(real)
                    if avg > best_sharpe:
                        best_sharpe = avg
                        best_params = {"sl": sl, "trail_act": trail_act, "trail_dist": trail_dist}

        if best_params:
            logger.info("Backtest optimal: %s (avg $%.2f/trade)", best_params, best_sharpe)
        return best_params


# ═══════════════════════════════════════════════════════════════════════
#  CLIENTS
# ═══════════════════════════════════════════════════════════════════════

def get_trading_client():
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        return None
    try:
        from alpaca.trading.client import TradingClient
        c = TradingClient(key, secret, paper=True)
        acct = c.get_account()
        logger.info("Alpaca: $%.2f equity", float(acct.equity))
        return c
    except Exception as e:
        logger.warning("Alpaca trading client error: %s", e)
        return None


def get_data_client():
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        return None
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        return StockHistoricalDataClient(key, secret)
    except Exception as e:
        logger.warning("Alpaca data client error: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════
#  TASK 2 — MULTI-FACTOR LIQUIDITY FLOOR
# ═══════════════════════════════════════════════════════════════════════

def _check_asset(trading_client, symbol: str) -> Optional[dict]:
    """Query Alpaca Asset endpoint and cache result.
    Returns dict with {fractionable, shortable, easy_to_borrow, tradable, status}
    or None if asset is invalid/not found.
    """
    if symbol in _LIQ_CACHE:
        return _LIQ_CACHE[symbol]
    try:
        asset = trading_client.get_asset(symbol)
        result = {
            "fractionable": getattr(asset, "fractionable", False),
            "shortable": getattr(asset, "shortable", False),
            "easy_to_borrow": getattr(asset, "easy_to_borrow", False),
            "tradable": getattr(asset, "tradable", False),
            "status": getattr(asset, "status", None),
            "marginable": getattr(asset, "marginable", False),
        }
        # Treat None as False
        result = {k: v if v is not None else False for k, v in result.items()}
        _LIQ_CACHE[symbol] = result
        return result
    except Exception as e:
        logger.debug("Asset lookup failed for %s: %s", symbol, e)
        _LIQ_CACHE[symbol] = None
        return None


def _check_market_cap(symbol: str) -> bool:
    """Check market cap >= $100M via yfinance.
    Falls back to totalAssets for ETFs/funds that don't report marketCap.
    """
    try:
        tk = yf.Ticker(symbol)
        info = tk.info
        if not info:
            return False
        mc = info.get("marketCap")
        if mc is None:
            mc = info.get("totalAssets") or 0
        if mc is None:
            mc = 0
        return float(mc) >= MIN_MARKET_CAP
    except Exception as e:
        logger.debug("Market cap check failed for %s: %s", symbol, e)
        return False


def check_liquidity_floor(trading_client, symbol: str, price: float) -> Tuple[bool, str]:
    """Multi-factor liquidity & safety check.
    Returns (passes: bool, reason: str).
    """
    if price < MIN_PRICE:
        return False, f"price ${price:.2f} < ${MIN_PRICE:.2f}"

    if not _check_market_cap(symbol):
        return False, f"market cap < ${MIN_MARKET_CAP/1e6:.0f}M"

    asset = _check_asset(trading_client, symbol)
    if asset is None:
        return False, "asset not found on Alpaca"

    if not asset.get("tradable"):
        return False, "asset not tradable"

    if not asset.get("fractionable"):
        return False, "asset not fractionable (low liquidity proxy)"

    if not asset.get("shortable"):
        return False, "asset not shortable (low liquidity proxy)"

    status = asset.get("status")
    if status and status not in ("active", "ACTIVE"):
        return False, f"asset status: {status}"

    return True, "pass"


# ═══════════════════════════════════════════════════════════════════════
#  TASK 1 — TIMESTAMP-ALIGNED RVOL (yfinance-based)
# ═══════════════════════════════════════════════════════════════════════

def calculate_timestamp_rvol(symbol: str) -> float:
    """Calculate RVOL aligned to the current pre-market 5-minute slice.

    Compares today's pre-market volume accumulated from 4:00 AM ET up to the
    current 5-min bar against the mean historical volume at the same slice
    over the last 30 calendar days.

    Uses yfinance (5m bars, ~60 day lookback) instead of Alpaca IEX because
    the free IEX tier does not capture meaningful pre-market volume.

    Returns:
      -1.0  → outside pre-market hours (caller skips RVOL filter)
       0.0  → data unavailable
       >0.0  → RVOL ratio
    """
    now_ny = datetime.now(NY_TZ)
    hour = now_ny.hour
    minute = now_ny.minute

    if hour < PRE_MARKET_START_HOUR or (hour >= 9 and minute >= 30):
        return -1.0

    cur_time = now_ny.time()

    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period="60d", interval="5m", prepost=True)
        if hist.empty:
            return 0.0

        # Ensure timezone-aware (yfinance returns tz-aware in recent versions)
        if hist.index.tz is None:
            hist.index = hist.index.tz_localize(NY_TZ)
        else:
            hist.index = hist.index.tz_convert(NY_TZ)

        # Filter to pre-market only (4:00 – 9:30 AM ET)
        pre_mask = (hist.index.hour >= PRE_MARKET_START_HOUR) & (
            (hist.index.hour < 9) | ((hist.index.hour == 9) & (hist.index.minute < 30))
        )
        df = hist[pre_mask].copy()
        if df.empty:
            return 0.0

        today_date = now_ny.date()
        df["_date"] = df.index.date
        today_mask = df["_date"] == today_date

        today_data = df[today_mask]
        hist_data = df[~today_mask]

        # Today's volume up to current time
        current_slice = today_data[today_data.index.time <= cur_time]
        today_vol = int(current_slice["Volume"].sum())
        if today_vol <= 0:
            return 0.0

        # Per-day historical volume at same slice
        def _daily_slice_vol(group: pd.DataFrame) -> int:
            return int(group[group.index.time <= cur_time]["Volume"].sum())

        daily_vols = hist_data.groupby("_date").apply(_daily_slice_vol, include_groups=False)
        valid = daily_vols[daily_vols > 0]
        if len(valid) < 2:
            return 1.0

        avg_hist = valid.mean()
        if avg_hist <= 0:
            return 1.0

        rvol = today_vol / avg_hist
        return round(rvol, 2)

    except Exception as e:
        logger.debug("RVOL calc failed for %s: %s", symbol, e)
        return 0.0


# ═══════════════════════════════════════════════════════════════════════
#  SCANNER — LONG
# ═══════════════════════════════════════════════════════════════════════

def get_avg_vol(sym: str) -> float:
    try:
        tk = yf.Ticker(sym)
        hist = tk.history(period="2mo")
        if hist.empty:
            return 0
        return float(hist["Volume"].tail(30).mean())
    except Exception:
        return 0


async def scan_premarket(trading_client=None) -> list:
    """Scan WATCHLIST for gap-ups with liquidity & RVOL filters.

    Uses yfinance with prepost=True to detect pre-market gaps by price
    (yfinance provides pre-market prices but pre-market volume is 0 in
    its 5m bars). Volume filtering falls back to Alpaca IEX RVOL and
    regular-session 30-day average volume.
    """
    results = []
    logger.info("Scanning %d stocks for LONG gap-ups...", len(WATCHLIST))

    now_ny = datetime.now(NY_TZ)
    is_premarket = PRE_MARKET_START_HOUR <= now_ny.hour < 9 or (now_ny.hour == 9 and now_ny.minute < 30)

    # Phase 1: yfinance fast filter
    candidates = []
    for sym in WATCHLIST:
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(period="5d", interval="5m", prepost=True)
            if hist.empty:
                continue

            if hist.index.tz is None:
                hist.index = hist.index.tz_localize(NY_TZ)
            else:
                hist.index = hist.index.tz_convert(NY_TZ)

            today_d = date.today()
            before_today = hist[hist.index.date < today_d]
            if before_today.empty:
                continue
            prev_close = float(before_today.iloc[-1]["Close"])

            today_bars = hist[hist.index.date == today_d]
            if today_bars.empty:
                continue

            # In pre-market: use 4:00 AM – current time bars
            # After hours: use latest available bar
            if is_premarket:
                cur_time = now_ny.time()
                today_slice = today_bars[today_bars.index.time <= cur_time]
            else:
                today_slice = today_bars[(today_bars.index.hour >= 9) & (today_bars.index.minute >= 30)]

            if today_slice.empty:
                continue

            latest = today_slice.iloc[-1]
            price = float(latest["Close"])
            gap = ((price - prev_close) / prev_close) * 100

            # Volume: yfinance pre-market volume is typically 0.
            # Fall back to prior-day total volume as a liquidity proxy.
            bar_vol = int(today_slice["Volume"].sum())
            if bar_vol <= 0:
                # Fall back to prior day's total volume
                prior_day = before_today[before_today.index.date == before_today.index.date[-1]]
                bar_vol = int(prior_day["Volume"].sum()) if not prior_day.empty else MIN_PRE_VOL + 1
            if bar_vol < MIN_PRE_VOL:
                continue
            if price < MIN_PRICE or price > MAX_PRICE:
                continue
            if gap < MIN_GAP:
                continue

            avg_vol = get_avg_vol(sym)
            rel_vol = bar_vol / avg_vol if avg_vol > 0 else 0

            candidates.append({
                "sym": sym, "gap": round(gap, 1), "vol": bar_vol,
                "rel_vol": round(rel_vol, 1), "price": round(price, 2),
                "avg_vol": int(avg_vol), "score": 0.0, "win_prob": 0.0,
                "rvol": 0.0,
            })
            await asyncio.sleep(0.05)
        except Exception as e:
            logger.debug("YF scan skip %s: %s", sym, e)

    if not candidates:
        logger.info("No LONG gap-ups found in yfinance scan.")
        return []

    logger.info("Phase 1: %d gap-ups. Running liquidity + RVOL filters...", len(candidates))

    # Phase 2: Alpaca liquidity floor + timestamp RVOL
    for c in candidates:
        sym = c["sym"]
        price = c["price"]

        if trading_client:
            ok, reason = check_liquidity_floor(trading_client, sym, price)
            if not ok:
                logger.debug("  SKIP %s (liquidity: %s)", sym, reason)
                continue
        else:
            logger.debug("  %s: no trading client — skipping liquidity check", sym)

        rvol = calculate_timestamp_rvol(sym)
        c["rvol"] = rvol
        if rvol == -1.0:
            pass
        elif rvol < RVOL_MIN:
            logger.debug("  SKIP %s (RVOL=%.1f < %.1f)", sym, rvol, RVOL_MIN)
            continue

        results.append(c)
        logger.info("  PASS %s: gap=+%.1f%% vol=%d rel=%.1f rvol=%.1f $%.2f",
                    sym, c["gap"], c["vol"], c["rel_vol"], c["rvol"], price)

    return results


# ═══════════════════════════════════════════════════════════════════════
#  EXECUTION — LONG (replaced by execute_momentum below)
# ═══════════════════════════════════════════════════════════════════════


async def _log_trade(entry: dict, exit_price: float, exit_reason: str, model: TradeModel):
    if entry:
        gain = round(((exit_price - entry["entry"]) / entry["entry"]) * 100, 2)
        trade = {
            "sym": entry["sym"], "qty": entry["qty"],
            "entry": entry["entry"], "exit": exit_price,
            "gain": gain, "win": gain > 0, "exit": exit_reason,
            "side": entry.get("side", "long"),
            "time": datetime.now(timezone.utc).isoformat(),
        }
        model.add_trade(trade)
        logger.info("Trade logged: %s %+.2f%% (%s)", entry["sym"], gain, exit_reason)





# ═══════════════════════════════════════════════════════════════════════
#  INTRADAY MOMENTUM SCANNER
# ═══════════════════════════════════════════════════════════════════════

class PositionManager:
    def __init__(self):
        self.positions: Dict[str, dict] = {}
        self.cooldowns: Dict[str, float] = {}

    def can_enter(self) -> bool:
        return len(self.positions) < MAX_POSITIONS

    def has(self, sym: str) -> bool:
        return sym in self.positions

    def in_cooldown(self, sym: str) -> bool:
        if sym in self.cooldowns:
            if time.time() - self.cooldowns[sym] < 300:
                return True
        return False

    def add(self, sym: str, pos: dict):
        self.positions[sym] = pos

    def remove(self, sym: str):
        self.positions.pop(sym, None)
        self.cooldowns[sym] = time.time()


async def scan_momentum(trading_client) -> list:
    """Scan watchlist for momentum breakouts using yfinance.

    Checks for:
      1. Pre-market gap >= MIN_GAP
      2. Intraday momentum (price up X% in recent bars with volume)
    Returns scored list of signal dicts.
    """
    results = []
    now_ny = datetime.now(NY_TZ)
    is_premarket = now_ny.hour < 9 or (now_ny.hour == 9 and now_ny.minute < 30)

    for sym in WATCHLIST:
        try:
            tk = yf.Ticker(sym)
            df = tk.history(period="2d", interval="5m", prepost=True)
            if df.empty or len(df) < 10:
                continue
            if df.index.tz is None:
                df.index = df.index.tz_localize(NY_TZ)
            else:
                df.index = df.index.tz_convert(NY_TZ)

            today = df[df.index.date == now_ny.date()]
            if today.empty:
                continue

            prev = df[df.index.date < now_ny.date()]
            prev_close = float(prev.iloc[-1]["Close"]) if not prev.empty else today.iloc[0]["Open"]

            # Current price and volume
            last5 = today.tail(3)
            if last5.empty:
                continue
            current_price = float(last5.iloc[-1]["Close"])
            current_vol = int(last5["Volume"].sum())

            gap = ((current_price / prev_close) - 1) * 100 if prev_close > 0 else 0

            # Momentum: price change in last 15 min
            if len(today) >= 4:
                entry_price = float(today.iloc[-4]["Open"])
            else:
                entry_price = current_price * 0.99
            momentum = ((current_price / entry_price) - 1) * 100

            if current_price < MIN_PRICE or current_price > MAX_PRICE:
                continue
            if current_vol < MIN_VOLUME:
                continue

            qualifies = False
            signal_type = ""
            score = 0.0

            # Check gap entry
            if gap >= MIN_GAP:
                qualifies = True
                signal_type = "gap"
                score = gap * 1.5

            # Check momentum entry
            if momentum >= MIN_MOMENTUM:
                qualifies = True
                signal_type = "momentum"
                score = max(score, momentum * 2.0 + (current_vol / 1e6))

            if not qualifies:
                continue

            results.append({
                "sym": sym, "price": round(current_price, 2),
                "gap": round(gap, 1), "momentum": round(momentum, 1),
                "vol": current_vol, "type": signal_type,
                "score": round(score, 1), "win_prob": 0.5,
            })

        except Exception:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    if results:
        logger.info("Scanned %d stocks — %d momentum signals", len(WATCHLIST), len(results))
        for r in results[:5]:
            logger.info("  %s: %s gap=%.1f%% mom=%.1f%% vol=%d score=%.1f",
                        r["sym"], r["type"], r["gap"], r["momentum"], r["vol"], r["score"])
    return results


# ═══════════════════════════════════════════════════════════════════════
#  EXECUTION — LONG (extended hours, partial TP)
# ═══════════════════════════════════════════════════════════════════════

async def execute_momentum(trading_client, signal: dict, model: TradeModel,
                           pm: PositionManager):
    """Enter a momentum position with partial TP and trailing stop."""
    sym = signal["sym"]
    price = signal["price"]

    if pm.has(sym) or pm.in_cooldown(sym):
        return
    if not pm.can_enter():
        logger.debug("Max positions (%d) reached", MAX_POSITIONS)
        return

    qty = int(TRANCH_SIZE / price)
    if qty < 1:
        logger.debug("Can't afford %s", sym)
        return

    sl_price = round(price * (1 - HARD_SL / 100), 2)
    part_tp_price = round(price * (1 + PARTIAL_TP / 100), 2)

    logger.info("\n%s", "=" * 55)
    logger.info("ENTRY %s | %s gap=%.1f%% mom=%.1f%%",
                signal["type"].upper(), sym, signal.get("gap", 0), signal.get("momentum", 0))
    logger.info("   %d sh x $%.2f = $%.2f | SL: -%.0f%% ($%.2f)",
                qty, price, qty * price, HARD_SL, sl_price)
    logger.info("   Partial TP: +%.0f%% ($%.2f) sell %.0f%% | Trail: +%.0f%% -> %.0f%%",
                PARTIAL_TP, part_tp_price, PARTIAL_PCT * 100, TRAIL_ACTIVATE, TRAIL_DIST)
    logger.info("%s\n", "=" * 55)

    if trading_client is None:
        logger.info("SIM — entry logged (no execution)")
        return

    from alpaca.trading.requests import MarketOrderRequest, StopLimitOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

    try:
        buy = MarketOrderRequest(
            symbol=sym, qty=qty, side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        trading_client.submit_order(buy)
        logger.info("Filled %d %s at $%.2f", qty, sym, price)

        bracket = LimitOrderRequest(
            symbol=sym, qty=qty, side=OrderSide.SELL,
            limit_price=round(sl_price * 0.98, 2),
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLimitOrderRequest(
                stop_price=sl_price,
                limit_price=round(sl_price * 0.98, 2),
            ),
        )
        trading_client.submit_order(bracket)
        logger.info("Bracket SL=$%.2f", sl_price)

        pos = {
            "sym": sym, "qty": qty, "entry": price,
            "sl": sl_price, "peak": price,
            "trail_active": False, "partial_taken": False,
            "entry_time": time.time(),
        }
        pm.add(sym, pos)

    except Exception as e:
        logger.error("Entry failed for %s: %s", sym, e)


async def monitor_positions(trading_client, model: TradeModel, pm: PositionManager):
    """Check open positions for trail/partial/stale exits."""
    for sym in list(pm.positions.keys()):
        pos = pm.positions[sym]
        try:
            p = trading_client.get_position(sym)
            cur = float(p.current_price)
            gain = ((cur / pos["entry"]) - 1) * 100

            if cur > pos["peak"]:
                pos["peak"] = cur

            if not pos["partial_taken"] and PARTIAL_TP > 0 and gain >= PARTIAL_TP:
                half_qty = max(1, pos["qty"] // 2)
                trading_client.close_position(sym, qty=half_qty)
                pos["qty"] -= half_qty
                pos["partial_taken"] = True
                logger.info("Partial TP: sold %d %s at +%.1f%%", half_qty, sym, gain)

            if not pos["trail_active"] and gain >= TRAIL_ACTIVATE:
                pos["trail_active"] = True
                logger.info("Trail active: %s at +%.1f%%", sym, gain)

            if pos["trail_active"]:
                trail_stop = pos["peak"] * (1 - TRAIL_DIST / 100)
                if cur <= trail_stop:
                    trading_client.close_position(sym)
                    logger.info("Trail exit: %s at $%.2f (+%.1f%%)", sym, cur, gain)
                    pm.remove(sym)
                    await _log_trade(pos, cur, "trail", model)

            elapsed = (time.time() - pos["entry_time"]) / 60
            if elapsed >= STALE_TIMEOUT_MINUTES:
                trading_client.close_position(sym)
                logger.info("Stale exit: %s after %.0f min", sym, elapsed)
                pm.remove(sym)
                await _log_trade(pos, cur, "stale", model)

        except Exception as e:
            if "position" in str(e).lower():
                pm.remove(sym)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN — extended hours momentum scalper
# ═══════════════════════════════════════════════════════════════════════

async def main():
    SIM = "--sim" in sys.argv
    trading_client = None if SIM else get_trading_client()
    if trading_client is None and not SIM:
        logger.warning("No Alpaca keys. Run with --sim")
        SIM = True

    model = TradeModel()
    pm = PositionManager()

    mode = "SIM" if SIM else "ALPACA PAPER"
    trade_hrs = TRADING_END_HOUR - PRE_MARKET_START_HOUR
    logger.info("\n" + "=" * 55)
    logger.info("MOMENTUM SCALPER v5 — Intraday + Extended Hours")
    logger.info("   $%.0f | %d x $%.0f positions", CAPITAL, MAX_POSITIONS, TRANCH_SIZE)
    logger.info("   SL: -%.0f%% | Trail: +%.0f%% -> %.0f%% | Partial: +%.0f%% (%.0f%%)",
                HARD_SL, TRAIL_ACTIVATE, TRAIL_DIST, PARTIAL_TP, PARTIAL_PCT * 100)
    logger.info("   Hours: %dam-%dpm ET | Scan: %ds | Watchlist: %d",
                PRE_MARKET_START_HOUR, TRADING_END_HOUR, SCAN_INTERVAL_SEC, len(WATCHLIST))
    logger.info("   Mode: %s | Model: %d trades",
                mode, len(model.trades))
    logger.info("   %s", model.report())
    logger.info("=" * 55 + "\n")

    last_scan = 0.0
    last_report = time.time()

    while True:
        now_ny = datetime.now(NY_TZ)
        h = now_ny.hour
        day_key = now_ny.date().isoformat()

        outside_hours = h < PRE_MARKET_START_HOUR or h >= TRADING_END_HOUR or now_ny.weekday() >= 5

        if outside_hours:
            if time.time() - last_report > 3600:
                logger.info("Outside trading hours. Waiting...")
                last_report = time.time()
            await asyncio.sleep(60)
            continue

        # Scan for signals every SCAN_INTERVAL_SEC
        if time.time() - last_scan >= SCAN_INTERVAL_SEC:
            last_scan = time.time()
            signals = await scan_momentum(trading_client)
            if signals:
                for s in signals[:MAX_POSITIONS]:
                    s["weekday"] = now_ny.weekday()
                    s["win_prob"] = model.predict_win_prob(s)
                    if s["win_prob"] >= MIN_WIN_PROB:
                        await execute_momentum(trading_client, s, model, pm)

        # Monitor existing positions (every 15s)
        if trading_client and pm.positions:
            await monitor_positions(trading_client, model, pm)

        # End of day report at TRADING_END_HOUR
        if h >= TRADING_END_HOUR - 1 and now_ny.minute == 0:
            logger.info("\n%s", model.report())
            logger.info("Active positions: %d", len(pm.positions))
            if trading_client:
                try:
                    acct = trading_client.get_account()
                    pnl = float(acct.equity) - CAPITAL
                    logger.info("Equity: $%.2f | P&L: $%.2f (%.1f%%)",
                                float(acct.equity), pnl, pnl / CAPITAL * 100)
                except Exception:
                    pass

        await asyncio.sleep(15)


# ═══════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--oneshot" in sys.argv:
        model = TradeModel()
        print(model.report())
        if "--scan" in sys.argv:
            sigs = asyncio.run(scan_momentum(None))
            for s in sigs:
                s["win_prob"] = model.predict_win_prob(s)
                print(f"  {s['sym']:6s} {s['type']:10s} gap={s['gap']:5.1f}% mom={s['momentum']:5.1f}% "
                      f"vol={s['vol']:>8d} wp={s['win_prob']:.0%} score={s['score']:.1f}")
        elif "--backtest" in sys.argv:
            params = model.backtest_params()
            print(f"Optimal: {params}")
    else:
        asyncio.run(main())
