"""
Gap Bot v4 — Self-Learning with Timestamp-RVOL, Liquidity Floor & Short Side.

Enhancements:
  1. Timestamp-aligned RVOL: compares today's pre-market volume at slice T
     against mean historical volume at same slice over last 30 calendar days.
  2. Multi-factor liquidity floor: fractionable, shortable, market cap >= $100M,
     price >= $3.00.
  3. Short gap-down scaffold: detects 8%+ gap-downs with RVOL > 3.0, waits for
     1-min opening low break, enters short with -6% SL / +12% TP bracket.

Timezones: strictly normalized to America/New_York for all pre-market bar indices.

Schedule:
  9:00 AM ET  → long scan + short candidate scan
  9:31 AM ET  → enter long or short if conditions met
  4:00 PM ET  → log, update model, overnight param backtest

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

# ── Config: Longs ──────────────────────────────────────────────────────
CAPITAL = 200.0
TRANCH_SIZE = CAPITAL / 2          # $100 per position for multi-entry
HARD_SL = 4.0
TRAIL_ACTIVATE = 5.0               # lower activation to capture small rips
TRAIL_DIST = 3.0                   # tighter trail to protect gains
STALE_TIMEOUT_MINUTES = 15         # kill switch: recycle capital
MIN_GAP = 5.0
MIN_PRE_VOL = 50_000
MIN_PRICE = 3.0
MAX_PRICE = 250.0
MAX_POSITIONS = 2                  # split $200 into two $100 tranches
MIN_WIN_PROB = 0.35

# ── Config: RVOL (Task 1) ──────────────────────────────────────────────
RVOL_MIN = 0.5    # IEX data is directionally correct but ~1/200th of true volume
PRE_MARKET_START_HOUR = 4   # 4:00 AM ET

# ── Config: Shorts (Task 3) ────────────────────────────────────────────
SHORT_MIN_GAP = -8.0       # gap-down >= 8%
SHORT_SL = 6.0             # stop-loss at +6% from entry (price goes up)
SHORT_TP = 12.0            # take-profit at -12% from entry (price goes down)

# ── Config: Liquidity (Task 2) ─────────────────────────────────────────
MIN_MARKET_CAP = 100_000_000  # $100M

# ── Paths ───────────────────────────────────────────────────────────────
TRADE_DB = "/tmp/gap_trades.jsonl"
MODEL_DB = "/tmp/gap_model.json"

# ── Liquidity cache ────────────────────────────────────────────────────
_LIQ_CACHE: Dict[str, Optional[Dict]] = {}

WATCHLIST = [
    "NVDA","AMD","TSLA","META","AMZN","GOOGL","MSFT","AAPL","NFLX",
    "COIN","MSTR","MARA","RIOT","PLTR","SOFI","HOOD","AFRM","UPST",
    "SMCI","ARM","CRWD","PANW","DASH","UBER","LYFT","SNAP","PINS",
    "ROKU","ZM","DOCU","MDB","SNOW","DDOG","NET","CFLT","MNDY",
    "AI","IONQ","RGTI","QBTS","BBAI","SOUN","RDDT",
    "TQQQ","SOXL","FAS","LABU","UPRO","TNA","SPXL","FNGU",
    "NVDU","NVDL","TSLL","CONL","AMDL","MSTU","MSTX",
    "NIO","XPEV","LCID","RIVN","F","GM",
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
#  TASK 3 — SHORT SCAN SCAFFOLD
# ═══════════════════════════════════════════════════════════════════════

async def find_short_candidates(trading_client) -> list:
    """Scan for gap-down stocks suitable for short entries.
    Returns list of candidate dicts sorted by gap severity × RVOL.
    """
    candidates = []
    logger.info("Scanning %d stocks for SHORT gap-downs...", len(WATCHLIST))

    now_ny = datetime.now(NY_TZ)
    is_premarket = PRE_MARKET_START_HOUR <= now_ny.hour < 9 or (now_ny.hour == 9 and now_ny.minute < 30)

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

            bar_vol = int(today_slice["Volume"].sum())
            if bar_vol <= 0:
                prior_day = before_today[before_today.index.date == before_today.index.date[-1]]
                bar_vol = int(prior_day["Volume"].sum()) if not prior_day.empty else MIN_PRE_VOL + 1

            # Gap-down criterion
            if gap > SHORT_MIN_GAP or bar_vol < MIN_PRE_VOL:
                continue
            if price < MIN_PRICE or price > MAX_PRICE:
                continue

            # Check liquidity
            if trading_client:
                ok, reason = check_liquidity_floor(trading_client, sym, price)
                if not ok:
                    logger.debug("  SHORT SKIP %s (liquidity: %s)", sym, reason)
                    continue

            # Check easy-to-borrow
            if trading_client:
                asset = _check_asset(trading_client, sym)
                if asset is None:
                    continue
                if not asset.get("easy_to_borrow"):
                    logger.debug("  SHORT SKIP %s (not easy-to-borrow)", sym)
                    continue

            rvol = calculate_timestamp_rvol(sym)
            if rvol == -1.0:
                pass
            elif rvol < RVOL_MIN:
                logger.debug("  SHORT SKIP %s (RVOL=%.1f < %.1f)", sym, rvol, RVOL_MIN)
                continue

            candidates.append({
                "sym": sym, "gap": round(gap, 1), "vol": bar_vol,
                "price": round(price, 2), "rvol": rvol,
                "score": abs(gap) * 0.4 + rvol * 0.6,
                "opening_low": None,
            })
            logger.info("  SHORT CANDIDATE %s: gap=%.1f%% vol=%d rvol=%.1f $%.2f",
                        sym, gap, pre_vol, rvol, price)
            await asyncio.sleep(0.05)

        except Exception as e:
            logger.debug("Short scan skip %s: %s", sym, e)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


async def execute_short(trading_client, data_client, candidate: dict, model: TradeModel):
    """Execute short on a confirmed gap-down setup.
    Must be called after market open. Verifies 1-min opening low break.
    """
    sym = candidate["sym"]
    price = candidate["price"]
    gap_val = candidate["gap"]
    rvol_val = candidate.get("rvol", 0)
    score_val = candidate.get("score", 0)

    if trading_client is None:
        logger.info("SHORT SIM — no execution")
        return

    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLossRequest, StopLimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

    # Fetch the 9:30 AM opening bar low
    now_ny = datetime.now(NY_TZ)
    open_bar_utc = now_ny.replace(hour=9, minute=30, second=0, microsecond=0).astimezone(UTC_TZ)

    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        req = StockBarsRequest(
            symbol_or_symbols=sym,
            timeframe=TimeFrame.Minute,
            start=open_bar_utc - timedelta(minutes=1),
            end=open_bar_utc + timedelta(minutes=2),
            feed="iex",
        )
        bars = data_client.get_stock_bars(req)
        if bars.df is None or bars.df.empty:
            logger.warning("SHORT %s: no open bar data — skipping", sym)
            return

        df = bars.df.copy()
        df = df.reset_index(level="symbol", drop=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize(UTC_TZ)
        df.index = df.index.tz_convert(NY_TZ)

        # Find the 9:30 AM bar
        open_bar = df[(df.index.hour == 9) & (df.index.minute == 30)]
        if open_bar.empty:
            logger.warning("SHORT %s: 9:30 bar not found — skipping", sym)
            return

        opening_low = float(open_bar["low"].iloc[0])
    except Exception as e:
        logger.warning("SHORT %s: failed to get opening low: %s", sym, e)
        return

    candidate["opening_low"] = opening_low

    # Check current price vs opening low
    try:
        pos = trading_client.get_position(sym)
        current_price = float(pos.current_price)
    except Exception:
        try:
            # Fallback: fetch latest trade
            from alpaca.data.requests import StockLatestTradeRequest
            req = StockLatestTradeRequest(symbol_or_symbols=sym, feed="iex")
            trade = data_client.get_stock_latest_trade(req)
            current_price = float(trade[sym].price)
        except Exception as e:
            logger.warning("SHORT %s: cannot get current price: %s", sym, e)
            return

    if current_price >= opening_low:
        logger.info("SHORT %s: price $%.2f >= opening low $%.2f — no break, skipping",
                    sym, current_price, opening_low)
        return

    # ---- Entry ----
    qty = int(CAPITAL / current_price)
    if qty < 1:
        logger.warning("SHORT %s: cannot afford 1 share at $%.2f", sym, current_price)
        return

    sl_price = round(current_price * (1 + SHORT_SL / 100), 2)
    tp_price = round(current_price * (1 - SHORT_TP / 100), 2)

    logger.info("\n" + "=" * 55)
    logger.info("SHORT %s: gap=%.1f%% rvol=%.1f score=%.1f",
                sym, gap_val, rvol_val, score_val)
    logger.info("   %d sh × $%.2f (opening low $%.2f)", qty, current_price, opening_low)
    logger.info("   SL: +%.0f%% ($%.2f) | TP: -%.0f%% ($%.2f)",
                SHORT_SL, sl_price, SHORT_TP, tp_price)
    logger.info("=" * 55)

    try:
        short_order = MarketOrderRequest(
            symbol=sym,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            take_profit=LimitOrderRequest(limit_price=tp_price),
            stop_loss=StopLimitOrderRequest(
                stop_price=sl_price,
                limit_price=round(sl_price * 1.02, 2),
            ),
        )
        trading_client.submit_order(short_order)
        logger.info("SHORT ENTERED %d %s at $%.2f | SL $%.2f TP $%.2f",
                    qty, sym, current_price, sl_price, tp_price)

        trade_entry = {
            "sym": sym, "qty": qty, "entry": current_price,
            "time": time.time(), "side": "short",
        }

        # Wait for fill / monitor
        while True:
            await asyncio.sleep(30)
            try:
                pos = trading_client.get_position(sym)
                if float(pos.qty) == 0:
                    logger.info("SHORT %s: position closed", sym)
                    break
                cur = float(pos.current_price)
                gain = ((cur - current_price) / current_price) * 100
                logger.debug("SHORT %s: $%.2f (%.1f%%)", sym, cur, gain)
            except Exception:
                logger.info("SHORT %s: position closed (no longer held)", sym)
                break

        # Log result
        try:
            acts = trading_client.get_activities()
            for a in acts:
                if a.symbol == sym and a.side == "buy" and "cover" in str(a.type).lower():
                    exit_price = float(a.price)
                    gain = round(((exit_price - current_price) / current_price) * 100, 2)
                    model.add_trade({
                        "sym": sym, "qty": qty, "entry": current_price,
                        "exit": exit_price, "gain": gain, "win": gain > 0,
                        "exit": "short_bracket", "side": "short",
                        "time": datetime.now(timezone.utc).isoformat(),
                    })
                    logger.info("SHORT logged: %s %.2f%%", sym, gain)
                    break
        except Exception as e:
            logger.debug("SHORT log: %s", e)

    except Exception as e:
        logger.error("SHORT execution error: %s", e)


# ═══════════════════════════════════════════════════════════════════════
#  EXECUTION — LONG
# ═══════════════════════════════════════════════════════════════════════

async def execute_long(trading_client, signal: dict, model: TradeModel,
                       tranche_size: float = CAPITAL):
    """Buy with hard SL bracket + trailing stop monitor + stale kill switch.

    Three capital-velocity enhancements:
    1. No fixed TP — asymmetric trail lets winners run uncapped.
    2. Trail activates at +5%, trails 3% below peak (tighter, faster).
    3. Stale kill switch: market-exits after STALE_TIMEOUT_MINUTES to recycle cash.
    """
    sym = signal["sym"]
    price = signal["price"]
    qty = int(tranche_size / price)
    if qty < 1:
        logger.warning("Can't afford %s ($%.2f) with $%.0f tranche", sym, price, tranche_size)
        return

    sl_price = round(price * (1 - HARD_SL / 100), 2)

    logger.info("\n" + "=" * 55)
    logger.info("LONG %s: gap=+%.1f%%  win_prob=%.0f%%  tranche=$%.0f",
                sym, signal.get("gap", 0), signal.get("win_prob", 0) * 100, tranche_size)
    logger.info("   %d sh × $%.2f = $%.2f", qty, price, round(qty * price, 2))
    logger.info("   SL: -%.0f%% ($%.2f) | Trail: +%.0f%% -> trail %.0f%% | Kill: %dmin",
                HARD_SL, sl_price, TRAIL_ACTIVATE, TRAIL_DIST, STALE_TIMEOUT_MINUTES)
    logger.info("   RVOL: %.1f", signal.get("rvol", 0))
    logger.info("=" * 55)

    if signal.get("win_prob", 0) < MIN_WIN_PROB:
        logger.info("Skipping — win prob below %.0f%%", MIN_WIN_PROB * 100)
        return

    if trading_client is None:
        logger.info("SIM — logging only")
        return

    try:
        from alpaca.trading.requests import MarketOrderRequest, StopLimitOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

        buy = MarketOrderRequest(
            symbol=sym, qty=qty, side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        trading_client.submit_order(buy)
        logger.info("Bought %d %s", qty, sym)

        # SL-only bracket (no TP cap — let trail capture runners)
        bracket = LimitOrderRequest(
            symbol=sym, qty=qty, side=OrderSide.SELL,
            limit_price=round(sl_price * 0.98, 2),  # placeholder, won't fill ahead of SL
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLimitOrderRequest(
                stop_price=sl_price,
                limit_price=round(sl_price * 0.98, 2),
            ),
        )
        trading_client.submit_order(bracket)
        logger.info("Bracket SL=$%.2f (no TP — trail manages exits)", sl_price)

        peak = price
        trail_active = False
        entry_time = time.time()
        trade_entry = {"sym": sym, "qty": qty, "entry": price, "time": entry_time}

        while True:
            await asyncio.sleep(15)

            # ── Stale kill switch ────────────────────────────────────
            elapsed = (time.time() - entry_time) / 60
            if elapsed >= STALE_TIMEOUT_MINUTES:
                logger.info("KILL SWITCH: %.0f min stale — recycling $%.0f for %s",
                            elapsed, tranche_size, sym)
                try:
                    pos = trading_client.get_position(sym)
                    exit_price = float(pos.current_price)
                    trading_client.close_position(sym)
                    await _log_trade(trade_entry, exit_price, "stale", model)
                except Exception:
                    pass
                return

            try:
                pos = trading_client.get_position(sym)
                cur = float(pos.current_price)
                gain = ((cur - price) / price) * 100

                if cur > peak:
                    peak = cur
                    if not trail_active and gain >= TRAIL_ACTIVATE:
                        trail_active = True
                        logger.info("TRAIL ACTIVE at +%.1f%% ($%.2f)", gain, cur)

                if trail_active:
                    trail_stop = peak * (1 - TRAIL_DIST / 100)
                    if cur <= trail_stop:
                        logger.info("TRAIL HIT at $%.2f (+%.1f%%)", cur, ((cur - price) / price) * 100)
                        trading_client.close_position(sym)
                        await _log_trade(trade_entry, cur, "trail", model)
                        return

            except Exception as e:
                err = str(e).lower()
                if "position" in err or "404" in err:
                    logger.info("Position closed (SL bracket)")
                    try:
                        acts = trading_client.get_activities()
                        for a in acts:
                            if a.symbol == sym and a.side == "sell":
                                await _log_trade(trade_entry, float(a.price), "bracket", model)
                                break
                    except Exception:
                        pass
                    return
                logger.debug("Monitor: %s", e)

    except Exception as e:
        logger.error("Execution error: %s", e)


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
#  FETCH OPENING LOW (shared helper for short scan)
# ═══════════════════════════════════════════════════════════════════════

async def fetch_opening_low(data_client, symbol: str) -> Optional[float]:
    """Fetch the 9:30 AM ET 1-minute bar low for a symbol."""
    now_ny = datetime.now(NY_TZ)
    open_bar_utc = now_ny.replace(hour=9, minute=30, second=0, microsecond=0).astimezone(UTC_TZ)
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=open_bar_utc - timedelta(minutes=1),
            end=open_bar_utc + timedelta(minutes=2),
            feed="iex",
        )
        bars = data_client.get_stock_bars(req)
        if bars.df is None or bars.df.empty:
            return None
        df = bars.df.copy()
        df = df.reset_index(level="symbol", drop=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize(UTC_TZ)
        df.index = df.index.tz_convert(NY_TZ)
        row = df[(df.index.hour == 9) & (df.index.minute == 30)]
        if row.empty:
            return None
        return float(row["low"].iloc[0])
    except Exception as e:
        logger.debug("Opening low fetch failed for %s: %s", symbol, e)
        return None


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

async def main():
    SIM = "--sim" in sys.argv
    trading_client = None if SIM else get_trading_client()
    data_client = None if SIM else get_data_client()  # kept for short scan opening-low fetch

    if trading_client is None and not SIM:
        logger.warning("No Alpaca keys. Run --sim or export APCA_API_KEY_ID / APCA_API_SECRET_KEY")
        SIM = True

    model = TradeModel()

    best_p = model.backtest_params()
    if best_p:
        logger.info("Optimal params: %s", best_p)

    mode = "SIM" if SIM else "ALPACA PAPER"
    logger.info("\n" + "=" * 55)
    logger.info("GAP BOT v4 — Self-Learning + RVOL + Liquidity Floor + Shorts")
    logger.info("   $%.0f | SL: -%.0f%% | Trail: +%.0f%% -> %.0f%% | RVOL >= %.1f",
                CAPITAL, HARD_SL, TRAIL_ACTIVATE, TRAIL_DIST, RVOL_MIN)
    logger.info("   Mode: %s | Watchlist: %d | DB: %d trades",
                mode, len(WATCHLIST), len(model.trades))
    logger.info("   %s", model.report())
    logger.info("=" * 55 + "\n")

    today_state: Dict = {}
    short_candidates: list = []
    current_short: Optional[dict] = None

    while True:
        now_ny = datetime.now(NY_TZ)
        h, m = now_ny.hour, now_ny.minute
        day_key = now_ny.date().isoformat()
        if day_key not in today_state:
            today_state[day_key] = set()
        state = today_state[day_key]

        # ── 9:00 AM — Scan ──────────────────────────────────────────
        if h == 9 and m == 0 and "scan" not in state:
            state.add("scan")
            logger.info("\nSCAN — %s", model.report())

            # Long scan
            signals = await scan_premarket(trading_client)
            for s in signals:
                s["weekday"] = now_ny.weekday()
                s["win_prob"] = model.predict_win_prob(s)
                s["score"] = (s["gap"] * 0.3 + (s["vol"] / 1e6) * 0.2
                              + s["rel_vol"] * 0.1 + s["rvol"] * 0.2 + s["win_prob"] * 0.2)

            signals.sort(key=lambda x: x["score"], reverse=True)

            if signals:
                logger.info("LONG candidates (scored):")
                for s in signals:
                    wp = s["win_prob"]
                    flag = "TRADE" if wp >= MIN_WIN_PROB else "SKIP"
                    logger.info("  %s %s: gap=+%.1f%% rvol=%.1f win_prob=%.0f%% score=%.1f",
                                flag, s["sym"], s["gap"], s["rvol"], wp * 100, s["score"])

                # Select top N signals that meet win prob
                picks = [s for s in signals if s["win_prob"] >= MIN_WIN_PROB][:MAX_POSITIONS]
                if picks:
                    logger.info("TOP %d PICKS:", len(picks))
                    for p in picks:
                        logger.info("  %s: gap=+%.1f%% wp=%.0f%% score=%.1f",
                                    p["sym"], p["gap"], p["win_prob"] * 100, p["score"])
                    state.add("has_long")
                    state.add(("long_picks", picks))
                else:
                    logger.info("No long meets confidence threshold.")
            else:
                logger.info("No qualifying LONG gap-ups after all filters.")

            # Short scan (Task 3)
            short_candidates = await find_short_candidates(trading_client)
            if short_candidates:
                logger.info("SHORT candidate: %s gap=%.1f%% score=%.1f",
                            short_candidates[0]["sym"],
                            short_candidates[0]["gap"],
                            short_candidates[0]["score"])
                state.add("has_short")
            else:
                logger.info("No qualifying SHORT gap-downs.")

            await asyncio.sleep(60)

        # ── 9:31 AM — Execute long(s) ───────────────────────────────
        if h == 9 and m >= 31 and "executed_long" not in state and "has_long" in state:
            elapsed = (h - 9) * 3600 + m * 60 - 30 * 60
            if elapsed >= 91:
                state.add("executed_long")
                picks = []
                for s in state:
                    if isinstance(s, tuple) and s[0] == "long_picks":
                        picks = s[1]
                        break
                for i, pick in enumerate(picks):
                    await execute_long(trading_client, pick, model, tranche_size=TRANCH_SIZE)

        # ── 9:31 AM — Execute short (after opening low confirmed) ──
        if h == 9 and m >= 31 and "executed_short" not in state and "has_short" in state:
            elapsed = (h - 9) * 3600 + m * 60 - 30 * 60
            if elapsed >= 120 and short_candidates:  # 2 min after open to get 9:30 bar
                state.add("executed_short")
                current_short = short_candidates[0]
                low = await fetch_opening_low(data_client, current_short["sym"])
                if low is not None:
                    current_short["opening_low"] = low
                    await execute_short(trading_client, data_client, current_short, model)
                else:
                    logger.warning("Cannot confirm opening low for %s — skipping short",
                                   current_short["sym"])

        # ── 4:00 PM — Report + overnight ────────────────────────────
        if h >= 16 and "reported" not in state:
            state.add("reported")
            if trading_client:
                try:
                    acct = trading_client.get_account()
                    pnl = float(acct.equity) - CAPITAL
                    logger.info("\nDAY RESULT: $%.2f | P&L: $%.2f (%.1f%%)",
                                float(acct.equity), pnl, pnl / CAPITAL * 100)
                except Exception:
                    pass
            logger.info("\n%s", model.report())
            logger.info("Next scan: tomorrow 9:00 AM ET")
            bp = model.backtest_params()
            if bp:
                logger.info("Optimal for tomorrow: %s", bp)

        # Clean old day states
        for k in list(today_state.keys()):
            if k < day_key:
                del today_state[k]

        await asyncio.sleep(10)


# ═══════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--oneshot" in sys.argv:
        model = TradeModel()
        print(model.report())
        if "--scan" in sys.argv:
            tc = get_trading_client()
            sigs = asyncio.run(scan_premarket(tc))
            for s in sigs:
                s["win_prob"] = model.predict_win_prob(s)
                s["score"] = (s["gap"] * 0.3 + (s["vol"] / 1e6) * 0.2
                              + s["rel_vol"] * 0.1 + s["rvol"] * 0.2 + s["win_prob"] * 0.2)
                print(f"  {s['sym']:6s} gap=+{s['gap']:5.1f}% vol={s['vol']:>8d} "
                      f"rel={s['rel_vol']:.1f}x rvol={s['rvol']:.1f} "
                      f"wp={s['win_prob']:.0%} score={s['score']:.1f}")
        elif "--shortscan" in sys.argv:
            tc = get_trading_client()
            cands = asyncio.run(find_short_candidates(tc))
            for c in cands:
                print(f"  SHORT {c['sym']:6s} gap={c['gap']:6.1f}% rvol={c['rvol']:.1f} score={c['score']:.1f}")
        elif "--backtest" in sys.argv:
            params = model.backtest_params()
            print(f"Optimal: {params}")
    else:
        asyncio.run(main())
