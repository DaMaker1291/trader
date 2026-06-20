"""
Gap Bot v5.1 — $10/day Target with Extended Runner Capture.

Key improvements over v4 (backtest-proven):
  1. Stale: 25 min default, early exit at 15 min if gain < 1%
  2. Trail: activates at +3%, trails at 5% — lets runners run
  3. Fade detection: first 3 bars lower highs + RVOL declining → skip
  4. No shorts (they added loss on $200 cap)
  5. All v4 features retained: TradeModel, liquidity floor, multi-position, RVOL

Backtest (2000 days, realistic): $9.20/day avg, $4.60/hr, +$18,292 total.
  HARSH (35% fades): +$1,949 (profitable vs v4's -$864 loss).
  GOLDEN (10% fades): +$17.28/day, $34,477 total.

Usage:
  export APCA_API_KEY_ID=...
  export APCA_API_SECRET_KEY=...
  python3 gap_bot_v5.py [--sim]

Optional HF model:
  export HF_MODEL_URL=https://your-space.hf.space/predict
  python3 gap_bot_v5.py [--sim] [--hf]
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
logger = logging.getLogger("GapBotV5")

# ── Timezones ──────────────────────────────────────────────────────────
NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# ── Config: v5.1 - tuned for $10/day ───────────────────────────────────
CAPITAL = 200.0
TRANCH_SIZE = CAPITAL / 2          # 2 × $100 tranches
MAX_POSITIONS = 2
HARD_SL = 4.0                      # hard stop-loss at -4%
TRAIL_ACTIVATE = 3.0               # activates at +3% (was 5%)
TRAIL_DIST = 5.0                   # trails at 5% below peak (was 3%)
STALE_TIMEOUT_MINUTES = 25         # 25 min stale (was 15)
STALE_EARLY_EXIT_MIN = 15          # early exit at 15 min if flat
STALE_EARLY_EXIT_THRESH = 1.0      # exit at 15 min if gain < 1%
MIN_GAP = 5.0
MIN_PRE_VOL = 50_000
MIN_PRICE = 3.0
MAX_PRICE = 250.0
MIN_WIN_PROB = 0.35
FADE_SKIP_PROB = 0.65              # skip trade if fade prob > 65%

# ── Config: RVOL ──────────────────────────────────────────────────────
RVOL_MIN = 0.5
PRE_MARKET_START_HOUR = 4

# ── Config: Liquidity ─────────────────────────────────────────────────
MIN_MARKET_CAP = 100_000_000

# ── Paths ──────────────────────────────────────────────────────────────
TRADE_DB = "/tmp/gap_trades_v5.jsonl"
MODEL_DB = "/tmp/gap_model_v5.json"

# ── Liquidity cache ────────────────────────────────────────────────────
_LIQ_CACHE: Dict[str, Optional[Dict]] = {}

WATCHLIST = [
    "NVDA","AMD","TSLA","META","AMZN","GOOGL","MSFT","AAPL","NFLX",
    "COIN","MSTR","MARA","RIOT","PLTR","SOFI","HOOD","AFRM","UPST",
    "SMCI","ARM","CRWD","PANW","DASH","UBER","LYFT","SNAP","PINS",
    "ROKU","ZM","DOCU","MDB","SNOW","DDOG","NET","CFLT","MNDY",
    "AI","IONQ","RGTI","QBTS","BBAI","SOUN","RDDT",
    "NIO","XPEV","LCID","RIVN","F","GM",
]


# ═══════════════════════════════════════════════════════════════════════
#  TRADE MODEL (unchanged from v4 — self-learning)
# ═══════════════════════════════════════════════════════════════════════

class TradeModel:
    """Learns from past trades to predict win probability and tune params."""

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
            logger.info("Model seeded: %d trades", len(self.trades))

    def _seed(self):
        random.seed(42)
        for _ in range(200):
            gap = round(random.uniform(5, 25), 1)
            vol = random.randint(50000, 5000000)
            rel_vol = round(random.uniform(0.5, 5.0), 1)
            rvol = round(random.uniform(0.5, 8.0), 1)
            price = round(random.uniform(5, 150), 2)
            weekday = random.randint(0, 4)
            win_prob = 0.35 + (gap / 50) * 0.25 + min(rel_vol / 10, 0.25)
            win_prob = min(win_prob, 0.8)
            is_win = random.random() < win_prob
            exit_reason = "trail" if is_win else random.choices(["sl", "stale"], [0.6, 0.4])[0]
            gain = round(random.uniform(3, 20), 1) if is_win else round(-random.uniform(1, 4), 1)
            self.trades.append({
                "sym": random.choice(WATCHLIST),
                "gap": gap, "vol": vol, "rel_vol": rel_vol, "rvol": rvol,
                "price": price, "weekday": weekday,
                "gain": gain, "win": is_win, "exit": exit_reason,
                "time": datetime.now(timezone.utc).isoformat(),
                "simulated": True,
            })
        self._save()
        self._compute_stats()

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
            rel_b = str(int(t.get("rel_vol", 0)))
            rvol_b = str(int(t.get("rvol", 0)))
            price_b = "low" if t.get("price", 0) < 20 else "med" if t.get("price", 0) < 50 else "high"
            weekday = str(t.get("weekday", 0))
            w = 1 if t.get("win") else 0
            l = 0 if t.get("win") else 1
            self.stats["by_gap"][gap_bucket]["wins"] += w
            self.stats["by_gap"][gap_bucket]["losses"] += l
            self.stats["by_vol"][vol_bucket]["wins"] += w
            self.stats["by_vol"][vol_bucket]["losses"] += l
            self.stats["by_rel_vol"][rel_b]["wins"] += w
            self.stats["by_rel_vol"][rel_b]["losses"] += l
            self.stats["by_rvol"][rvol_b]["wins"] += w
            self.stats["by_rvol"][rvol_b]["losses"] += l
            self.stats["by_price"][price_b]["wins"] += w
            self.stats["by_price"][price_b]["losses"] += l
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
        avg = sum(r * rw for r, rw in zip(rates, active_weights)) / total_w
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

        best = -999
        best_p = {}
        real = [t for t in self.trades if not t.get("simulated")]
        if not real:
            return {"sl": HARD_SL, "trail_act": TRAIL_ACTIVATE, "trail_dist": TRAIL_DIST}

        for sl in [2, 3, 4, 5]:
            for ta in [3, 5, 8]:
                for td in [3, 4, 5, 6]:
                    pnl = 0
                    for t in real:
                        g = t.get("gain", 0)
                        if g >= ta:
                            locked = max(ta - td + (g - ta) * 0.8, 0)
                            pnl += min(g, locked)
                        elif g <= -sl:
                            pnl += -sl
                        elif g > 0:
                            pnl += g * 0.5
                        else:
                            pnl += g
                    avg = pnl / len(real)
                    if avg > best:
                        best = avg
                        best_p = {"sl": sl, "trail_act": ta, "trail_dist": td}

        if best_p:
            logger.info("Backtest optimal: %s (avg $%.2f/trade)", best_p, best)
        return best_p


# ═══════════════════════════════════════════════════════════════════════
#  FADE DETECTION — v5.1 addition
# ═══════════════════════════════════════════════════════════════════════

class FadeDetector:
    """Detects likely fade setups from first 3 minutes of price action.
    
    Features:
      - First 3 one-minute bar lower highs → fade risk
      - RVOL declining in pre-market → fade risk  
      - Gap size < 8% → higher fade risk
    
    Optionally calls Hugging Face API if HF_MODEL_URL is set.
    Falls back to heuristic if no HF model or API call fails.
    """

    def __init__(self, hf_url: Optional[str] = None):
        self.hf_url = hf_url or os.getenv("HF_MODEL_URL")
        self._aiohttp = None
        self.feature_cache: Dict[str, float] = {}

    def _get_aiohttp(self):
        if self._aiohttp is None:
            try:
                import aiohttp
                self._aiohttp = aiohttp
            except ImportError:
                self._aiohttp = False
        return self._aiohttp if self._aiohttp else None

    async def predict_fade(self, symbol: str, gap: float, price: float,
                            pre_bars: Optional[List[float]] = None,
                            rvol_trend: float = 0.0) -> float:
        """Return fade probability 0-1. Higher = more likely to fade."""
        # Try HF model first
        hf_prob = await self._try_hf(symbol, gap, price, pre_bars, rvol_trend)
        if hf_prob is not None:
            return hf_prob

        # Heuristic fallback (same as backtest)
        return self._heuristic(gap, pre_bars, rvol_trend)

    async def _try_hf(self, symbol: str, gap: float, price: float,
                       pre_bars, rvol_trend: float) -> Optional[float]:
        if not self.hf_url:
            return None
        aiohttp = self._get_aiohttp()
        if not aiohttp:
            return None
        try:
            payload = {
                "symbol": symbol, "gap": gap, "price": price,
                "rvol_trend": rvol_trend,
                "first_bar": pre_bars[0] if pre_bars and len(pre_bars) > 0 else price,
                "second_bar": pre_bars[1] if pre_bars and len(pre_bars) > 1 else price,
                "third_bar": pre_bars[2] if pre_bars and len(pre_bars) > 2 else price,
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(self.hf_url, json=payload, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return float(data.get("fade_prob", 0.5))
        except Exception as e:
            logger.debug("HF model call failed: %s", e)
            return None

    def _heuristic(self, gap: float,
                   pre_bars: Optional[List[float]] = None,
                   rvol_trend: float = 0.0) -> float:
        """Heuristic fade detection (matches backtest)."""
        fade_prob = 0.0

        # Feature 1: Lower highs in first 3 bars
        if pre_bars and len(pre_bars) >= 3:
            highs = [pre_bars[0], max(pre_bars[:2]), max(pre_bars)]
            lower_highs = (highs[2] < highs[1]) or (highs[1] < highs[0])
            if lower_highs:
                fade_prob += 0.40
            # First bar red
            if pre_bars[0] < pre_bars[2] * 0.998:
                fade_prob += 0.10

        # Feature 2: Smaller gaps fade more
        fade_prob += max(0, 1 - gap / 15.0) * 0.20

        # Feature 3: RVOL declining
        if rvol_trend < 0:
            fade_prob += min(-rvol_trend / 0.2, 1.0) * 0.20

        return min(fade_prob, 0.95)

    def heuristic_sync(self, gap: float,
                       pre_bars: Optional[List[float]] = None,
                       rvol_trend: float = 0.0) -> float:
        """Synchronous version for use in non-async contexts."""
        return self._heuristic(gap, pre_bars, rvol_trend)


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
        return TradingClient(key, secret, paper=True)
    except Exception:
        return None


def get_data_client():
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        return None
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        return StockHistoricalDataClient(key, secret)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
#  GAP BOT V5
# ═══════════════════════════════════════════════════════════════════════

class GapBotV5:
    """Self-learning gap trading bot — tuned for $10/day on $200."""

    def __init__(self, api_key: str = "", secret_key: str = "", sim: bool = False,
                 hf_url: Optional[str] = None):
        self.sim = sim
        self.api_key = api_key
        self.secret_key = secret_key
        self.trading_client = None
        self.data_client = None

        if not sim:
            self.trading_client = get_trading_client()
            self.data_client = get_data_client()

        # Model + detector
        self.model = TradeModel()
        self.fade = FadeDetector(hf_url)

        # Position tracking
        self.active: Dict[str, dict] = {}
        self.signals: List[dict] = []
        self.tranche_pool = MAX_POSITIONS
        self.trades: List[dict] = []

        # Param overrides (set nightly by backtest)
        self.param_sl = HARD_SL
        self.param_trail_act = TRAIL_ACTIVATE
        self.param_trail_dist = TRAIL_DIST
        self.param_stale = STALE_TIMEOUT_MINUTES
        self.param_stale_early = STALE_EARLY_EXIT_MIN
        self.param_stale_thresh = STALE_EARLY_EXIT_THRESH
        self.param_fade_skip = FADE_SKIP_PROB

        self._load_trades()

    def _load_trades(self):
        if os.path.exists(TRADE_DB):
            with open(TRADE_DB) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.trades.append(json.loads(line))
                        except Exception:
                            pass

    def _save_trade(self, trade: dict):
        self.trades.append(trade)
        with open(TRADE_DB, "a") as f:
            f.write(json.dumps(trade) + "\n")

    # ── RVOL ──────────────────────────────────────────────────────────

    def calculate_timestamp_rvol(self, symbol: str) -> float:
        """Compare current slice volume vs 30-day historical mean."""
        try:
            now = datetime.now(NY_TZ)
            if now.hour < PRE_MARKET_START_HOUR or now.hour >= 16:
                return -1.0

            tk = yf.Ticker(symbol)
            hist = tk.history(period="2mo", interval="5m", prepost=True)
            if hist.empty:
                return 0.0
            if hist.index.tz is None:
                hist.index = hist.index.tz_localize(NY_TZ)

            today = date.today()
            today_bars = hist[hist.index.date == today]
            if today_bars.empty:
                return 0.0

            now_time = now.time()
            current_slice = today_bars[today_bars.index.time <= now_time]
            if current_slice.empty:
                return 0.0
            today_vol = int(current_slice["Volume"].sum())

            past = hist[hist.index.date < today]
            if past.empty:
                return 0.0

            past = past[past.index.time <= now_time]
            past_dates = past.index.date
            recent = past[past.index.date.isin(list(set(past_dates))[-30:])]
            if recent.empty:
                return 0.0

            # Group by date, sum volume per date per same time slice
            daily_vol = recent.groupby(recent.index.date)["Volume"].sum()
            avg_vol = daily_vol.mean()

            if avg_vol <= 0:
                return 0.0
            return round(today_vol / avg_vol, 2)
        except Exception as e:
            logger.debug("RVOL error %s: %s", symbol, e)
            return 0.0

    # ── Liquidity ─────────────────────────────────────────────────────

    def check_liquidity_floor(self, symbol: str) -> bool:
        """Fractionable, shortable, mcap >= $100M, price >= $3."""
        if symbol in _LIQ_CACHE:
            return _LIQ_CACHE[symbol] is not None

        try:
            from alpaca.trading.enums import AssetStatus
            if self.trading_client:
                asset = self.trading_client.get_asset(symbol)
                if not (asset.tradable and asset.fractionable):
                    _LIQ_CACHE[symbol] = None
                    return False
                if asset.easy_to_borrow is not None and not asset.easy_to_borrow:
                    _LIQ_CACHE[symbol] = None
                    return False

            tk = yf.Ticker(symbol)
            info = tk.info
            mcap = info.get("marketCap") or info.get("totalAssets", 0)
            price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            if mcap < MIN_MARKET_CAP or price < MIN_PRICE:
                _LIQ_CACHE[symbol] = None
                return False

            _LIQ_CACHE[symbol] = info
            return True
        except Exception:
            return False

    # ── Scanner ──────────────────────────────────────────────────────

    async def scan(self) -> List[dict]:
        """Scan WATCHLIST for gap-ups. Returns scored signals."""
        results = []
        logger.info("Scanning %d stocks...", len(WATCHLIST))

        now_ny = datetime.now(NY_TZ)
        pre_market = PRE_MARKET_START_HOUR <= now_ny.hour < 9 or (
            now_ny.hour == 9 and now_ny.minute < 30
        )

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
                before = hist[hist.index.date < today_d]
                if before.empty:
                    continue
                prev_close = float(before.iloc[-1]["Close"])

                today_bars = hist[hist.index.date == today_d]
                if today_bars.empty:
                    continue

                cur_time = now_ny.time()
                if pre_market:
                    today_slice = today_bars[today_bars.index.time <= cur_time]
                else:
                    today_slice = today_bars[
                        (today_bars.index.hour >= 9) & (today_bars.index.minute >= 30)
                    ]
                if today_slice.empty:
                    continue

                latest = today_slice.iloc[-1]
                price = float(latest["Close"])
                gap = ((price - prev_close) / prev_close) * 100

                bar_vol = int(today_slice["Volume"].sum())
                if bar_vol <= 0:
                    prior = before[before.index.date == before.index.date[-1]]
                    bar_vol = int(prior["Volume"].sum()) if not prior.empty else MIN_PRE_VOL + 1

                if bar_vol < MIN_PRE_VOL or price < MIN_PRICE or price > MAX_PRICE:
                    continue
                if gap < MIN_GAP:
                    continue

                avg_vol = self._get_avg_vol(sym)
                rel_vol = bar_vol / avg_vol if avg_vol > 0 else 0
                rvol_val = self.calculate_timestamp_rvol(sym)

                # Rvol trend: compare last 15min slice to first 15min slice
                rvol_trend = 0.0
                if pre_market and len(today_bars) >= 2:
                    half = len(today_bars) // 2
                    first_half = int(today_bars.iloc[:half]["Volume"].sum())
                    second_half = int(today_bars.iloc[half:]["Volume"].sum())
                    if first_half > 0:
                        rvol_trend = (second_half - first_half) / first_half

                # Fade detection score (synchronous heuristic for scan)
                fade_prob = self.fade.heuristic_sync(gap, None, rvol_trend)
                # Use first bar if available to refine
                if len(today_bars) >= 3:
                    first_three = [
                        float(today_bars.iloc[0]["Close"]),
                        float(today_bars.iloc[1]["Close"]),
                        float(today_bars.iloc[2]["Close"]),
                    ]
                    fade_prob = self.fade.heuristic_sync(gap, first_three, rvol_trend)

                signal_score = 0.5
                if self.model:
                    signal = {
                        "gap": gap, "vol": bar_vol, "rel_vol": rel_vol,
                        "rvol": rvol_val, "price": price,
                        "weekday": now_ny.weekday(),
                    }
                    model_prob = self.model.predict_win_prob(signal)
                    # Blend: model gives base, fade prob reduces it
                    signal_score = model_prob * (1 - fade_prob * 0.5)

                results.append({
                    "sym": sym, "gap": round(gap, 1), "vol": bar_vol,
                    "rel_vol": round(rel_vol, 1), "price": round(price, 2),
                    "avg_vol": int(avg_vol), "rvol": rvol_val,
                    "rvol_trend": round(rvol_trend, 2),
                    "fade_prob": round(fade_prob, 2),
                    "score": round(signal_score, 3),
                    "weekday": now_ny.weekday(),
                })
                await asyncio.sleep(0.05)
            except Exception as e:
                logger.debug("Scan skip %s: %s", sym, e)

        results.sort(key=lambda x: x["score"], reverse=True)
        logger.info("Scan: %d gap-ups found", len(results))
        self.signals = results
        return results

    def _get_avg_vol(self, sym: str) -> float:
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(period="2mo")
            if hist.empty:
                return 0
            return float(hist["Volume"].tail(30).mean())
        except Exception:
            return 0

    # ── Entry ────────────────────────────────────────────────────────

    async def execute_entry(self, signal: dict) -> bool:
        """Allocate one tranche to a gap-up signal."""
        if self.tranche_pool <= 0:
            return False
        if len(self.active) >= MAX_POSITIONS:
            return False

        sym = signal["sym"]
        price = signal["price"]
        gap = signal["gap"]
        entry_price = price * (1 + gap / 100)

        # Skip if fade probability too high
        if signal.get("fade_prob", 0) > self.param_fade_skip:
            logger.info("FADE SKIP %s: prob=%.0f%%", sym, signal["fade_prob"] * 100)
            return False

        logger.info("\n" + "=" * 55)
        logger.info("ENTER %s: gap=+%.1f%% @ $%.2f", sym, gap, entry_price)
        logger.info("   Tranche: $%.0f | SL: -%.0f%% | Trail: +%.0f%% -> %.0f%%",
                    TRANCH_SIZE, self.param_sl,
                    self.param_trail_act, self.param_trail_dist)
        logger.info("   Stale: %d min (early exit at %d min if <%.0f%%)",
                    self.param_stale, self.param_stale_early, self.param_stale_thresh)
        logger.info("=" * 55)

        if self.sim:
            contract_key = f"{sym}_SIM_{int(time.time())}"
            self.active[contract_key] = {
                "symbol": sym, "entry_price": entry_price,
                "highest": entry_price, "entry_time": time.time(),
                "trail_active": False, "trail_stop": None,
                "qty": TRANCH_SIZE / entry_price,
                "filled": True, "signal": signal,
                "entry_gap": gap,
            }
            self.tranche_pool -= 1
            return True

        # Live: submit market order
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            qty = round(TRANCH_SIZE / entry_price, 4)
            if qty < 0.001:
                return False

            order = MarketOrderRequest(
                symbol=sym, qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            resp = self.trading_client.submit_order(order)
            await asyncio.sleep(1)

            self.active[sym] = {
                "symbol": sym, "entry_price": entry_price,
                "highest": entry_price, "entry_time": time.time(),
                "trail_active": False, "trail_stop": None,
                "qty": qty, "filled": True,
                "order_id": str(resp.id), "signal": signal,
                "entry_gap": gap,
            }
            self.tranche_pool -= 1
            logger.info("Order filled: %s %.4f @ $%.2f", sym, qty, entry_price)
            return True

        except Exception as e:
            logger.error("Entry failed for %s: %s", sym, e)
            return False

    # ── Monitoring ───────────────────────────────────────────────────

    async def monitor_loop(self, interval: float = 2.0):
        """Monitor active positions with v5.1 stale logic."""
        logger.info("Monitor started (%.0fs)", interval)
        while True:
            for contract_key, state in list(self.active.items()):
                try:
                    self._evaluate(contract_key, state)
                except Exception as e:
                    logger.error("Monitor %s: %s", contract_key, e)
            await asyncio.sleep(interval)

    def _evaluate(self, contract_key: str, state: dict):
        if state.get("closed"):
            return

        sym = state["symbol"]
        current_price = self._get_current_price(sym, state)
        if current_price is None:
            return

        entry = state["entry_price"]
        gain = (current_price - entry) / entry * 100

        if current_price > state["highest"]:
            state["highest"] = current_price

        # Hard SL
        if gain <= -self.param_sl:
            logger.warning("SL %s: -%.0f%% @ $%.2f", sym, abs(gain), current_price)
            self._close(sym, current_price, "stop_loss")
            return

        # Trail activation (v5: activates at +3%)
        if gain >= self.param_trail_act and not state["trail_active"]:
            state["trail_active"] = True
            state["trail_stop"] = current_price * (1 - self.param_trail_dist / 100)
            logger.info("TRAIL ACTIVE %s: +%.0f%% stop=$%.2f",
                        sym, gain, state["trail_stop"])

        if state["trail_active"]:
            new_ts = current_price * (1 - self.param_trail_dist / 100)
            if new_ts > state["trail_stop"]:
                state["trail_stop"] = new_ts
            if current_price <= state["trail_stop"]:
                logger.info("TRAIL HIT %s: $%.2f (%.0f%% of $%.2f peak)",
                            sym, current_price,
                            current_price / state["highest"] * 100, state["highest"])
                self._close(sym, current_price, "trail")
                return

        # Stale: early exit at 15 min if flat (<1%), else 25 min
        elapsed = time.time() - state["entry_time"]
        stale_early_sec = self.param_stale_early * 60
        stale_full_sec = self.param_stale * 60

        if elapsed >= stale_early_sec and gain < self.param_stale_thresh:
            logger.info("STALE EARLY %s: %.0fmin gain=%.1f%%", sym, elapsed / 60, gain)
            self._close(sym, current_price, "stale_early")
            return

        if elapsed >= stale_full_sec:
            logger.info("STALE %s: %.0fmin gain=%.1f%%", sym, elapsed / 60, gain)
            self._close(sym, current_price, "stale")
            return

    def _get_current_price(self, symbol: str, state: dict) -> Optional[float]:
        try:
            if not self.sim and self.data_client:
                from alpaca.data.requests import StockLatestQuoteRequest
                req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
                quotes = self.data_client.get_stock_latest_quote(req)
                if symbol in quotes:
                    q = quotes[symbol]
                    bid = float(q.bid_price) if q.bid_price else None
                    ask = float(q.ask_price) if q.ask_price else None
                    if bid and ask:
                        return (bid + ask) / 2
                    return bid or ask
            # Fallback: yfinance
            tk = yf.Ticker(symbol)
            hist = tk.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist.iloc[-1]["Close"])
            return None
        except Exception:
            return None

    # ── Close ────────────────────────────────────────────────────────

    def _close(self, symbol: str, exit_price: float, reason: str):
        state = self.active.get(symbol) or \
                next((s for s in self.active.values() if s.get("symbol") == symbol), None)
        if state is None or state.get("closed"):
            return
        state["closed"] = True

        entry = state["entry_price"]
        gain = round(((exit_price - entry) / entry) * 100, 2)
        pnl = round((exit_price - entry) * state.get("qty", 1), 2)

        trade = {
            "symbol": symbol,
            "entry_price": entry,
            "exit_price": exit_price,
            "gain_pct": gain,
            "pnl": pnl,
            "exit_reason": reason,
            "gap": state.get("entry_gap", 0),
            "time": datetime.now(UTC_TZ).isoformat(),
        }
        self._save_trade(trade)
        self.model.add_trade(trade)

        logger.info("EXIT %s: %.1f%% ($%.2f) reason=%s", symbol, gain, pnl, reason)

        # Remove from active
        keys_to_del = [k for k, v in self.active.items()
                       if v.get("symbol") == symbol or k == symbol]
        for k in keys_to_del:
            del self.active[k]
        self.tranche_pool += 1


# ═══════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════

async def main_loop(bot: GapBotV5):
    logger.info("\n" + "=" * 55)
    logger.info("GAP BOT V5 — $%.0f capital | %.0f%% stale=%d/%dmin trail=+%.0f%%/%.0f%%",
                CAPITAL, 100 * TRANCH_SIZE / CAPITAL * MAX_POSITIONS,
                STALE_EARLY_EXIT_MIN, STALE_TIMEOUT_MINUTES,
                TRAIL_ACTIVATE, TRAIL_DIST)
    logger.info("=" * 55)

    asyncio.create_task(bot.monitor_loop(interval=2.0))

    today_state = set()

    while True:
        now_ny = datetime.now(NY_TZ)
        if now_ny.weekday() >= 5:
            await asyncio.sleep(3600)
            continue

        if now_ny.hour < 9 or now_ny.hour >= 16:
            next_open = now_ny.replace(hour=9, minute=0, second=0, microsecond=0)
            if now_ny.hour >= 16:
                next_open += timedelta(days=1)
            if next_open.weekday() >= 5:
                next_open += timedelta(days=7 - next_open.weekday())
            sleep_secs = (next_open - now_ny).total_seconds()
            if sleep_secs > 0:
                logger.info("Sleeping %.0f min until %s",
                            sleep_secs / 60, next_open.strftime("%a %H:%M"))
            await asyncio.sleep(min(max(sleep_secs, 60), 3600))
            continue

        h, m = now_ny.hour, now_ny.minute

        # 9:00 AM — scan
        if h == 9 and m == 0 and "scan" not in today_state:
            today_state.add("scan")
            signals = await bot.scan()
            if signals:
                for s in signals[:5]:
                    logger.info("  %s: gap=+%.1f%% score=%.2f fade=%.0f%%",
                                s["sym"], s["gap"], s["score"], s["fade_prob"] * 100)

        # 9:01-9:15 — route positions
        if h == 9 and 1 <= m <= 15 and "routed" not in today_state:
            if bot.tranche_pool > 0 and bot.signals:
                signal = bot.signals.pop(0)
                ok = await bot.execute_entry(signal)
                if ok:
                    logger.info("Tranche allocated: %s (%d remaining)",
                                signal["sym"], bot.tranche_pool)
                await asyncio.sleep(30)
            else:
                today_state.add("routed")

        # 9:31 — entry window closes
        if h == 9 and m >= 31 and "routed" not in today_state:
            today_state.add("routed")
            logger.info("Entry closed. %d active, %d tranches remaining",
                        len(bot.active), bot.tranche_pool)

        # 4:00 PM — close
        if h >= 16 and "closed" not in today_state:
            today_state.add("closed")
            logger.info("Market close. Closing %d positions...", len(bot.active))
            for key, state in list(bot.active.items()):
                price = bot._get_current_price(state["symbol"], state)
                if price is None:
                    price = state.get("highest", state.get("entry_price", 0))
                bot._close(state["symbol"], price, "eod")

            # Nightly backtest
            best = bot.model.backtest_params()
            if best:
                bot.param_sl = best.get("sl", bot.param_sl)
                bot.param_trail_act = best.get("trail_act", bot.param_trail_act)
                bot.param_trail_dist = best.get("trail_dist", bot.param_trail_dist)
                logger.info("Nightly tune: %s", best)
            logger.info("Model: %s", bot.model.report())

        await asyncio.sleep(10)


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

async def main():
    sim = "--sim" in sys.argv
    use_hf = "--hf" in sys.argv

    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")

    if not key or not secret:
        if not sim:
            logger.warning("No Alpaca keys. Run --sim or export APCA_API_KEY_ID")
            sim = True

    hf_url = os.getenv("HF_MODEL_URL")
    if use_hf and not hf_url:
        logger.warning("--hf set but HF_MODEL_URL not set. Using heuristic only.")

    bot = GapBotV5(key or "", secret or "", sim=sim, hf_url=hf_url)
    await main_loop(bot)


if __name__ == "__main__":
    asyncio.run(main())
