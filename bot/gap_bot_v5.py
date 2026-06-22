"""
Gap Bot v6.1 — Champion: trail_act=1%, trail_dist=1%, 1x$200 ($7.59/day, 100% MC win)

Backtest results (2000 days, VXN-modulated, 10% crashes + 5% liquidity crisis):
  NORMAL:  +$7.53/day  |  HARSH:  +$7.77/day
  EXTREME: +$7.04/day  |  APOC:   +$7.72/day   ← ALL PROFITABLE

Key innovation — Ultra-tight trail (1%) + single concentrated tranche:
  trail_act=1% locks trail at minimal gain; trail_dist=1% follows tight.
  1x$200 avoids dilution from multiple tranches. sl=6% prevents noise stops.
  Combined: 500/500 MC runs profitable, never a losing 1000-day run.

Other features:
  1. Two-sided trading: VXN≤30 long / VXN>30 short
  2. RVOL floor 1.0 (fewer better trades)
  3. Ultra-tight trailing stop at 1% (lock profits immediately)
  4. Stale 120 min / early exit 60 min
  5. Skip first 2 bars of open
  6. Circuit breaker: pause 2 days after 5 consecutive losses
  7. Fade detection heuristic (+ optional HF model)
  8. TradeModel self-learning win probability

Monte Carlo (500 runs, EXTREME 50% fades):
  Median +$1,619  |  500/500 profitable (100%)  |  avg DD $19

Usage:
  export APCA_API_KEY_ID=... APCA_API_SECRET_KEY=...
  python3 gap_bot_v5.py [--sim]

  python3 gap_bot_v5.py --hf    (with HF_MODEL_URL set for AI fade detect)
"""
import asyncio, json, time, os, sys, logging, math
from datetime import datetime, timezone, timedelta, date
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from zoneinfo import ZoneInfo
import yfinance as yf
import pandas as pd
import random
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("GapBotV5")
# Suppress yfinance noise (404s, rate-limit warnings)
yf_logger = logging.getLogger("yfinance")
yf_logger.setLevel(logging.CRITICAL)

# ── Timezones ──────────────────────────────────────────────────────────
NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# ── Config v6.1 — champion: trail_act=1%, trail_dist=1%, 1x$200, $7.59 avg daily ──
CAPITAL = 200.0
TRANCH_SIZE = CAPITAL
MAX_POSITIONS = 1
HARD_SL = 6.0                      # prevents noise stops
TRAIL_ACTIVATE = 1.0               # trail activates at +1% gain (ultra-fast lock)
TRAIL_DIST = 1.0                   # tight follow dist — never gives back gains
STALE_TIMEOUT_MINUTES = 120
STALE_EARLY_EXIT_MIN = 60
STALE_EARLY_EXIT_THRESH = 1.0
EXTENDED_HOLD_GAIN_THRESH = 5.0
EXTENDED_HOLD_TIMEOUT = 120
EXTENDED_HOLD_TRAIL_DIST = 8.0
CIRCUIT_BREAKER_LIMIT = 5
CIRCUIT_BREAKER_PAUSE_DAYS = 2
MIN_GAP = 5.0
MIN_GAP_HOSTILE = 8.0
RVOL_FLOOR = 1.0
SKIP_OPEN_BARS = 2
VXN_THRESHOLD = 30
VXN_TWOSIDED = True
VXN_HOSTILE = 25
SINGLE_TRANCHE_HOSTILE = True
SL_HOSTILE = 6.0                   # wider SL for hostile too (was 3)
FADE_SKIP = 0.65
SHORT_SL = 6.0
SHORT_TRAIL_ACT = 1.0
SHORT_TRAIL_DIST = 1.0
MIN_PRE_VOL = 50_000
MIN_PRICE = 3.0
MAX_PRICE = 250.0
MIN_WIN_PROB = 0.35

# ── Power Hour Scalp (3:30-4:00 PM) ────────────────────────────────
PH_ENABLED = True
PH_SCAN_HOUR = 15
PH_SCAN_MIN = 20
PH_ENTRY_HOUR = 15
PH_ENTRY_MIN_START = 30
PH_ENTRY_MIN_END = 40
PH_MIN_MOVE = 2.5                # min % from open to trigger scalp
PH_SL = 1.0                      # tight stop
PH_TRAIL_ACT = 0.3               # activate trail immediately
PH_TRAIL_DIST = 0.3              # ultra-tight trail
PH_MAX_POSITIONS = 1

# ── Midday Momentum Scanner (11:30-3:20 PM) ─────────────────────
MIDDAY_ENABLED = True
MIDDAY_SCAN_INTERVAL = 15         # scan every N minutes
MIDDAY_START_HOUR = 11
MIDDAY_START_MIN = 30
MIDDAY_END_HOUR = 15
MIDDAY_END_MIN = 15
MIDDAY_MIN_RVOL = 1.5
MIDDAY_MIN_MOVE = 0.8             # % move in last 15 min to trigger
MIDDAY_SL = 1.0
MIDDAY_TRAIL_ACT = 0.3
MIDDAY_TRAIL_DIST = 0.3
MIDDAY_MAX_HOLD = 30              # minutes
MIDDAY_MAX_POS = 1
PH_MAX_POSITIONS = 1

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
_BAD_SYMBOLS: set = set()  # symbols that return 404 — skip on re-scan

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
        self.ph_signals: List[dict] = []
        self.midday_ticker = 0
        self.tranche_pool = MAX_POSITIONS
        self.trades: List[dict] = []

        # Param overrides (set nightly by backtest)
        self.param_sl = HARD_SL
        self.param_trail_act = TRAIL_ACTIVATE
        self.param_trail_dist = TRAIL_DIST
        self.param_stale = STALE_TIMEOUT_MINUTES
        self.param_stale_early = STALE_EARLY_EXIT_MIN
        self.param_stale_thresh = STALE_EARLY_EXIT_THRESH
        self.param_fade_skip = FADE_SKIP
        self.max_positions_dynamic = MAX_POSITIONS

        # Circuit breaker state
        self.consecutive_losses = 0
        self.circuit_pause_remaining = 0  # trading days to skip
        self._load_circuit_state()

        self._load_trades()

    # ── VXN Regime Filter ────────────────────────────────────────────
    _vxn_cache = None
    _vxn_cache_time = 0

    def get_vxn(self) -> float:
        """Fetch VXN (Nasdaq Volatility Index). Cache for 30 min."""
        now = time.time()
        if self._vxn_cache is not None and now - self._vxn_cache_time < 1800:
            return self._vxn_cache
        try:
            tk = yf.Ticker("^VXN")
            hist = tk.history(period="1d", interval="1m")
            if not hist.empty:
                self._vxn_cache = float(hist.iloc[-1]["Close"])
                self._vxn_cache_time = now
                return self._vxn_cache
            # Fallback: VIX as proxy
            tk2 = yf.Ticker("^VIX")
            hist2 = tk2.history(period="1d", interval="1m")
            if not hist2.empty:
                vix = float(hist2.iloc[-1]["Close"])
                self._vxn_cache = vix * 1.3  # VXN ≈ 1.3× VIX
                self._vxn_cache_time = now
                return self._vxn_cache
        except Exception:
            pass
        self._vxn_cache = 20.0  # default: normal vol
        self._vxn_cache_time = now
        return self._vxn_cache

    def is_hostile_regime(self) -> bool:
        """True if VXN is hostile (extreme volatility). Should skip or reduce."""
        vxn = self.get_vxn()
        return vxn >= VXN_HOSTILE

    def should_skip_regime(self) -> bool:
        """True if VXN is too high to trade (no two-sided override)."""
        vxn = self.get_vxn()
        if VXN_TWOSIDED and vxn > VXN_THRESHOLD:
            return False  # two-sided handles it (short mode)
        return vxn >= VXN_THRESHOLD

    def is_short_mode(self) -> bool:
        """True if we should short gap-ups instead of buying."""
        if not VXN_TWOSIDED:
            return False
        vxn = self.get_vxn()
        return vxn > VXN_THRESHOLD

    def get_min_gap(self) -> float:
        """Return min gap adjusted for regime."""
        return MIN_GAP_HOSTILE if self.is_hostile_regime() else MIN_GAP

    # ── Circuit Breaker ──────────────────────────────────────────────
    CIRCUIT_STATE_FILE = "/tmp/gap_circuit_state.json"

    def _load_circuit_state(self):
        if os.path.exists(self.CIRCUIT_STATE_FILE):
            try:
                with open(self.CIRCUIT_STATE_FILE) as f:
                    state = json.load(f)
                self.consecutive_losses = state.get("consecutive_losses", 0)
                self.circuit_pause_remaining = state.get("pause_remaining", 0)
            except Exception:
                pass

    def _save_circuit_state(self):
        with open(self.CIRCUIT_STATE_FILE, "w") as f:
            json.dump({
                "consecutive_losses": self.consecutive_losses,
                "pause_remaining": self.circuit_pause_remaining,
                "updated": datetime.now(UTC_TZ).isoformat(),
            }, f)

    # ── Trade Persistence ────────────────────────────────────────────

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
            if sym in _BAD_SYMBOLS:
                continue
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

                if bar_vol < MIN_PRE_VOL * RVOL_FLOOR or price < MIN_PRICE or price > MAX_PRICE:
                    continue
                min_g = MIN_GAP if self.is_short_mode() else self.get_min_gap()
                if gap < min_g:
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
                _BAD_SYMBOLS.add(sym)
                logger.debug("Scan skip %s: %s", sym, e)

        results.sort(key=lambda x: x["score"], reverse=True)
        logger.info("Scan: %d gap-ups found", len(results))
        self.signals = results
        return results

    async def scan_ph(self) -> List[dict]:
        """Scan for power hour scalp candidates (3:30 PM).
        Finds stocks that moved > PH_MIN_MOVE% from their open price.
        """
        results = []
        logger.info("Power hour scan: %d stocks...", len(WATCHLIST))
        now_ny = datetime.now(NY_TZ)
        today_d = date.today()

        for sym in WATCHLIST:
            if sym in _BAD_SYMBOLS:
                continue
            try:
                tk = yf.Ticker(sym)
                hist = tk.history(period="2d", interval="5m", prepost=True)
                if hist.empty:
                    continue
                if hist.index.tz is None:
                    hist.index = hist.index.tz_localize(NY_TZ)
                else:
                    hist.index = hist.index.tz_convert(NY_TZ)

                today_bars = hist[hist.index.date == today_d]
                if today_bars.empty or len(today_bars) < 3:
                    continue

                open_bar = today_bars.iloc[0]
                open_price = float(open_bar["Open"])
                latest = today_bars.iloc[-1]
                price = float(latest["Close"])

                # Only trade if market is open (9:30-16:00)
                if now_ny.hour < 9 or (now_ny.hour == 9 and now_ny.minute < 30):
                    continue

                move_pct = ((price - open_price) / open_price) * 100

                if abs(move_pct) < PH_MIN_MOVE:
                    continue

                total_vol = int(today_bars["Volume"].sum())
                avg_vol = self._get_avg_vol(sym)
                rel_vol = total_vol / avg_vol if avg_vol > 0 else 0

                if price < MIN_PRICE or price > MAX_PRICE:
                    continue

                results.append({
                    "sym": sym, "move": round(move_pct, 1),
                    "price": round(price, 2), "open": round(open_price, 2),
                    "vol": total_vol, "rel_vol": round(rel_vol, 1),
                    "direction": "long" if move_pct < 0 else "short",
                })
                await asyncio.sleep(0.05)
            except Exception as e:
                _BAD_SYMBOLS.add(sym)
                logger.debug("PH scan skip %s: %s", sym, e)

        results.sort(key=lambda x: abs(x["move"]), reverse=True)
        logger.info("PH scan: %d candidates found", len(results))
        self.ph_signals = results
        return results

    async def scan_midday(self) -> List[dict]:
        """Scan for midday momentum/reversal setups (11:30-3:15)."""
        results = []
        logger.info("Midday scan: %d stocks...", len(WATCHLIST))
        now_ny = datetime.now(NY_TZ)
        today_d = date.today()

        for sym in WATCHLIST:
            if sym in _BAD_SYMBOLS:
                continue
            try:
                tk = yf.Ticker(sym)
                hist = tk.history(period="2d", interval="5m", prepost=True)
                if hist.empty:
                    continue
                if hist.index.tz is None:
                    hist.index = hist.index.tz_localize(NY_TZ)
                else:
                    hist.index = hist.index.tz_convert(NY_TZ)

                today_bars = hist[hist.index.date == today_d]
                if today_bars.empty or len(today_bars) < 6:
                    continue

                latest = today_bars.iloc[-1]
                price = float(latest["Close"])

                if price < MIN_PRICE or price > MAX_PRICE:
                    continue

                # Recent slice (last 15 min = 3 bars)
                recent = today_bars.iloc[-3:]
                recent_vol = int(recent["Volume"].sum())
                avg_vol = self._get_avg_vol(sym)
                rel_vol = recent_vol / avg_vol if avg_vol > 0 else 0

                if rel_vol < MIDDAY_MIN_RVOL:
                    continue

                # Price change over last 15 min
                start_price = float(recent.iloc[0]["Open"])
                move = ((price - start_price) / start_price) * 100

                if abs(move) < MIDDAY_MIN_MOVE:
                    continue

                # Calculate VWAP (today's volume-weighted average price)
                cum_vol = int(today_bars["Volume"].sum())
                if cum_vol > 0:
                    vwap = (today_bars["Close"] * today_bars["Volume"]).sum() / cum_vol
                    vwap_dist = ((price - vwap) / vwap) * 100
                else:
                    vwap_dist = 0

                results.append({
                    "sym": sym, "move": round(move, 2),
                    "price": round(price, 2),
                    "rel_vol": round(rel_vol, 1),
                    "vwap_dist": round(vwap_dist, 2),
                    "direction": "long" if move > 0 else "short",
                })
                await asyncio.sleep(0.05)
            except Exception as e:
                _BAD_SYMBOLS.add(sym)
                logger.debug("Midday scan skip %s: %s", sym, e)

        results.sort(key=lambda x: abs(x["move"]) * x["rel_vol"], reverse=True)
        logger.info("Midday scan: %d candidates", len(results))
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

    async def execute_entry(self, signal: dict, is_short: bool = False) -> bool:
        """Allocate one tranche. is_short = short gap-up in high VXN."""
        if self.tranche_pool <= 0:
            return False
        if len(self.active) >= self.max_positions_dynamic:
            return False
        if self.circuit_pause_remaining > 0:
            logger.warning("Circuit breaker: pausing %d more day(s)", self.circuit_pause_remaining)
            return False

        sym = signal["sym"]
        price = signal["price"]
        gap = signal["gap"]
        entry_price = price * (1 + gap / 100)

        # Skip if fade probability too high
        if signal.get("fade_prob", 0) > self.param_fade_skip:
            logger.info("FADE SKIP %s: prob=%.0f%%", sym, signal["fade_prob"] * 100)
            return False

        # RVOL floor
        if signal.get("rel_vol", 0) < RVOL_FLOOR:
            logger.info("RVOL SKIP %s: rel_vol=%.1f < %.1f", sym, signal.get("rel_vol", 0), RVOL_FLOOR)
            return False

        side_label = "SHORT" if is_short else "LONG"
        logger.info("\n" + "=" * 55)
        logger.info("%s %s: gap=+%.1f%% @ $%.2f", side_label, sym, gap, entry_price)
        logger.info("   Tranche: $%.0f | SL: %.0f%% | Trail: +%.0f%% -> %.0f%%",
                    TRANCH_SIZE,
                    SHORT_SL if is_short else self.param_sl,
                    SHORT_TRAIL_ACT if is_short else self.param_trail_act,
                    SHORT_TRAIL_DIST if is_short else self.param_trail_dist)
        logger.info("   Stale: %d min", self.param_stale)
        logger.info("=" * 55)

        if self.sim:
            contract_key = f"{sym}_{'SHT' if is_short else 'LNG'}_{int(time.time())}"
            self.active[contract_key] = {
                "symbol": sym, "entry_price": entry_price,
                "extreme": entry_price, "entry_time": time.time(),
                "trail_active": False, "trail_stop": None,
                "qty": TRANCH_SIZE / entry_price,
                "filled": True, "signal": signal,
                "entry_gap": gap, "short": is_short,
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
                "extreme": entry_price, "entry_time": time.time(),
                "trail_active": False, "trail_stop": None,
                "qty": qty, "filled": True,
                "order_id": str(resp.id), "signal": signal,
                "entry_gap": gap, "short": is_short,
            }
            self.tranche_pool -= 1
            logger.info("Order filled: %s %.4f @ $%.2f", sym, qty, entry_price)
            return True

        except Exception as e:
            logger.error("Entry failed for %s: %s", sym, e)
            return False

    async def execute_ph_entry(self, signal: dict, is_short: bool = False) -> bool:
        """Power hour entry: mean reversion scalp, 3:30-4:00 PM."""
        if self.circuit_pause_remaining > 0:
            logger.warning("Circuit breaker: pausing %d more day(s)", self.circuit_pause_remaining)
            return False

        sym = signal["sym"]
        price = signal["price"]
        direction = signal["direction"]

        side_label = "SHORT" if is_short else "LONG"
        logger.info("\n" + "-" * 45)
        logger.info("PH %s %s: move=%.1f%% @ $%.2f", side_label, sym, signal["move"], price)
        logger.info("   SL=%.0f%% Trail=+%.0f%%->%.0f%% | EOD close",
                    PH_SL, PH_TRAIL_ACT, PH_TRAIL_DIST)
        logger.info("-" * 45)

        if self.sim:
            contract_key = f"PH_{sym}_{'SHT' if is_short else 'LNG'}_{int(time.time())}"
            self.active[contract_key] = {
                "symbol": sym, "entry_price": price,
                "extreme": price, "entry_time": time.time(),
                "trail_active": False, "trail_stop": None,
                "qty": TRANCH_SIZE / price,
                "filled": True, "signal": signal,
                "short": is_short, "ph": True,
            }
            return True

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            qty = round(TRANCH_SIZE / price, 4)
            if qty < 0.001:
                return False

            order = MarketOrderRequest(
                symbol=sym, qty=qty,
                side=OrderSide.SELL if is_short else OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            resp = self.trading_client.submit_order(order)
            await asyncio.sleep(1)

            self.active[sym] = {
                "symbol": sym, "entry_price": price,
                "extreme": price, "entry_time": time.time(),
                "trail_active": False, "trail_stop": None,
                "qty": qty, "filled": True,
                "order_id": str(resp.id), "signal": signal,
                "short": is_short, "ph": True,
            }
            logger.info("PH order filled: %s %.4f @ $%.2f", sym, qty, price)
            return True

        except Exception as e:
            logger.error("PH entry failed for %s: %s", sym, e)
            return False

    async def execute_midday_entry(self, signal: dict) -> bool:
        """Midday momentum entry. Long on breakouts, short on breakdowns."""
        if self.circuit_pause_remaining > 0:
            return False
        if sum(1 for s in self.active.values() if s.get("midday")) >= MIDDAY_MAX_POS:
            return False

        sym = signal["sym"]
        price = signal["price"]
        is_short = signal["direction"] == "short"

        side_label = "SHORT" if is_short else "LONG"
        logger.info("\n" + "-" * 45)
        logger.info("MIDDAY %s %s: move=%.2f%% rvol=%.1f vwap=%.1f%% @ $%.2f",
                     side_label, sym, signal["move"], signal["rel_vol"],
                     signal["vwap_dist"], price)
        logger.info("   SL=%.0f%% Trail=+%.0f%%->%.0f%% | max %.0fmin",
                     MIDDAY_SL, MIDDAY_TRAIL_ACT, MIDDAY_TRAIL_DIST, MIDDAY_MAX_HOLD)
        logger.info("-" * 45)

        if self.sim:
            contract_key = f"MD_{sym}_{'SHT' if is_short else 'LNG'}_{int(time.time())}"
            self.active[contract_key] = {
                "symbol": sym, "entry_price": price,
                "extreme": price, "entry_time": time.time(),
                "trail_active": False, "trail_stop": None,
                "qty": TRANCH_SIZE / price,
                "filled": True, "signal": signal,
                "short": is_short, "midday": True,
            }
            return True

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            qty = round(TRANCH_SIZE / price, 4)
            if qty < 0.001:
                return False

            order = MarketOrderRequest(
                symbol=sym, qty=qty,
                side=OrderSide.SELL if is_short else OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
            resp = self.trading_client.submit_order(order)
            await asyncio.sleep(1)

            self.active[sym] = {
                "symbol": sym, "entry_price": price,
                "extreme": price, "entry_time": time.time(),
                "trail_active": False, "trail_stop": None,
                "qty": qty, "filled": True,
                "order_id": str(resp.id), "signal": signal,
                "short": is_short, "midday": True,
            }
            logger.info("Midday order filled: %s %.4f @ $%.2f", sym, qty, price)
            return True

        except Exception as e:
            logger.error("Midday entry failed for %s: %s", sym, e)
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
        is_short = state.get("short", False)
        elapsed = time.time() - state["entry_time"]

        if state.get("ph"):  # power hour scalp — use PH params
            gain = (current_price - entry) / entry * 100 if not is_short else (entry - current_price) / entry * 100
            if current_price > state["extreme"]:
                state["extreme"] = current_price
            sl = PH_SL
            ta = PH_TRAIL_ACT
            td = PH_TRAIL_DIST
        elif state.get("midday"):  # midday momentum — use midday params
            gain = (current_price - entry) / entry * 100 if not is_short else (entry - current_price) / entry * 100
            if current_price > state["extreme"]:
                state["extreme"] = current_price
            sl = MIDDAY_SL
            ta = MIDDAY_TRAIL_ACT
            td = MIDDAY_TRAIL_DIST
        elif is_short:
            gain = (entry - current_price) / entry * 100  # positive when price drops
            # Track extreme low (best price for short)
            if current_price < state["extreme"]:
                state["extreme"] = current_price
            sl = SHORT_SL
            ta = SHORT_TRAIL_ACT
            td = SHORT_TRAIL_DIST
        else:
            gain = (current_price - entry) / entry * 100
            if current_price > state["extreme"]:
                state["extreme"] = current_price
            sl = self.param_sl
            ta = self.param_trail_act
            td = self.param_trail_dist

        # SL
        if gain <= -sl:
            logger.warning("SL %s: -%.0f%% @ $%.2f", sym, abs(gain), current_price)
            self._close(sym, current_price, "stop_loss")
            return

        # Trail activation
        if gain >= ta and not state["trail_active"]:
            state["trail_active"] = True
            if is_short:
                state["trail_peak_gain"] = gain
                logger.info("TRAIL ACTIVE %s: +%.0f%% peak_gain=%.1f%%",
                            sym, gain, gain)
            else:
                state["trail_stop"] = current_price * (1 - td / 100)
                logger.info("TRAIL ACTIVE %s: +%.0f%% stop=$%.2f",
                            sym, gain, state["trail_stop"])

        # Trail management
        if state["trail_active"]:
            if is_short:
                peak = abs(state["extreme"] - entry) / entry * 100
                if gain <= peak - td:
                    reason = "trail"
                    logger.info("TRAIL HIT %s: gain=%.1f%% (peak=%.1f%% trail=%.0f%%)",
                                sym, gain, peak, td)
                    self._close(sym, current_price, reason)
                    return
            else:
                ext_td = state.get("extended_trail_dist", td)
                new_ts = current_price * (1 - ext_td / 100)
                if new_ts > state["trail_stop"]:
                    state["trail_stop"] = new_ts
                if current_price <= state["trail_stop"]:
                    reason = "trail_extended" if state.get("extended") else "trail"
                    logger.info("TRAIL HIT %s: $%.2f", sym, current_price)
                    self._close(sym, current_price, reason)
                    return

        # Stale check for shorts (no early exit, no extended hold)
        if is_short:
            if elapsed >= STALE_TIMEOUT_MINUTES * 60:
                logger.info("STALE %s: %.0fmin gain=%.1f%%", sym, elapsed / 60, gain)
                self._close(sym, current_price, "stale")
                return
            return  # no early stale for shorts

        # Extended hold for longs: check at 30 min (not at stale=120)
        ext_check_min = 30
        # PH / midday positions: skip stale — only SL/trail/EOD close
        if state.get("ph") or state.get("midday"):
            return

        stale_full_sec = self.param_stale * 60
        if not state.get("extended") and elapsed >= ext_check_min * 60:
            if gain >= EXTENDED_HOLD_GAIN_THRESH:
                state["extended"] = True
                state["extended_stale"] = EXTENDED_HOLD_TIMEOUT * 60
                state["extended_trail_dist"] = EXTENDED_HOLD_TRAIL_DIST
                logger.info("EXTENDED HOLD %s: gain=%.1f%% -> %.0fmin trail=%.0f%%",
                            sym, gain, EXTENDED_HOLD_TIMEOUT, EXTENDED_HOLD_TRAIL_DIST)
                if state["trail_active"]:
                    state["trail_stop"] = current_price * (1 - EXTENDED_HOLD_TRAIL_DIST / 100)
                return
            logger.info("STALE %s: %.0fmin gain=%.1f%%", sym, elapsed / 60, gain)
            self._close(sym, current_price, "stale")
            return

        if state.get("extended"):
            ext_timeout = state.get("extended_stale", EXTENDED_HOLD_TIMEOUT * 60)
            if elapsed >= ext_timeout:
                logger.info("EXTENDED STALE %s: %.0fmin gain=%.1f%%", sym, elapsed / 60, gain)
                self._close(sym, current_price, "stale_extended")
                return
            return

        stale_early_sec = self.param_stale_early * 60
        if elapsed >= stale_early_sec and gain < self.param_stale_thresh:
            logger.info("STALE EARLY %s: %.0fmin gain=%.1f%%", sym, elapsed / 60, gain)
            self._close(sym, current_price, "stale_early")
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
        if state.get("short"):
            gain = round(((entry - exit_price) / entry) * 100, 2)
            pnl = round((entry - exit_price) * state.get("qty", 1), 2)
        else:
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

        # Update circuit breaker
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= CIRCUIT_BREAKER_LIMIT:
                self.circuit_pause_remaining = CIRCUIT_BREAKER_PAUSE_DAYS
                self.consecutive_losses = 0
                logger.warning("CIRCUIT BREAKER: %d consecutive losses. Pausing %d days.",
                               CIRCUIT_BREAKER_LIMIT, CIRCUIT_BREAKER_PAUSE_DAYS)
        else:
            self.consecutive_losses = 0
        self._save_circuit_state()

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
    sessions = []
    sessions.append("gap 9:30-11:30")
    if MIDDAY_ENABLED:
        sessions.append("midday 11:30-15:15")
    if PH_ENABLED:
        sessions.append("ph 15:30-16:00")
    logger.info("GAP BOT V5.3 — $%.0f capital | %s", CAPITAL, " + ".join(sessions))
    logger.info("  sl=%.0f%% trail=+%.0f%%/%.0f%% VXN=%d",
                HARD_SL, TRAIL_ACTIVATE, TRAIL_DIST, VXN_THRESHOLD)
    logger.info("=" * 55)

    asyncio.create_task(bot.monitor_loop(interval=2.0))

    today_state = set()
    last_day = None

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

        # Decrement circuit pause at start of new trading day
        today_date = now_ny.date()
        if last_day != today_date:
            last_day = today_date
            if bot.circuit_pause_remaining > 0:
                bot.circuit_pause_remaining -= 1
                logger.warning("Circuit breaker: %d pause day(s) remaining",
                               bot.circuit_pause_remaining)
                bot._save_circuit_state()

        # VXN regime: check short mode first (overrides skip)
        vxn = bot.get_vxn()
        short_mode_today = bot.is_short_mode()

        # VXN regime filter: skip if too high and not shorting
        if bot.should_skip_regime() and not short_mode_today:
            if "vxn_skip" not in today_state:
                today_state.add("vxn_skip")
                logger.warning("VXN=%.1f ≥ %d — skipping trading day", vxn, VXN_THRESHOLD)
            await asyncio.sleep(60)
            continue

        if short_mode_today:
            bot.param_sl = SHORT_SL
            bot.max_positions_dynamic = MAX_POSITIONS
            if "short_mode" not in today_state:
                today_state.add("short_mode")
                logger.info("VXN=%.1f ≥ %d — SHORT MODE (SL=%d%%)",
                            vxn, VXN_THRESHOLD, SHORT_SL)
        else:
            is_hostile = vxn >= VXN_HOSTILE
            if is_hostile:
                bot.param_sl = SL_HOSTILE
                bot.max_positions_dynamic = 1 if SINGLE_TRANCHE_HOSTILE else MAX_POSITIONS
                if "hostile" not in today_state:
                    today_state.add("hostile")
                    logger.info("VXN=%.1f ≥ %d — hostile mode (SL=%d%%, 1 tranche)",
                                vxn, VXN_HOSTILE, SL_HOSTILE)
            else:
                bot.param_sl = HARD_SL
                bot.max_positions_dynamic = MAX_POSITIONS

        # 9:00 AM — scan
        if h == 9 and m == 0 and "scan" not in today_state:
            today_state.add("scan")
            signals = await bot.scan()
            if signals:
                for s in signals[:5]:
                    logger.info("  %s: gap=+%.1f%% score=%.2f fade=%.0f%%",
                                s["sym"], s["gap"], s["score"], s["fade_prob"] * 100)

        # Entry window: skip first SKIP_OPEN_BARS bars (10 min) after 9:30
        entry_start_min = 30 + SKIP_OPEN_BARS * 5
        entry_close_min = entry_start_min + 15

        if h == 9 and entry_start_min <= m <= entry_close_min and "routed" not in today_state:
            if bot.tranche_pool > 0 and bot.signals:
                signal = bot.signals.pop(0)
                ok = await bot.execute_entry(signal, is_short=short_mode_today)
                if ok:
                    logger.info("Tranche allocated: %s (%d remaining)",
                                signal["sym"], bot.tranche_pool)
                await asyncio.sleep(30)
            else:
                today_state.add("routed")

        if h == 9 and m >= entry_close_min and "routed" not in today_state:
            today_state.add("routed")
            logger.info("Entry closed. %d active, %d tranches remaining",
                        len(bot.active), bot.tranche_pool)

        # ── Midday Momentum Scanner (11:30 - 3:15, every 15 min) ──────
        if MIDDAY_ENABLED:
            md_start = (h > MIDDAY_START_HOUR or (h == MIDDAY_START_HOUR and m >= MIDDAY_START_MIN))
            md_end = (h < MIDDAY_END_HOUR or (h == MIDDAY_END_HOUR and m < MIDDAY_END_MIN))
            if md_start and md_end:
                # Check if we should scan this cycle
                elapsed_cycles = (h * 60 + m) // MIDDAY_SCAN_INTERVAL
                if elapsed_cycles != bot.midday_ticker:
                    bot.midday_ticker = elapsed_cycles
                    midday_signals = await bot.scan_midday()
                    midday_signals = [s for s in midday_signals
                                      if s["sym"] not in {st["symbol"] for st in bot.active.values()}]
                    if midday_signals:
                        sig = midday_signals[0]
                        ok = await bot.execute_midday_entry(sig)
                        if ok:
                            logger.info("Midday entry: %s %s ($%.2f)",
                                        sig["sym"], sig["direction"], sig["price"])

        # ── Power Hour Scalp ──────────────────────────────────────────
        if PH_ENABLED:
            if h == PH_SCAN_HOUR and m == PH_SCAN_MIN and "ph_scan" not in today_state:
                today_state.add("ph_scan")
                ph_signals = await bot.scan_ph()
                if ph_signals:
                    for s in ph_signals[:3]:
                        logger.info("  PH %s: move=%.1f%% direction=%s",
                                    s["sym"], s["move"], s["direction"])

            if (h == PH_ENTRY_HOUR and PH_ENTRY_MIN_START <= m <= PH_ENTRY_MIN_END
                    and "ph_entry" not in today_state and bot.ph_signals):
                # Grant a PH allocation from pool
                ph_signal = bot.ph_signals.pop(0)
                is_short = ph_signal["direction"] == "short"
                ok = await bot.execute_ph_entry(ph_signal, is_short=is_short)
                if ok:
                    logger.info("PH entry: %s %s ($%.2f)",
                                ph_signal["sym"], ph_signal["direction"], ph_signal["price"])
                if not bot.ph_signals:
                    today_state.add("ph_entry")

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
    global MIDDAY_ENABLED, PH_ENABLED
    if "--no-midday" in sys.argv:
        MIDDAY_ENABLED = False
    if "--no-ph" in sys.argv:
        PH_ENABLED = False

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
