"""
Options Gap Bot v1 — Gamma Acceleration on High-Volatility Gap Days.

Substitutes linear stock equity with OTM option contracts (0.30 delta,
≤7DTE) to exploit exponential gamma leverage on pre-market gap catalysts.

Capital: $200 split into 4 × $50 independent tranches.
Each tranche buys 1+ integer option contracts.
No bracket orders (unsupported on options) — uses internal async
state machine for risk management.

Infrastructure:
  - self.active_tracked_options: dict tracking all open positions
  - Manual trailing stop / hard SL / stale kill switch
  - Timezone: America/New_York

Usage:
  export APCA_API_KEY_ID=...
  export APCA_API_SECRET_KEY=...
  python3 options_gap_bot.py [--sim]
"""
import asyncio, json, time, os, sys, logging, math, random
from datetime import datetime, timedelta, date
from typing import Dict, Optional, List, Any
from collections import defaultdict
from zoneinfo import ZoneInfo
import yfinance as yf
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("OptionsGapBot")

# ── Market Hours Gatekeeper ────────────────────────────────────────────
def is_options_market_open() -> bool:
    """Returns True only during US Options regular trading session: Mon-Fri 9:30-16:00 ET."""
    now = datetime.now(NY_TZ)
    if now.weekday() >= 5:
        return False
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now <= close_t

# ── Timezones ──────────────────────────────────────────────────────────
NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# ── Config ─────────────────────────────────────────────────────────────
CAPITAL = 200.0
TRANCHES = 4                               # total capital divisions
TRANCH_SIZE = CAPITAL / TRANCHES           # $50 per position
MAX_POSITIONS = 2                          # launch cap: only top 2 setups

MIN_GAP = 5.0
MIN_PRE_VOL = 50_000
MIN_PRICE = 3.0
MAX_PRICE = 250.0
MIN_WIN_PROB = 0.35
RVOL_MIN = 0.5
PRE_MARKET_START_HOUR = 4

# Options-specific
OTM_PCT = 0.07                             # ~0.30 delta heuristic: strike = price × (1 ± 7%)
OPTION_DTE_MAX = 14                        # max days to expiration (next weekly)
OPTION_SL_PCT = 0.40                       # hard stop-loss at -40% premium decay
TRAIL_ACTIVATE_PCT = 0.50                  # trail activates at +50% gain
TRAIL_DISTANCE_PCT = 0.30                  # trail locks 30% below peak premium
STALE_TIMEOUT_MINUTES = 5                  # 5min: IV crush + theta decay salvage
ORDER_LIMIT_PRICE_BUMP = 1.05              # pay up to 5% above ask for fill priority

# Paths
TRADE_DB = "/tmp/options_trades.jsonl"
MODEL_DB = "/tmp/options_model.json"

# ── Watchlist (same as gap_bot) ────────────────────────────────────────
WATCHLIST = [
    "NVDA","AMD","TSLA","META","AMZN","GOOGL","MSFT","AAPL","NFLX",
    "COIN","MSTR","MARA","RIOT","PLTR","SOFI","HOOD","AFRM","UPST",
    "SMCI","ARM","CRWD","PANW","DASH","UBER","LYFT","SNAP","PINS",
    "ROKU","ZM","DOCU","MDB","SNOW","DDOG","NET",
    "AI","IONQ","RGTI","QBTS","BBAI","SOUN",
    "NIO","XPEV","LCID","RIVN","F","GM",
]


# ═══════════════════════════════════════════════════════════════════════
#  OPTIONS TRADE MODEL — Self-Learning
# ═══════════════════════════════════════════════════════════════════════

class OptionsModel:
    """Learns from past option trades to predict win probability
    and select optimal strike, SL, trail, and stale parameters."""

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
            "by_otm": defaultdict(lambda: {"wins": 0, "losses": 0}),
            "by_dte": defaultdict(lambda: {"wins": 0, "losses": 0}),
            "by_side": defaultdict(lambda: {"wins": 0, "losses": 0}),
            "by_weekday": defaultdict(lambda: {"wins": 0, "losses": 0}),
            "by_exit": defaultdict(lambda: {"wins": 0, "losses": 0}),
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
            logger.info("OptionsModel loaded: %d past trades", len(self.trades))
        else:
            self._seed()
            logger.info("OptionsModel seeded with %d simulated trades", len(self.trades))

    def _seed(self):
        random.seed(42)
        for _ in range(200):
            gap = round(random.uniform(5, 25), 1)
            otm = round(random.uniform(3, 20), 1)
            dte = random.randint(1, 14)
            side = random.choice(["call", "call", "put"])
            weekday = random.randint(0, 4)
            entry_p = round(random.uniform(0.20, 3.00), 2)
            peak_mult = random.uniform(1.0, 8.0)
            peak_p = round(entry_p * peak_mult, 2)
            ug_move = round(random.uniform(-8, 20), 1)
            win_prob = 0.25 + (gap / 50) * 0.2 + min(ug_move / 30, 0.3) + (1 - otm / 25) * 0.1
            win_prob = max(0.1, min(win_prob, 0.85))
            is_win = random.random() < win_prob
            exit_p = peak_p if is_win else round(entry_p * random.uniform(0.5, 0.95), 2)
            exit_reason = random.choices(
                ["trail", "stop_loss", "stale", "eod"],
                [0.35, 0.15, 0.25, 0.25] if is_win else [0.05, 0.50, 0.30, 0.15]
            )[0]
            gain_pct = round(((exit_p - entry_p) / entry_p) * 100, 1)
            self.trades.append({
                "sym": random.choice(WATCHLIST),
                "gap": gap, "otm_pct": otm, "dte": dte, "side": side,
                "weekday": weekday, "entry_premium": entry_p,
                "exit_premium": exit_p, "gain_pct": gain_pct,
                "pnl": round(gain_pct / 100 * entry_p * 100, 2),
                "exit_reason": exit_reason,
                "win": is_win,
                "time": datetime.now(UTC_TZ).isoformat(),
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
        self.stats["avg_win_pct"] = sum(t.get("gain_pct", 0) for t in wins) / len(wins) if wins else 0
        self.stats["avg_loss_pct"] = sum(t.get("gain_pct", 0) for t in losses) / len(losses) if losses else 0
        self.stats["total_pnl"] = sum(t.get("pnl", 0) for t in self.trades)
        self.stats["max_win"] = max((t.get("gain_pct", 0) for t in wins), default=0)
        self.stats["max_loss"] = min((t.get("gain_pct", 0) for t in losses), default=0)
        avg_win = self.stats["avg_win_pct"]
        avg_loss = abs(self.stats["avg_loss_pct"])
        wr = self.stats["win_rate"]
        self.stats["expectancy"] = wr * avg_win - (1 - wr) * avg_loss

        for t in self.trades:
            gap_b = str(int(t.get("gap", 0) / 5) * 5)
            otm_b = str(int(t.get("otm_pct", 10) / 5) * 5)
            dte_b = "1-2" if t.get("dte", 7) <= 2 else "3-7" if t.get("dte", 7) <= 7 else "8-14"
            side_b = t.get("side", "call")
            wd_b = str(t.get("weekday", 0))
            exit_b = t.get("exit_reason", "unknown")
            w = 1 if t.get("win") else 0
            l = 0 if t.get("win") else 1
            self.stats["by_gap"][gap_b]["wins"] += w
            self.stats["by_gap"][gap_b]["losses"] += l
            self.stats["by_otm"][otm_b]["wins"] += w
            self.stats["by_otm"][otm_b]["losses"] += l
            self.stats["by_dte"][dte_b]["wins"] += w
            self.stats["by_dte"][dte_b]["losses"] += l
            self.stats["by_side"][side_b]["wins"] += w
            self.stats["by_side"][side_b]["losses"] += l
            self.stats["by_weekday"][wd_b]["wins"] += w
            self.stats["by_weekday"][wd_b]["losses"] += l
            self.stats["by_exit"][exit_b]["wins"] += w
            self.stats["by_exit"][exit_b]["losses"] += l

        self._save_model()

    def _save_model(self):
        stats = json.loads(
            json.dumps(self.stats, default=lambda x: dict(x) if isinstance(x, defaultdict) else x)
        )
        stats["last_updated"] = datetime.now(UTC_TZ).isoformat()
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
            (self.stats["by_otm"], str(int(signal.get("otm_pct", 7) / 5) * 5)),
            (self.stats["by_side"], signal.get("side", "call")),
            (self.stats["by_weekday"], str(signal.get("weekday", datetime.now(NY_TZ).weekday()))),
        ]
        rates = []
        for bucket, key in pairs:
            r = wr_from(bucket, key)
            if r is not None:
                rates.append(r)

        if not rates:
            return self.stats["win_rate"] or 0.5

        avg = sum(rates) / len(rates)
        blended = avg * 0.7 + (self.stats["win_rate"] or 0.5) * 0.3
        return min(max(blended, 0.1), 0.9)

    def best_otm_pct(self) -> float:
        """Return the OTM% bucket with highest win rate for strike selection."""
        best_wr, best_otm = 0.0, OTM_PCT
        for otm_key, bucket in self.stats["by_otm"].items():
            total = bucket["wins"] + bucket["losses"]
            if total >= 5:
                wr = bucket["wins"] / total
                if wr > best_wr:
                    best_wr = wr
                    best_otm = float(otm_key)
        if best_otm == 0:
            best_otm = 5.0
        return best_otm / 100.0  # return as decimal (e.g., 0.07)

    def score_signal(self, signal: dict) -> float:
        """Score a signal combining win probability, gap size, and volatility."""
        prob = self.predict_win_prob(signal)
        gap_bonus = min(signal.get("gap", 0) / 30.0, 1.0) * 0.2
        vol_bonus = min(signal.get("rel_vol", 0) / 5.0, 1.0) * 0.1
        return round(prob + gap_bonus + vol_bonus, 3)

    def backtest_params(self) -> dict:
        """Find optimal SL, trail_activate, trail_distance, stale_timeout
        over real (non-simulated) trade history."""
        if len(self.trades) < 20:
            return {
                "sl": OPTION_SL_PCT, "trail_act": TRAIL_ACTIVATE_PCT,
                "trail_dist": TRAIL_DISTANCE_PCT, "stale": STALE_TIMEOUT_MINUTES,
            }

        real = [t for t in self.trades if not t.get("simulated")]
        if not real:
            return {
                "sl": OPTION_SL_PCT, "trail_act": TRAIL_ACTIVATE_PCT,
                "trail_dist": TRAIL_DISTANCE_PCT, "stale": STALE_TIMEOUT_MINUTES,
            }

        best_exp = -999.0
        best_params = {}
        for sl in [0.25, 0.35, 0.40, 0.50]:
            for trail_act in [0.30, 0.50, 0.75, 1.00]:
                for trail_dist in [0.20, 0.25, 0.30, 0.40]:
                    for stale in [3, 5, 7, 10]:
                        total_pnl = 0.0
                        for t in real:
                            first_entry = t.get("entry_premium", 1.0)
                            exit_p = t.get("exit_premium", first_entry)
                            peak_mult = t.get("peak_premium_seen", exit_p) / first_entry
                            gain = (exit_p - first_entry) / first_entry
                            # simulate exits
                            if gain >= trail_act:
                                locked = gain - (peak_mult - 1) * trail_dist
                                locked = min(gain, max(locked, 0))
                                total_pnl += locked
                            elif gain <= -sl:
                                total_pnl += -sl
                            elif gain > 0:
                                total_pnl += gain * 0.5
                            else:
                                total_pnl += gain
                        avg = total_pnl / len(real)
                        if avg > best_exp:
                            best_exp = avg
                            best_params = {
                                "sl": sl, "trail_act": trail_act,
                                "trail_dist": trail_dist, "stale": stale,
                            }

        if best_params:
            logger.info("OptionsModel backtest optimal: %s (avg $%.2f/trade)", best_params, best_exp)
        return best_params

    def report(self) -> str:
        s = self.stats
        return (
            f"OptionsModel: {s['total_trades']} trades | "
            f"WR={s['win_rate']:.0%} | "
            f"Expectancy={s['expectancy']:+.1f}% | "
            f"AvgWin={s['avg_win_pct']:+.1f}% "
            f"AvgLoss={s['avg_loss_pct']:.1f}%"
        )


# ═══════════════════════════════════════════════════════════════════════
#  OPTIONS GAP BOT — CLASS
# ═══════════════════════════════════════════════════════════════════════

class OptionsGapBot:
    """Self-contained options gap-trading engine.

    Public lifecycle:
      1. scan()           — find gap-ups via yfinance
      2. route_positions()— for each gap, query options chain, enter
      3. monitor_loop()   — continuous risk management (SL, trail, stale)
    """

    def __init__(self, api_key: str, secret_key: str, sim: bool = False):
        self.sim = sim
        self.api_key = api_key
        self.secret_key = secret_key
        self.tranche_pool = TRANCHES          # remaining available tranches

        if not sim:
            self._init_clients()
        else:
            self.trading_client = None
            self.data_client = None

        # Active position state machine
        self.active: Dict[str, dict] = {}     # contract_symbol -> state

        # Completed trade log
        self.trades: List[dict] = []
        self._load_trades()

        # Scanned signals from most recent scan
        self.signals: List[dict] = []

        # Learning model
        self.model = OptionsModel()

        # Track underlying prices at entry for later P&L analysis
        self._entry_prices: Dict[str, float] = {}

        # Risk param overrides (set nightly by model backtest)
        self.param_sl = OPTION_SL_PCT
        self.param_trail_act = TRAIL_ACTIVATE_PCT
        self.param_trail_dist = TRAIL_DISTANCE_PCT
        self.param_stale = STALE_TIMEOUT_MINUTES

    # ── Client Initialization ──────────────────────────────────────────

    def _init_clients(self):
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import OptionHistoricalDataClient
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        self.data_client = OptionHistoricalDataClient(self.api_key, self.secret_key)
        acct = self.trading_client.get_account()
        logger.info("Alpaca $%.2f equity | Options level: %s",
                    float(acct.equity),
                    getattr(acct, "options_approved_level", "unknown"))

    # ── Trade Persistence ──────────────────────────────────────────────

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
        logger.info("Loaded %d past option trades", len(self.trades))

    def _save_trade(self, trade: dict):
        self.trades.append(trade)
        with open(TRADE_DB, "a") as f:
            f.write(json.dumps(trade) + "\n")

    # ── Scanner (yfinance) ─────────────────────────────────────────────

    def _get_avg_vol(self, sym: str) -> float:
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(period="2mo")
            if hist.empty:
                return 0
            return float(hist["Volume"].tail(30).mean())
        except Exception:
            return 0

    async def scan(self) -> List[dict]:
        """Scan WATCHLIST for gap-ups. Returns scored signals list."""
        results = []
        logger.info("Scanning %d stocks for gap-ups...", len(WATCHLIST))

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

                if pre_market:
                    cur_time = now_ny.time()
                    today_slice = today_bars[today_bars.index.time <= cur_time]
                else:
                    today_slice = today_bars[
                        (today_bars.index.hour >= 9)
                        & (today_bars.index.minute >= 30)
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

                signal = {
                    "sym": sym,
                    "gap": round(gap, 1),
                    "vol": bar_vol,
                    "rel_vol": round(rel_vol, 1),
                    "price": round(price, 2),
                    "avg_vol": int(avg_vol),
                    "otm_pct": round(self.model.best_otm_pct() * 100, 1),
                    "weekday": now_ny.weekday(),
                    "score": 0.0,
                }
                signal["score"] = self.model.score_signal(signal)
                results.append(signal)
                await asyncio.sleep(0.05)
            except Exception as e:
                logger.debug("Scan skip %s: %s", sym, e)

        results.sort(key=lambda x: x["score"], reverse=True)
        logger.info("Scan: %d gap-ups found", len(results))
        self.signals = results
        return results

    # ═══════════════════════════════════════════════════════════════════
    #  TASK 1 — OPTIONS CHAIN QUERY
    # ═══════════════════════════════════════════════════════════════════

    def get_optimal_contract(self, symbol: str, underlying_price: float,
                             side: str = "call") -> Optional[dict]:
        """Query Alpaca option chain for the optimal 0.30-delta-like contract.

        Heuristic: selects the first OTM strike ~7% from the underlying price,
        expiring within OPTION_DTE_MAX calendar days. This approximates
        0.30 delta without requiring live Greeks.

        Returns dict with keys:
          contract_symbol, strike, expiration, premium_ask, premium_mid, qty_max
        or None if no viable contract found.
        """
        if self.sim:
            return self._sim_contract(symbol, underlying_price, side)

        from alpaca.trading.requests import GetOptionContractsRequest
        from alpaca.trading.enums import AssetStatus

        now_ny = datetime.now(NY_TZ)
        today_d = now_ny.date()
        max_expiry = today_d + timedelta(days=OPTION_DTE_MAX)

        # Determine target strike range — use learned OTM% from model
        learned_otm = self.model.best_otm_pct()
        otm_lo = learned_otm * 0.4
        otm_hi = learned_otm * 2.0
        if side == "call":
            target_strike_min = round(underlying_price * (1 + otm_lo), 2)
            target_strike_max = round(underlying_price * (1 + otm_hi), 2)
            contract_type_filter = "call"
        else:
            target_strike_min = round(underlying_price * (1 - otm_hi), 2)
            target_strike_max = round(underlying_price * (1 - otm_lo), 2)
            contract_type_filter = "put"

        try:
            req = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                status=AssetStatus.ACTIVE,
                expiration_date_gte=today_d + timedelta(days=1),  # skip 0DTE
                expiration_date_lte=max_expiry,
            )
            resp = self.trading_client.get_option_contracts(req)
            contracts = resp.option_contracts
        except Exception as e:
            logger.warning("Chain query failed for %s: %s", symbol, e)
            return None

        if not contracts:
            logger.debug("No contracts found for %s", symbol)
            return None

        # Filter: matching type, tradable, within strike range
        candidates = [
            c for c in contracts
            if c.type == contract_type_filter
            and c.tradable
            and target_strike_min <= float(c.strike_price) <= target_strike_max
        ]
        if not candidates:
            logger.debug("No OTM candidates for %s %s (strike range %.2f-%.2f)",
                         symbol, side, target_strike_min, target_strike_max)
            return None

        # Score by ATM proximity × affordability.
        # We want the closest to ATM (highest gamma) that fits within one tranche.
        scored = []
        for c in candidates:
            ask_price = self._get_option_ask(c.symbol)
            if ask_price is None:
                try:
                    ask_price = float(c.close_price)
                except (ValueError, TypeError):
                    continue
            if ask_price <= 0:
                continue
            if ask_price < 0.05:
                continue
            contract_cost = ask_price * 100
            max_qty = int(TRANCH_SIZE / contract_cost) if contract_cost > 0 else 0
            if max_qty < 1:
                continue
            # Score: lower strike (closer to ATM) = higher delta = better gamma
            strike_dist = abs(float(c.strike_price) - underlying_price)
            scored.append((strike_dist, c, ask_price, max_qty))

        if not scored:
            logger.debug("No affordable OTM candidates for %s %s (max cost $%.2f)",
                         symbol, side, TRANCH_SIZE)
            return None

        # Best = lowest strike distance (closest to ATM) that's affordable
        scored.sort(key=lambda x: x[0])
        _, best, ask_price, max_qty = scored[0]

        contract_cost = ask_price * 100

        logger.info("OPTION %s: strike=$%.2f exp=%s ask=$%.2f cost=$%.2f max_qty=%d",
                    best.symbol, float(best.strike_price), best.expiration_date,
                    ask_price, contract_cost, max_qty)

        return {
            "contract_symbol": best.symbol,
            "strike": float(best.strike_price),
            "expiration": str(best.expiration_date),
            "premium_ask": ask_price,
            "premium_mid": self._get_option_mid(best.symbol) or ask_price,
            "qty_max": max_qty,
        }

    def _sim_contract(self, symbol: str, underlying_price: float,
                      side: str = "call") -> dict:
        """Generate a simulated option contract for dry-run testing."""
        strike_mult = 1.07 if side == "call" else 0.93
        strike = round(underlying_price * strike_mult, 2)
        est_premium = round(underlying_price * 0.02, 2)   # ~2% of underlying
        contract_cost = est_premium * 100
        max_qty = max(1, int(TRANCH_SIZE / contract_cost)) if contract_cost > 0 else 1
        today_d = date.today()
        exp = today_d + timedelta(days=OPTION_DTE_MAX)
        logger.info("SIM OPTION %s: strike=$%.2f exp=%s premium=$%.2f qty=%d",
                    symbol, strike, exp.isoformat(), est_premium, max_qty)
        return {
            "contract_symbol": f"{symbol}{exp.strftime('%y%m%d')}{'C' if side=='call' else 'P'}{int(strike*1000):08d}",
            "strike": strike,
            "expiration": exp.isoformat(),
            "premium_ask": est_premium,
            "premium_mid": est_premium,
            "qty_max": max_qty,
        }

    def _get_option_ask(self, contract_symbol: str) -> Optional[float]:
        """Fetch current ask price for an option contract."""
        try:
            from alpaca.data.requests import OptionLatestQuoteRequest
            req = OptionLatestQuoteRequest(symbol_or_symbols=[contract_symbol])
            quotes = self.data_client.get_option_latest_quote(req)
            if contract_symbol in quotes:
                return float(quotes[contract_symbol].ask_price)
            return None
        except Exception as e:
            logger.debug("Quote fetch failed for %s: %s", contract_symbol, e)
            return None

    def _get_option_mid(self, contract_symbol: str) -> Optional[float]:
        """Fetch mid price (bid+ask)/2 for an option contract."""
        try:
            from alpaca.data.requests import OptionLatestQuoteRequest
            req = OptionLatestQuoteRequest(symbol_or_symbols=[contract_symbol])
            quotes = self.data_client.get_option_latest_quote(req)
            if contract_symbol in quotes:
                q = quotes[contract_symbol]
                if q.bid_price and q.ask_price:
                    return (float(q.bid_price) + float(q.ask_price)) / 2
            return None
        except Exception:
            return None

    # ═══════════════════════════════════════════════════════════════════
    #  TASK 2 — CAPITAL ALLOCATION & ORDER ENTRY
    # ═══════════════════════════════════════════════════════════════════

    async def execute_options_entry(self, symbol: str, side: str = "call",
                                    underlying_price: Optional[float] = None,
                                    entry_gap: float = 0.0) -> bool:
        """Allocate one tranche to an option position.

        Flow:
          1. Query optimal contract.
          2. Determine qty from TRANCH_SIZE / (premium × 100).
          3. Submit limit order.
          4. On fill, populate self.active state entry.

        Returns True if order was submitted (may not be filled yet).
        """
        if self.tranche_pool <= 0:
            logger.debug("No tranches available for %s", symbol)
            return False
        if len(self.active) >= MAX_POSITIONS:
            logger.debug("At max positions (%d) — skipping %s", MAX_POSITIONS, symbol)
            return False

        if underlying_price is None:
            underlying_price = self._get_underlying_price(symbol)
            if underlying_price is None:
                logger.warning("Cannot determine price for %s", symbol)
                return False

        contract = self.get_optimal_contract(symbol, underlying_price, side)
        if contract is None:
            return False

        qty = min(contract["qty_max"], self.tranche_pool)
        if qty < 1:
            logger.debug("Qty < 1 for %s %s", symbol, contract["contract_symbol"])
            return False

        entry_premium = contract["premium_ask"]
        limit_price = round(entry_premium * ORDER_LIMIT_PRICE_BUMP, 2)
        estimated_cost = round(entry_premium * 100 * qty, 2)
        sl_price = round(entry_premium * (1 - self.param_sl), 2)

        logger.info("\n" + "=" * 55)
        logger.info("OPTION ENTRY %s %s", side.upper(), symbol)
        logger.info("   Contract: %s strike=$%.2f exp=%s",
                    contract["contract_symbol"], contract["strike"], contract["expiration"])
        logger.info("   %d contract(s) × $%.2f ask × 100 = $%.2f",
                    qty, entry_premium, estimated_cost)
        logger.info("   SL: -%.0f%% ($%.2f) | Trail: +%.0f%% -> trail %.0f%%",
                    OPTION_SL_PCT * 100, sl_price,
                    TRAIL_ACTIVATE_PCT * 100, TRAIL_DISTANCE_PCT * 100)
        logger.info("   Kill switch: %d min", STALE_TIMEOUT_MINUTES)
        logger.info("=" * 55)

        if self.sim:
            logger.info("SIM — tracking only")
            self.active[contract["contract_symbol"]] = {
                "contract_symbol": contract["contract_symbol"],
                "underlying": symbol,
                "side": side,
                "qty": qty,
                "entry_premium": entry_premium,
                "highest_premium_seen": entry_premium,
                "stop_loss_price": sl_price,
                "trail_active": False,
                "trail_stop_price": None,
                "entry_time": time.time(),
                "filled": True,
                "entry_underlying_price": underlying_price,
                "entry_gap": entry_gap,
            }
            self.tranche_pool -= 1
            return True

        # Submit limit order
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        try:
            order = LimitOrderRequest(
                symbol=contract["contract_symbol"],
                qty=qty,
                side=OrderSide.BUY,
                limit_price=limit_price,
                time_in_force=TimeInForce.DAY,
            )
            resp = self.trading_client.submit_order(order)
            order_id = resp.id
            logger.info("Order submitted: %s (limit $%.2f) id=%s",
                        contract["contract_symbol"], limit_price, order_id)

            # Wait briefly for fill, then record state
            await asyncio.sleep(2)
            filled_premium = self._check_fill(contract["contract_symbol"])
            if filled_premium is None:
                logger.info("Order not yet filled — tracking unfilled order")
                self.active[contract["contract_symbol"]] = {
                    "contract_symbol": contract["contract_symbol"],
                    "underlying": symbol,
                    "side": side,
                    "qty": qty,
                    "entry_premium": entry_premium,
                    "highest_premium_seen": entry_premium,
                    "stop_loss_price": sl_price,
                    "trail_active": False,
                    "trail_stop_price": None,
                    "entry_time": time.time(),
                    "filled": False,
                    "order_id": order_id,
                    "entry_underlying_price": underlying_price,
                    "entry_gap": entry_gap,
                }
            else:
                self.active[contract["contract_symbol"]] = {
                    "contract_symbol": contract["contract_symbol"],
                    "underlying": symbol,
                    "side": side,
                    "qty": qty,
                    "entry_premium": filled_premium,
                    "highest_premium_seen": filled_premium,
                    "stop_loss_price": round(filled_premium * (1 - self.param_sl), 2),
                    "trail_active": False,
                    "trail_stop_price": None,
                    "entry_time": time.time(),
                    "filled": True,
                    "entry_underlying_price": underlying_price,
                    "entry_gap": entry_gap,
                }
                logger.info("Position opened: %d %s @ $%.2f",
                            qty, contract["contract_symbol"], filled_premium)

            self.tranche_pool -= 1
            return True

        except Exception as e:
            logger.error("Order failed for %s: %s", contract["contract_symbol"], e)
            return False

    def _get_underlying_price(self, symbol: str) -> Optional[float]:
        """Get current stock price via yfinance."""
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period="1d", interval="1m")
            if not hist.empty:
                return float(hist.iloc[-1]["Close"])
            tk2 = yf.Ticker(symbol)
            info = tk2.info
            return float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        except Exception:
            return None

    def _check_fill(self, contract_symbol: str) -> Optional[float]:
        """Check if an open order filled by looking at positions."""
        try:
            pos = self.trading_client.get_position(contract_symbol)
            if pos and float(pos.qty) > 0:
                return float(pos.avg_entry_price)
            return None
        except Exception:
            return None

    # ═══════════════════════════════════════════════════════════════════
    #  TASK 3 — STATE-MACHINE MONITORING & EXIT ROUTING
    # ═══════════════════════════════════════════════════════════════════

    async def monitor_loop(self, interval: float = 2.0):
        """Background loop: evaluate all active positions every `interval` seconds.

        For each tracked position:
          1. Fetch latest premium (mid price).
          2. Check hard stop-loss (-40% from entry).
          3. Update trailing stop if new high reached.
          4. Check trail trigger/breach.
          5. Check stale kill switch (15 min).
        """
        logger.info("Monitor loop started (interval=%.0fs)", interval)
        while True:
            if not is_options_market_open() and not self.active:
                await asyncio.sleep(60)
                continue
            for contract_symbol, state in list(self.active.items()):
                try:
                    self._evaluate_position(contract_symbol, state)
                except Exception as e:
                    logger.error("Monitor eval error %s: %s", contract_symbol, e)
            await asyncio.sleep(interval)

    def _evaluate_position(self, contract_symbol: str, state: dict):
        """Evaluate one tracked position against risk rules."""
        if state.get("closed"):
            return

        # Fetch current premium
        current_premium = self._get_current_premium(contract_symbol, state)

        if current_premium is None:
            # Check if position was filled (for unfilled orders)
            if not state.get("filled"):
                filled = self._check_fill(contract_symbol)
                if filled is not None:
                    state["filled"] = True
                    state["entry_premium"] = filled
                    state["highest_premium_seen"] = filled
                    state["stop_loss_price"] = round(filled * (1 - self.param_sl), 2)
                    logger.info("Order filled: %s @ $%.2f", contract_symbol, filled)
                return
            return

        entry = state["entry_premium"]
        gain = (current_premium - entry) / entry

        # ── Hard Stop-Loss ───────────────────────────────────────────
        if current_premium <= state["stop_loss_price"]:
            logger.warning("SL HIT %s: $%.2f (-%.0f%% of $%.2f entry)",
                           contract_symbol, current_premium,
                           abs(gain) * 100, entry)
            self._close_position(contract_symbol, current_premium, "stop_loss")
            return

        # ── Update peak ──────────────────────────────────────────────
        if current_premium > state["highest_premium_seen"]:
            state["highest_premium_seen"] = current_premium

            # Activate trail at threshold
            if gain >= self.param_trail_act and not state["trail_active"]:
                state["trail_active"] = True
                state["trail_stop_price"] = round(
                    current_premium * (1 - self.param_trail_dist), 2
                )
                logger.info("TRAIL ACTIVE %s: +%.0f%% high=$%.4f trail_stop=$%.4f",
                            contract_symbol, gain * 100,
                            current_premium, state["trail_stop_price"])

            # Update trail stop on new highs
            if state["trail_active"]:
                new_trail = round(current_premium * (1 - self.param_trail_dist), 2)
                if new_trail > state["trail_stop_price"]:
                    state["trail_stop_price"] = new_trail
                    logger.info("TRAIL LOCKED %s: new high=$%.4f stop=$%.4f",
                                contract_symbol, current_premium, new_trail)

        # ── Trail breach ─────────────────────────────────────────────
        if state["trail_active"] and state["trail_stop_price"] is not None:
            if current_premium <= state["trail_stop_price"]:
                logger.info("TRAIL HIT %s: $%.2f (%.0f%% of $%.2f peak)",
                            contract_symbol, current_premium,
                            (current_premium / state["highest_premium_seen"]) * 100,
                            state["highest_premium_seen"])
                self._close_position(contract_symbol, current_premium, "trail")
                return

        # ── Stale kill switch ────────────────────────────────────────
        elapsed = time.time() - state["entry_time"]
        if elapsed >= self.param_stale * 60:
            logger.info("STALE %s: %.0fmin -> recycling $%.0f tranche",
                        contract_symbol, elapsed / 60, TRANCH_SIZE)
            self._close_position(contract_symbol, current_premium, "stale")
            return

    def _get_current_premium(self, contract_symbol: str, state: dict) -> Optional[float]:
        """Fetch latest tradeable premium for an option position.

        Returns mid price if available, else position's current price.
        """
        # 1. Try Alpaca latest quote
        try:
            from alpaca.data.requests import OptionLatestQuoteRequest
            req = OptionLatestQuoteRequest(symbol_or_symbols=[contract_symbol])
            quotes = self.data_client.get_option_latest_quote(req)
            if contract_symbol in quotes:
                q = quotes[contract_symbol]
                bid = float(q.bid_price) if q.bid_price else None
                ask = float(q.ask_price) if q.ask_price else None
                if bid and ask:
                    return (bid + ask) / 2
                return bid or ask
        except Exception:
            pass

        # 2. Fall back to position current_price
        try:
            pos = self.trading_client.get_position(contract_symbol)
            if pos and float(pos.qty) > 0:
                return float(pos.current_price)
        except Exception:
            pass

        return None

    def _close_position(self, contract_symbol: str, exit_premium: float,
                        reason: str):
        """Market-sell an option position and recycle the tranche."""
        state = self.active.get(contract_symbol)
        if state is None or state.get("closed"):
            return

        state["closed"] = True

        if self.sim:
            self._log_and_recycle(contract_symbol, exit_premium, reason, state)
            return

        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce

            qty = state["qty"]
            order = MarketOrderRequest(
                symbol=contract_symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            resp = self.trading_client.submit_order(order)
            logger.info("Close order submitted: %s %d @ $%.2f (id=%s)",
                        contract_symbol, qty, exit_premium, resp.id)

            self._log_and_recycle(contract_symbol, exit_premium, reason, state)

        except Exception as e:
            logger.error("Close failed for %s: %s", contract_symbol, e)

    def _log_and_recycle(self, contract_symbol: str, exit_premium: float,
                         reason: str, state: dict):
        """Log completed trade and restore tranche to pool."""
        entry = state["entry_premium"]
        gain_pct = round(((exit_premium - entry) / entry) * 100, 2)
        pnl = round((exit_premium - entry) * 100 * state["qty"], 2)

        # Compute underlying move if we recorded entry price
        underlying_move = 0.0
        entry_ug = state.get("entry_underlying_price")
        if entry_ug and entry_ug > 0:
            cur_ug = self._get_underlying_price(state["underlying"])
            if cur_ug:
                underlying_move = round((cur_ug - entry_ug) / entry_ug * 100, 1)

        # Compute OTM% at entry
        otm_pct = 0.0
        contract_sym = state.get("contract_symbol", contract_symbol)
        # Parse strike from contract symbol — last part before expiry
        try:
            # OCC format: SYMYYMMDDC/PSTRIKE (e.g. SMCI250619C00034000)
            # strike is last 8 digits in micro format (strike * 1000)
            strike_str = contract_sym[-8:]
            strike_val = int(strike_str) / 1000.0
            if entry_ug and entry_ug > 0:
                otm_pct = round(abs(strike_val - entry_ug) / entry_ug * 100, 1)
        except (ValueError, IndexError):
            pass

        dte = 1
        try:
            date_part = contract_sym.split("_")[0] if "_" in contract_sym else contract_sym
            date_str = date_part[-9:-5]  # YYMM
            # extract YYMMDD
            year_str = "20" + contract_sym[-15:-13]
            month_str = contract_sym[-13:-11]
            day_str = contract_sym[-11:-9]
            exp_date = date(int(year_str), int(month_str), int(day_str))
            dte = max(1, (exp_date - date.today()).days)
        except (ValueError, IndexError):
            pass

        trade = {
            "contract_symbol": contract_symbol,
            "underlying": state["underlying"],
            "side": state["side"],
            "qty": state["qty"],
            "entry_premium": entry,
            "exit_premium": exit_premium,
            "gap": state.get("entry_gap", 0),
            "underlying_move_pct": underlying_move,
            "dte": dte,
            "otm_pct": otm_pct,
            "peak_premium_seen": state["highest_premium_seen"],
            "gain_pct": gain_pct,
            "pnl": pnl,
            "win": gain_pct > 0,
            "exit_reason": reason,
            "time": datetime.now(UTC_TZ).isoformat(),
        }
        self._save_trade(trade)
        self.model.add_trade(trade)

        logger.info("EXIT %s: %.1f%% ($%.2f) reason=%s",
                    contract_symbol, gain_pct, pnl, reason)

        # Remove from active tracking
        if contract_symbol in self.active:
            del self.active[contract_symbol]
        self.tranche_pool += 1


# ═══════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════

async def main_loop(bot: OptionsGapBot):
    """Orchestrate the daily trading cycle."""
    logger.info("\n" + "=" * 55)
    logger.info("OPTIONS GAP BOT — $%.0f capital | %d × $%.0f tranches",
                CAPITAL, TRANCHES, TRANCH_SIZE)
    logger.info("   OTM ~%.0f%% | SL: -%.0f%% | Trail: +%.0f%% -> %.0f%%",
                OTM_PCT * 100, OPTION_SL_PCT * 100,
                TRAIL_ACTIVATE_PCT * 100, TRAIL_DISTANCE_PCT * 100)
    logger.info("   Max positions: %d | Kill switch: %d min",
                MAX_POSITIONS, STALE_TIMEOUT_MINUTES)
    logger.info("=" * 55 + "\n")

    # Start risk monitor as background task
    asyncio.create_task(bot.monitor_loop(interval=2.0))

    today_state = set()

    while True:
        if not is_options_market_open():
            now_ny = datetime.now(NY_TZ)
            next_open = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
            if now_ny >= now_ny.replace(hour=16, minute=0, second=0, microsecond=0):
                next_open += timedelta(days=1)
            if next_open.weekday() >= 5:
                next_open += timedelta(days=(7 - next_open.weekday()))
            sleep_secs = (next_open - now_ny).total_seconds()
            logger.info("Markets closed. Sleeping %.0f min until %s",
                        sleep_secs / 60, next_open.strftime("%a %H:%M ET"))
            await asyncio.sleep(min(sleep_secs, 3600))
            continue

        now_ny = datetime.now(NY_TZ)
        h, m = now_ny.hour, now_ny.minute

        # ── 9:00 AM — Scan ──────────────────────────────────────────
        if h == 9 and m == 0 and "scan" not in today_state:
            today_state.add("scan")
            signals = await bot.scan()
            if signals:
                logger.info("Top signals:")
                for s in signals[:5]:
                    logger.info("  %s: gap=+%.1f%% vol=%d rel=%.1f",
                                s["sym"], s["gap"], s["vol"], s["rel_vol"])
                today_state.add("has_signals")

        # ── 9:01-9:15 — Route positions (stagger entry) ────────────
        if h == 9 and 1 <= m <= 15 and "routed" not in today_state and "has_signals" in today_state:
            if bot.tranche_pool > 0 and bot.signals:
                signal = bot.signals.pop(0)
                ok = await bot.execute_options_entry(
                    signal["sym"], side="call",
                    underlying_price=signal["price"],
                    entry_gap=signal.get("gap", 0.0)
                )
                if ok:
                    logger.info("Tranche allocated: %s (%d remaining)",
                                signal["sym"], bot.tranche_pool)
                else:
                    logger.debug("Skipped %s (no viable option contract)", signal["sym"])
                await asyncio.sleep(30)  # stagger to avoid price impact
            else:
                today_state.add("routed")
                logger.info("All tranches allocated or no signals remaining")

        # ── 9:31 AM — Entry window closes ──────────────────────────
        if h == 9 and m >= 31 and "routed" not in today_state:
            today_state.add("routed")
            logger.info("Entry window closed. %d active positions, %d tranches remaining",
                        len(bot.active), bot.tranche_pool)

        # ── 4:00 PM — Force-close any remaining positions ─────────
        if h >= 16 and "closed_day" not in today_state:
            today_state.add("closed_day")
            logger.info("Market close. Closing %d remaining positions...", len(bot.active))
            for contract_symbol, state in list(bot.active.items()):
                premium = bot._get_current_premium(contract_symbol, state)
                if premium is None:
                    premium = state.get("highest_premium_seen", state["entry_premium"])
                bot._close_position(contract_symbol, premium, "eod")
            logger.info("Day complete. %d total trades logged.", len(bot.trades))
            # Report performance
            if bot.trades:
                today_trades = [t for t in bot.trades[-20:]
                                if t.get("time", "").startswith(datetime.now(UTC_TZ).strftime("%Y-%m-%d"))]
                wins = sum(1 for t in today_trades if t.get("gain_pct", 0) > 0)
                total_pnl = sum(t.get("pnl", 0) for t in today_trades)
                logger.info("Today: %d trades | %d wins | P&L $%.2f",
                            len(today_trades), wins, total_pnl)
            logger.info("Model: %s", bot.model.report())
            # Overnight param backtest — tune SL/trail/stale
            best = bot.model.backtest_params()
            bot.param_sl = best["sl"]
            bot.param_trail_act = best["trail_act"]
            bot.param_trail_dist = best["trail_dist"]
            bot.param_stale = best["stale"]
            logger.info("Overnight tune: SL=%.0f%% trail_act=+%.0f%% trail_dist=%.0f%% stale=%dmin",
                        best["sl"] * 100, best["trail_act"] * 100,
                        best["trail_dist"] * 100, best["stale"])

        await asyncio.sleep(10)


# ═══════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

async def main():
    sim = "--sim" in sys.argv
    oneshot = "--oneshot" in sys.argv

    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")

    if not key or not secret:
        if not sim:
            logger.warning("No Alpaca keys. Run --sim or export APCA_API_KEY_ID")
            sim = True

    bot = OptionsGapBot(key or "", secret or "", sim=sim)

    if oneshot:
        if "--scan" in sys.argv:
            await bot.scan()
            for s in bot.signals:
                print(f"  {s['sym']:6s} gap=+{s['gap']:5.1f}% vol={s['vol']:>8d} rel={s['rel_vol']:.1f}x")
        elif "--chain" in sys.argv:
            sym = sys.argv[sys.argv.index("--chain") + 1]
            price = bot._get_underlying_price(sym) or 0
            print(f"{sym}: underlying=$~{price:.2f}")
            c = bot.get_optimal_contract(sym, price, "call")
            if c:
                print(f"  CALL: {c['contract_symbol']} strike=${c['strike']:.2f} "
                      f"ask=${c['premium_ask']:.2f} qty_max={c['qty_max']}")
            p = bot.get_optimal_contract(sym, price, "put")
            if p:
                print(f"  PUT:  {p['contract_symbol']} strike=${p['strike']:.2f} "
                      f"ask=${p['premium_ask']:.2f} qty_max={p['qty_max']}")
        return

    await main_loop(bot)


if __name__ == "__main__":
    asyncio.run(main())
