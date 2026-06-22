"""Crypto bot — trades only during active momentum, sits in cash during quiet hours."""
import asyncio, json, time, os
from datetime import datetime, timezone
from dataclasses import dataclass
from collections import deque
from typing import Optional, Dict, List, Tuple
import aiohttp
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("Crypto")

SCAN_INTERVAL = 1
BET_PCT = 1.0
SL_PCT = 0.7
TRAIL_ACT = 1.5
TRAIL_DIST = 0.5
STALE_EXIT_SEC = 120
COOLDOWN_SEC = 15
# Volatility gate — only trade when best momentum >= this
MIN_MOMENTUM = 0.15  # % — lower to catch more tradeable momentum

STATUS_PATH = "/tmp/crypto_status.json"
START_TIME = time.time()

MIN_PRICE = 0.50
MIN_VOL = 500_000


@dataclass
class Pos:
    sym: str; direction: int; entry: float; qty: float; entry_t: float
    high: float; low: float; sl: float; trail_act: bool = False; trail_stop: float = 0.0


class Bot:
    def __init__(self):
        self.cash = 100.0
        self.equity = 100.0
        self.peak = 100.0
        self.pos: Optional[Pos] = None
        self.trades: List[Dict] = []
        self.trade_count = 0
        self._tickers: Dict[str, dict] = {}
        self._cache: Dict[str, Tuple[float, float]] = {}
        self._hist: Dict[str, deque] = {}
        self._ws_ready = False
        self._cooldowns: Dict[str, float] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_report = time.time()
        self._eq_hist: List = []

    async def _session_get(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False))
        return self._session

    async def _fetch_all(self) -> List[str]:
        for _ in range(3):
            try:
                s = await self._session_get()
                async with s.get("https://api.binance.com/api/v3/exchangeInfo", timeout=aiohttp.ClientTimeout(total=10)) as r:
                    d = await r.json()
                    return [x["symbol"] for x in d.get("symbols", []) if x["symbol"].endswith("USDT") and x["status"] == "TRADING"]
            except: await asyncio.sleep(2)
        return []

    async def _ws_loop(self):
        all_syms = await self._fetch_all() or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        streams = [f"{s.lower()}@ticker" for s in all_syms]
        sub = {"method": "SUBSCRIBE", "params": streams, "id": 1}
        while True:
            try:
                s = await self._session_get()
                async with s.ws_connect("wss://stream.binance.com:9443/ws", heartbeat=30) as ws:
                    await ws.send_json(sub)
                    for _ in range(10):
                        msg = await ws.receive()
                        if isinstance(json.loads(msg.data), dict) and "id" in json.loads(msg.data): break
                    self._ws_ready = True
                    logger.info("WS: %d streams", len(streams))
                    while True:
                        msg = await ws.receive()
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                d = json.loads(msg.data)
                                if not isinstance(d, dict): continue
                                sym = d.get("s", "")
                                if not sym.endswith("USDT"): continue
                                price = float(d.get("c", 0))
                                chg = float(d.get("P", 0))
                                vol = float(d.get("q", "0") or 0)
                                if price > 0:
                                    now = time.time()
                                    self._tickers[sym] = {"p": price, "chg": chg, "vol": vol}
                                    self._cache[sym] = (now, price)
                                    if sym not in self._hist:
                                        self._hist[sym] = deque(maxlen=600)
                                    self._hist[sym].append(price)
                            except: pass
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR): break
            except Exception as e:
                logger.warning("WS: %s", e)
                self._ws_ready = False
                await asyncio.sleep(3)

    def momentum(self, sym: str) -> Tuple[float, int]:
        h = self._hist.get(sym)
        if not h or len(h) < 5: return (0.0, 0)
        p = list(h); n = len(p); cur = p[-1]
        m5s = (cur / p[-5] - 1) * 100 if n>=5 else 0
        m15s = (cur / p[min(-1,-15)] - 1) * 100 if n>=15 else 0
        m30s = (cur / p[min(-1,-30)] - 1) * 100 if n>=30 else 0
        mc = m5s*0.60 + m15s*0.25 + m30s*0.15
        if abs(mc) < 0.03: return (0.0, 0)
        return (abs(mc), 1 if mc>0 else -1)

    def eligible(self) -> int:
        return sum(1 for s in self._tickers if self._tickers[s]["p"] > MIN_PRICE and self._tickers[s].get("vol",0) > MIN_VOL)

    def best_momentum(self) -> float:
        """Return the strongest momentum across all eligible pairs (0 if none)."""
        best = 0.0
        for sym in self._tickers:
            td = self._tickers[sym]
            if td["p"] <= MIN_PRICE or td.get("vol",0) < MIN_VOL: continue
            mom,_ = self.momentum(sym)
            if mom > best: best = mom
        return best

    def pick(self):
        """Only pick if market has real momentum."""
        now = time.time()
        best_sym = None; best_mom = -1; best_dir = 0; best_pr = 0

        # First check: is anything moving enough?
        market_mom = 0.0

        for sym in self._tickers:
            td = self._tickers[sym]
            if td["p"] <= MIN_PRICE or td.get("vol",0) < MIN_VOL: continue
            if self._cooldowns.get(sym,0) > now: continue
            mom, d = self.momentum(sym)
            if mom > 0:
                market_mom = max(market_mom, mom)
            if mom > best_mom:
                best_mom = mom; best_sym = sym; best_dir = d; best_pr = td["p"]

        # Volatility gate: skip if nothing meaningful is moving
        if best_mom < MIN_MOMENTUM:
            return None, market_mom

        return (best_sym, best_pr, best_dir, best_mom), market_mom

    def enter(self, sym: str, price: float, direction: int, mom: float):
        n = self.cash * BET_PCT
        if n < 1.0: return
        qty = n / price
        sl = price * (1 - SL_PCT/100) if direction==1 else price * (1 + SL_PCT/100)
        self.pos = Pos(sym, direction, price, qty, time.time(), price, price, sl)
        self.cash -= n
        self.trade_count += 1
        dl = "LONG" if direction==1 else "SHORT"
        logger.info("🔥 %s: $%.2f %s @ $%.4f mom=%.4f%% SL=$%.4f", dl, n, sym, price, mom, sl)

    def monitor(self) -> bool:
        if not self.pos: return False
        p = self._cache.get(self.pos.sym)
        if not p: return False
        cur = p[1]; entry = self.pos.entry
        if self.pos.direction == 1:
            g = ((cur - entry)/entry)*100
            self.pos.high = max(self.pos.high, cur)
            if cur <= self.pos.sl: self.close("sl", cur, g); return True
            if time.time()-self.pos.entry_t > STALE_EXIT_SEC: self.close("stale", cur, g); return True
            if self.pos.high >= entry * (1 + TRAIL_ACT/100):
                ts = self.pos.high * (1 - TRAIL_DIST/100); self.pos.trail_act = True; self.pos.trail_stop = ts
                if cur <= ts: self.close("trail", cur, g); return True
        else:
            g = ((entry - cur)/entry)*100
            self.pos.low = min(self.pos.low, cur)
            if cur >= self.pos.sl: self.close("sl", cur, g); return True
            if time.time()-self.pos.entry_t > STALE_EXIT_SEC: self.close("stale", cur, g); return True
            if self.pos.low <= entry * (1 - TRAIL_ACT/100):
                ts = self.pos.low * (1 + TRAIL_DIST/100); self.pos.trail_act = True; self.pos.trail_stop = ts
                if cur >= ts: self.close("trail", cur, g); return True
        return False

    def close(self, reason: str, exit_p: float, gain: float):
        pos = self.pos
        pv = pos.qty * (exit_p if pos.direction==1 else 2*pos.entry - exit_p)
        self.cash += max(0, pv); self.equity = self.cash
        self.peak = max(self.peak, self.equity)
        dl = "LONG" if pos.direction==1 else "SHORT"
        self.trades.append({"sym":pos.sym,"dir":dl,"entry":pos.entry,"exit":exit_p,"gain":gain,"reason":reason})
        logger.info("📋 %s %s %+.2f%% (%s) Cash=$%.2f", pos.sym, dl, gain, reason, self.cash)
        self._cooldowns[pos.sym] = time.time() + COOLDOWN_SEC
        self.pos = None

    def update_eq(self):
        if self.pos:
            p = self._cache.get(self.pos.sym)
            if p:
                v = self.pos.qty * (p[1] if self.pos.direction==1 else 2*self.pos.entry-p[1])
                self.equity = self.cash + max(0, v)
            else: self.equity = self.cash
        else: self.equity = self.cash

    def status(self) -> str:
        s = ""
        if self.pos:
            p = self._cache.get(self.pos.sym)
            if p:
                g = ((p[1]-self.pos.entry)/self.pos.entry)*100 if self.pos.direction==1 else ((self.pos.entry-p[1])/self.pos.entry)*100
                d = "L" if self.pos.direction==1 else "S"
                s = f" | {d}:{self.pos.sym} @ ${p[1]:.4f} ({g:+.2f}%)"
        return f"💰 ${self.equity:.2f} (peak ${self.peak:.2f}) | Trades={self.trade_count}{s}"

    def export(self):
        el = (time.time()-START_TIME)/3600; pnl = self.equity-100.0
        js = {"equity":round(self.equity,2),"peak":round(self.peak,2),"pnl":round(pnl,2),
              "trades":self.trade_count,"run_hrs":round(el,3),"$_per_hr":round(pnl/el,2) if el>0 else 0,
              "eligible":self.eligible(),"best_mom":round(self.best_momentum(),3)}
        try:
            with open(STATUS_PATH,"w") as f: json.dump(js,f)
        except: pass

    async def cleanup(self):
        if self._session and not self._session.closed: await self._session.close()


async def main():
    logger.info("="*60)
    logger.info("🔥 VOLATILITY-GATED: only trade when momentum > %.1f%%", MIN_MOMENTUM)
    logger.info("   SL:%.1f%% Trail:act@%.1f%% dist=%.1f%% Stale:%ds Bet:%.0f%%",
                SL_PCT, TRAIL_ACT, TRAIL_DIST, STALE_EXIT_SEC, BET_PCT*100)
    logger.info("   Price≥$%.2f Vol≥%.0fK 436 pairs L+S", MIN_PRICE, MIN_VOL/1000)
    logger.info("="*60)

    b = Bot()
    wt = asyncio.create_task(b._ws_loop())
    timeout = 15
    while timeout>0 and not b._ws_ready:
        await asyncio.sleep(0.5); timeout-=0.5
    await asyncio.sleep(10)

    try:
        while True:
            t0 = time.time()
            b.update_eq(); b.monitor()

            if b.pos is None:
                pk, market_mom = b.pick()
                if pk:
                    sym,pr,dir_,mom = pk
                    dl = "LONG" if dir_==1 else "SHORT"
                    logger.info("📡 %s %s @ $%.4f mom=%.4f%% market=%.3f%%", dl, sym, pr, mom, market_mom)
                    b.enter(sym,pr,dir_,mom)

            b._eq_hist.append((time.time(), b.equity))
            b.export()

            if time.time()-b._last_report>=60:
                el=(time.time()-START_TIME)/3600; pnl=b.equity-100.0
                r=pnl/el if el>0 else 0; bm=b.best_momentum()
                status = "TRADING" if b.pos else ("QUIET" if bm<MIN_MOMENTUM else "READY")
                logger.info("📊 %s | $%.2f / %.2f hrs = $%.2f/hr (%d trades, %d eligible, best_mom=%.3f%%)",
                            status, pnl, el, r, b.trade_count, b.eligible(), bm)
                b._last_report=time.time()

            logger.info(b.status())
            await asyncio.sleep(max(0,SCAN_INTERVAL-(time.time()-t0)))
    except KeyboardInterrupt: pass
    finally:
        wt.cancel()
        if b.pos:
            p=b._cache.get(b.pos.sym)
            b.close("exit",p[1] if p else b.pos.entry,0)
        el=(time.time()-START_TIME)/3600; pnl=b.equity-100.0
        logger.info(f"FINAL: ${b.equity:.2f} P&L:${pnl:+.2f} Rate:${pnl/el:.2f}/hr Trades:{b.trade_count}" if el>0 else "")
        await b.cleanup()

if __name__=="__main__":
    asyncio.run(main())
