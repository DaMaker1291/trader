"""
ULTRA BOT v2 — gap bot (proven) + momentum scan (mid-day) + power hour scalp.
Clean integration: runs proven run_v5 for gap, adds scan on top.
"""

import sys, os, random, math
sys.path.insert(0, os.path.dirname(__file__) or '.')
from backtest_stress import *

random.seed(42)
CAPITAL = 200.0

# ── Momentum event generator ────────────────────────────────────────
def gen_midday(sc: str, open_p: float, n_bars: int = 120) -> List[float]:
    """Generate mid-day bars with random momentum surges."""
    r = random.Random()
    bars = []
    cur = open_p * (1 + r.uniform(-0.03, 0.05))  # drift from open
    for i in range(n_bars):
        # 2% chance per bar of a momentum surge (lasts 3-8 bars)
        if r.random() < 0.02:
            strength = r.uniform(0.01, 0.04)
            duration = r.randint(3, 8)
            direction = 1 if r.random() > 0.4 else -1
            for j in range(duration):
                cur += cur * (direction * strength / duration + r.gauss(0, 0.001))
                bars.append(cur)
        else:
            cur += cur * r.gauss(0, 0.002)
            bars.append(cur)
    return bars

def scan_trades(bars: List[float], capital: float, sl: float = 1.5,
                ta: float = 0.3, td: float = 0.3, stale: int = 15) -> float:
    """Scan bars for momentum surges and trade them. Returns total PnL."""
    pos_size = capital / 2
    total = 0.0
    positions = []

    for idx, price in enumerate(bars):
        if idx < 5: continue
        # Check exits first
        for pos in list(positions):
            pos["bars"] += 1
            gain = (price - pos["entry"]) / pos["entry"] * 100 if pos["dir"] == 1 else \
                   (pos["entry"] - price) / pos["entry"] * 100
            if gain <= -sl or pos["bars"] >= stale:
                total += gain / 100 * pos["size"]
                positions.remove(pos)
                continue
            if gain >= ta and not pos["trail"]:
                pos["trail"] = True
                pos["ts"] = price * (1 - td/100) if pos["dir"] == 1 else price * (1 + td/100)
            if pos["trail"]:
                if pos["dir"] == 1:
                    ns = price * (1 - td/100)
                    if ns > pos["ts"]: pos["ts"] = ns
                    if price <= pos["ts"]:
                        total += gain / 100 * pos["size"]
                        positions.remove(pos)
                else:
                    ns = price * (1 + td/100)
                    if ns < pos["ts"]: pos["ts"] = ns
                    if price >= pos["ts"]:
                        total += gain / 100 * pos["size"]
                        positions.remove(pos)

        if len(positions) >= 2: continue
        # Detect momentum surge: 0.5%+ move in 2 bars
        if idx < 2: continue
        move = (price - bars[idx-2]) / bars[idx-2] * 100
        if abs(move) < 0.5: continue
        direction = 1 if move > 0 else -1
        positions.append({"entry": price, "trail": False, "ts": 0,
                          "bars": 0, "dir": direction, "size": pos_size})
    return total


# ── RUN ─────────────────────────────────────────────────────────────

vxn = generate_vxn_timeseries(2000)
days_cache = {}
for m in ["NORMAL","HARSH","EXTREME","APOC"]:
    days_cache[m] = [generate_day(vxn[i], stress_level=m) for i in range(2000)]

print("=" * 95)
print("  ULTRA BOT v2 — gap bot + momentum scan + power hour scalp")
print("=" * 95)

for scan_name, scan_params in [
    ("Momentum scan (1.5/0.3/0.3)", {"sl": 1.5, "ta": 0.3, "td": 0.3}),
    ("Momentum scan (2.0/0.5/0.5)", {"sl": 2.0, "ta": 0.5, "td": 0.5}),
    ("Momentum scan (1.0/0.3/0.3)", {"sl": 1.0, "ta": 0.3, "td": 0.3}),
]:
    print(f"\n  --- {scan_name} ---")
    print(f"  {'Mix':<12} {'Gap $/d':<9} {'Scan $/d':<10} {'PH $/d':<9} {'Total $/d':<10} {'Gap $/hr':<9} {'Total $/hr':<10}")
    print(f"  {'-'*70}")

    totals = {}
    for m in ["NORMAL","HARSH","EXTREME","APOC"]:
        days = days_cache[m]

        # Gap bot (proven)
        gap_trades = run_v5(days, dict(CONFIG_V5), circuit_breaker=0)
        gap_metrics = compute_metrics(gap_trades) if gap_trades else {"da": 0, "tp": 0}
        gap_daily = gap_metrics["da"]

        # Mid-day momentum scan
        scan_total = 0.0
        for day in days:
            bars = gen_midday(day["scenario"], day["open"], n_bars=120)
            cap = CAPITAL  # fresh capital each day
            scan_total += scan_trades(bars, cap, **scan_params)

        # Power hour scalp (uses same params)
        ph_total = 0.0
        for day in days:
            bars = gen_midday(day["scenario"], day["open"] * 1.02, n_bars=30)
            cap = CAPITAL
            ph_total += scan_trades(bars, cap, sl=2.0, ta=0.5, td=0.5)

        scan_daily = scan_total / 2000
        ph_daily = ph_total / 2000
        total_daily = gap_daily + scan_daily + ph_daily
        gap_hr = gap_daily / 2
        total_hr = total_daily / 6

        totals[m] = {"gap": gap_daily, "scan": scan_daily, "ph": ph_daily,
                      "total": total_daily, "gap_hr": gap_hr, "total_hr": total_hr}
        print(f"  {m:<12} ${gap_daily:<6.2f}  ${scan_daily:<7.2f} ${ph_daily:<6.2f}  ${total_daily:<7.2f}  ${gap_hr:<6.2f}  ${total_hr:<6.2f}")

    gap_avg = sum(totals[m]["gap"] for m in totals) / 4
    scan_avg = sum(totals[m]["scan"] for m in totals) / 4
    ph_avg = sum(totals[m]["ph"] for m in totals) / 4
    total_avg = gap_avg + scan_avg + ph_avg
    print(f"  {'-'*70}")
    print(f"  {'AVG':<12} ${gap_avg:<6.2f}  ${scan_avg:<7.2f} ${ph_avg:<6.2f}  ${total_avg:<7.2f}  ${gap_avg/2:<6.2f}  ${total_avg/6:<6.2f}")
    print(f"  Dilution: ${total_avg/6:.2f}/hr vs ${gap_avg/2:.2f}/hr gap ({(total_avg/6)/(gap_avg/2)-1:.0%})")


# ── Final projection with best params ───────────────────────────────
print("\n" + "=" * 95)
print("  BEST CASE: gap bot + momentum scan (1.5/0.3/0.3) + PH scalp")
print("=" * 95)

# Re-run with best params to get accurate daily
all_gap = all_scan = all_ph = 0.0
for m in ["NORMAL","HARSH","EXTREME","APOC"]:
    days = days_cache[m]
    gt = run_v5(days, dict(CONFIG_V5), circuit_breaker=0)
    gm = compute_metrics(gt) if gt else {"da":0}
    all_gap += gm["da"]
    st = 0.0
    for day in days:
        bars = gen_midday(day["scenario"], day["open"])
        st += scan_trades(bars, CAPITAL, sl=1.5, ta=0.3, td=0.3)
    all_scan += st / 2000
    pt = 0.0
    for day in days:
        bars = gen_midday(day["scenario"], day["open"] * 1.02, n_bars=30)
        pt += scan_trades(bars, CAPITAL, sl=2.0, ta=0.5, td=0.5)
    all_ph += pt / 2000

gap_avg = all_gap / 4
scan_avg = all_scan / 4
ph_avg = all_ph / 4
total_avg = gap_avg + scan_avg + ph_avg
gap_pct = gap_avg / CAPITAL * 100
total_pct = total_avg / CAPITAL * 100

print(f"\n  Gap bot:      ${gap_avg:.2f}/day ({gap_pct:.2f}%/day)")
print(f"  Mid-day scan: ${scan_avg:.2f}/day")
print(f"  Power hour:   ${ph_avg:.2f}/day")
print(f"  ALL HOURS:    ${total_avg:.2f}/day ({total_pct:.2f}%/day)")
print(f"  $/hr: gap ${gap_avg/2:.2f} → blended ${total_avg/6:.2f}")

print(f"\n  {'─'*60}")
print(f"  COMPOUND PROJECTIONS — starting $200, compounding daily")
print(f"  {'─'*60}")

for name, daily_pct in [("Gap bot only (2hr)", gap_pct), ("All-hours (6hr)", total_pct)]:
    print(f"\n  {name}:")
    print(f"  {'Period':<12} {'Balance':<18} {'Monthly $':<12}")
    print(f"  {'-'*42}")
    bal = CAPITAL
    for end, label in [(21,"1mo"), (63,"3mo"), (126,"6mo"), (252,"1yr")]:
        start_bal = bal
        for d in range(end):
            bal += bal * daily_pct / 100
        print(f"  {label:<12} ${bal:<14.0f}  ${bal - start_bal:<9.0f}")

print(f"\n  {'='*60}")
print(f"  REALITY CHECK: $200 × {total_pct:.2f}%/day × 252 days")
print(f"  {'='*60}")
total_return = CAPITAL * (1 + total_pct / 100) ** 252
print(f"  ${CAPITAL:.0f} →  ${total_return:,.0f}")
if total_return > 1_000_000:
    print(f"  That's ${total_return/1_000_000:.1f} MILLION dollars from $200.")
print(f"  Daily target: ${total_avg:.2f}")
print(f"  Needed trades/day: gap ~0.5 + scan ~2 + PH ~1 = ~3.5 trades total")
