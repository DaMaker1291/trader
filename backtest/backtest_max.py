import sys, random, math
sys.path.insert(0, '.')
from backtest_stress import *

random.seed(42)

BASE = {
    'tranches': 2, 'tranche_pct': 0.50, 'sl': 4.0,
    'trail_act': 3.0, 'trail_dist': 3.0,
    'stale': 120, 'stale_early': 60, 'stale_thresh': 1.0,
    'min_gap': 5.0, 'min_vol': 50000, 'min_price': 3.0,
    'fade_skip': 0.65, 'rvol_floor': 1.0, 'skip_open_bars': 2,
    'min_gap_hostile': 8.0, 'sl_hostile': 3.0, 'single_tranche_hostile': True,
    'vxn_threshold': 30, 'vxn_hostile': 25,
    'extended_hold_thresh': 5.0, 'extended_hold_trail': 8.0, 'extended_hold_max': 120,
    'two_sided': True, 'short_act': 3.0, 'short_dist': 3.0, 'short_sl': 3.0,
}

VARIANTS = []
V = VARIANTS.append

V(('baseline v5.5', {}))
V(('trail=2%', {'trail_dist': 2.0}))
V(('trail=4%', {'trail_dist': 4.0}))
V(('trail=5%', {'trail_dist': 5.0}))
V(('trail_act=2%', {'trail_act': 2.0}))
V(('trail_act=5%', {'trail_act': 5.0}))
V(('trail_act=8%', {'trail_act': 8.0}))
V(('trail_act=20%', {'trail_act': 20.0}))
V(('sl=2%', {'sl': 2.0, 'sl_hostile': 2.0, 'short_sl': 2.0}))
V(('sl=5%', {'sl': 5.0, 'sl_hostile': 5.0, 'short_sl': 5.0}))
V(('sl=6%', {'sl': 6.0, 'sl_hostile': 6.0, 'short_sl': 6.0}))
V(('min_gap=3%', {'min_gap': 3.0, 'min_gap_hostile': 5.0}))
V(('min_gap=8%', {'min_gap': 8.0, 'min_gap_hostile': 10.0}))
V(('min_gap=10%', {'min_gap': 10.0}))
V(('rvol=0.5', {'rvol_floor': 0.5}))
V(('rvol=1.5', {'rvol_floor': 1.5}))
V(('rvol=2.0', {'rvol_floor': 2.0}))
V(('no skip open', {'skip_open_bars': 0}))
V(('skip=3 bars', {'skip_open_bars': 3}))
V(('skip=5 bars', {'skip_open_bars': 5}))
V(('stale=180 EOD', {'stale': 180, 'stale_early': 120}))
V(('stale=60', {'stale': 60, 'stale_early': 30}))
V(('stale=45 ext=90', {'stale': 45, 'stale_early': 20, 'extended_hold_max': 90}))
V(('vxn=25', {'vxn_threshold': 25}))
V(('vxn=35', {'vxn_threshold': 35}))
V(('vxn=40', {'vxn_threshold': 40}))
V(('long-only vxn=30', {'two_sided': False, 'vxn_threshold': 30}))
V(('long-only vxn=35', {'two_sided': False, 'vxn_threshold': 35}))
V(('ext_thresh=2%', {'extended_hold_thresh': 2.0}))
V(('ext_thresh=10%', {'extended_hold_thresh': 10.0}))
V(('ext_max=180', {'extended_hold_max': 180, 'stale': 180}))
V(('short_sl=2%', {'short_sl': 2.0}))
V(('short_sl=4%', {'short_sl': 4.0}))
V(('short_act=2%', {'short_act': 2.0}))
V(('short_act=5%', {'short_act': 5.0}))
V(('3x$67', {'tranches': 3, 'tranche_pct': 0.33}))
V(('1x$200', {'tranches': 1, 'tranche_pct': 1.0}))
V(('4x$50', {'tranches': 4, 'tranche_pct': 0.25}))
V(('2 tr hostile', {'single_tranche_hostile': False}))
V(('fade_skip=0.5', {'fade_skip': 0.5}))
V(('fade_skip=0.8', {'fade_skip': 0.8}))
V(('no fade skip', {'fade_skip': 1.0}))
V(('short at vxn=25', {'vxn_threshold': 25}))
V(('short at vxn=35', {'vxn_threshold': 35}))
V(('cb=3', {'circuit_breaker': 3}))
V(('cb=10', {'circuit_breaker': 10}))

vxn = generate_vxn_timeseries(2000)
days_cache = {}
for m in ['NORMAL','HARSH','EXTREME','APOC']:
    days_cache[m] = [generate_day(vxn[i], stress_level=m) for i in range(2000)]

def run_custom(days, cfg):
    p = cfg
    trades = []
    tc = 200 * p['tranche_pct']
    cons = 0
    cp = 0
    for idx, day in enumerate(days):
        if cp > 0:
            cp -= 1
            cons = 0
            continue
        v = day.get('vxn', 20)
        short = p.get('two_sided', False) and v > p.get('vxn_threshold', 30)
        if not short and v > p.get('vxn_threshold', 30):
            continue
        if day['gap'] < p['min_gap']:
            continue
        if day['price'] < p['min_price']:
            continue
        if day['pre_vol'] < p['min_vol']:
            continue
        if day.get('rel_vol', 0) < p.get('rvol_floor', 0.5):
            continue
        sc = day['scenario']
        entry = day['open']
        bars = simulate_bars(sc, entry)
        skip = p.get('skip_open_bars', 0)
        bars = bars[skip:] if skip > 0 else bars
        if not bars:
            continue
        fp = detect_fade(bars[:3], day['gap'], day['rvol_trend'])
        if not short and fp > p['fade_skip']:
            continue
        tr = p['tranches']
        if not short and p.get('single_tranche_hostile', False) and v >= p.get('vxn_hostile', 99):
            tr = 1
        for _ in range(tr):
            if short:
                sl = p.get('short_sl', p['sl'])
                ta = p.get('short_act', p['trail_act'])
                td = p.get('short_dist', p['trail_dist'])
            else:
                if v >= p.get('vxn_hostile', 99):
                    sl = p.get('sl_hostile', p['sl'])
                else:
                    sl = p['sl']
                ta = p['trail_act']
                td = p['trail_dist']
            pos = {'entry': entry, 'extreme': entry, 'trail': False,
                   'ts': None, 'extended': False, 'extended_trail': False}
            ep = None
            rsn = None
            for mn, bar in enumerate(bars):
                if short:
                    g = (pos['entry'] - bar) / pos['entry'] * 100
                    if bar < pos['extreme']:
                        pos['extreme'] = bar
                else:
                    g = (bar - pos['entry']) / pos['entry'] * 100
                    if bar > pos['extreme']:
                        pos['extreme'] = bar
                if g <= -sl:
                    ep = bar
                    rsn = 'sl'
                    break
                if g >= ta and not pos['trail']:
                    pos['trail'] = True
                if pos['trail']:
                    if short:
                        pk = abs(pos['extreme'] - pos['entry']) / pos['entry'] * 100
                        if g <= pk - td:
                            ep = bar
                            rsn = 'trail'
                            break
                    else:
                        if pos['ts'] is None:
                            pos['ts'] = bar * (1 - td / 100)
                        nt = bar * (1 - td / 100)
                        if nt > pos['ts']:
                            pos['ts'] = nt
                        if bar <= pos['ts']:
                            ep = bar
                            rsn = 'trail'
                            break
                if not short:
                    ext_min = 30
                    if not pos.get('extended') and mn >= ext_min - 1:
                        if g > p.get('extended_hold_thresh', 5.0):
                            pos['extended'] = True
                            pos['extended_trail'] = True
                            pos['sb'] = p.get('extended_hold_max', 120)
                            if pos['trail']:
                                pos['ts'] = bar * (1 - td / 100)
                            continue
                    if mn >= p.get('stale_early', 60) - 1 and g < p.get('stale_thresh', 1.0) and not pos.get('extended'):
                        ep = bar
                        rsn = 'stale_early'
                        break
                stale_max = pos.get('sb', p['stale'])
                if mn >= max(p['stale'], stale_max) - 1:
                    ep = bar
                    rsn = 'stale'
                    break
            if ep is None:
                ep = bars[-1]
                rsn = 'eod'
            slip = day.get('slippage', 0.003)
            if short:
                ep *= (1 + slip)
            else:
                ep *= (1 - slip)
            if short:
                gp = (pos['entry'] - ep) / pos['entry'] * 100
            else:
                gp = (ep - pos['entry']) / pos['entry'] * 100
            pnl = gp / 100 * tc
            trades.append({'pnl': pnl, 'scenario': sc, 'day': idx})
            if pnl <= 0:
                cons += 1
            else:
                cons = 0
            cb = p.get('circuit_breaker', 5)
            if cb > 0 and cons >= cb:
                cp = 2
    return trades

# Run all
all_results = []
for name, mod in VARIANTS:
    cfg = dict(BASE)
    cfg.update(mod)
    total = 0
    for m in ['NORMAL','HARSH','EXTREME','APOC']:
        t = run_custom(days_cache[m], cfg)
        r = compute_metrics(t)
        if r['trades']:
            total += r['da']
    all_results.append((total / 4, name, mod))

all_results.sort(key=lambda x: -x[0])

# Get baseline
bl_avg = None
for avg, name, _ in all_results:
    if 'baseline' in name:
        bl_avg = avg
        break

print('=' * 100)
print('  TOP 20 STRATEGIES — ranked by avg daily profit across all 4 mixes')
print('=' * 100)
hdr = '  #   Strategy' + ' ' * 30 + 'NORMAL  HARSH  EXTREME  APOC   AVG    vs base'
print(hdr)
print('  ' + '-' * 95)

for i, (avg, name, mod) in enumerate(all_results[:20]):
    cfg = dict(BASE)
    cfg.update(mod)
    ds = []
    for m in ['NORMAL','HARSH','EXTREME','APOC']:
        t = run_custom(days_cache[m], cfg)
        r = compute_metrics(t)
        ds.append(r['da'] if r['trades'] else 0.0)
    dlt = avg - (bl_avg or 0)
    dlt_str = ('+' if dlt > 0 else '') + '${:.2f}'.format(dlt)
    print('  {:2d}. {:<32s} ${:<5.2f} ${:<5.2f} ${:<6.2f} ${:<5.2f} ${:<5.2f} {}'.format(
        i + 1, name, ds[0], ds[1], ds[2], ds[3], avg, dlt_str
    ))

# Winner
win_avg, win_name, win_mod = all_results[0]
print()
print('=' * 100)
print('  WINNER: ' + win_name)
print('=' * 100)
cfg = dict(BASE)
cfg.update(win_mod)
print('  Changed params:')
for k in sorted(cfg):
    b = BASE.get(k)
    if cfg[k] != b:
        print('    {}: {}  (was {})'.format(k, cfg[k], b))
print()
print('  Mix         Trades  WR     P&L      $/hr   Daily   MaxDD   DD%   PF')
print('  ' + '-' * 75)
for m in ['NORMAL','HARSH','EXTREME','APOC']:
    t = run_custom(days_cache[m], cfg)
    r = compute_metrics(t)
    dd_pct = r['mddpct'] if r['trades'] else 0
    print('  {:<12s} {:5d} {:4.0%}  ${:>6.0f}  ${:>4.2f}  ${:>5.2f}  ${:>5.0f}  {:4.1f}%  {:4.2f}'.format(
        m, r['trades'], r['wr'], r['tp'], r['dph'], r['da'], r['mdd'], dd_pct, r['pf']
    ))

print()
print('  MONTE CARLO (500 runs, EXTREME, circuit breaker):')
MC = 500
ps = []
for s in range(MC):
    random.seed(s + 1000)
    vd = generate_vxn_timeseries(1000)
    dd = [generate_day(vd[i], stress_level='EXTREME') for i in range(1000)]
    t = run_custom(dd, cfg)
    r = compute_metrics(t)
    ps.append(r['tp'])
ps.sort()
pos = sum(1 for p in ps if p > 0)
print('  Median: ${:.0f}  |  Profitable: {}/{} ({:.0%})'.format(ps[250], pos, MC, pos / MC))
print('  5th: ${:.0f}  |  95th: ${:.0f}'.format(ps[25], ps[475]))
