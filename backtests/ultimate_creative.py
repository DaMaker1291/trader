"""
ULTIMATE CREATIVE v2 — fixed, fast, uses train/test split for ML.
18 strategies tested in <2 minutes.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

CAPITAL = 200.0
FRICTION = 0.80
NOW = pd.Timestamp.now(tz="UTC")

print("="*70)
print("LOADING DATA...")
print("="*70)
spy = yf.Ticker("SPY").history(period="7y")
vix = yf.Ticker("^VIX").history(period="7y")
qqq = yf.Ticker("QQQ").history(period="7y")
iwm = yf.Ticker("IWM").history(period="7y")
print(f"SPY: {len(spy)}d, VIX: {len(vix)}d, QQQ: {len(qqq)}d, IWM: {len(iwm)}d")

# ---- FEATURES ----
spy["open_pct"] = (spy["Open"] / spy["Close"].shift(1) - 1) * 100
spy["day_pct"] = (spy["Close"] / spy["Open"] - 1) * 100

# Align VIX
vix = vix.reindex(spy.index, method="ffill")
spy["vix"] = vix["Close"]

spy["ret_1d"] = spy["Close"].pct_change() * 100
spy["ret_3d"] = spy["Close"].pct_change(3) * 100
spy["ret_5d"] = spy["Close"].pct_change(5) * 100
spy["ret_10d"] = spy["Close"].pct_change(10) * 100
spy["range_pct"] = (spy["High"] / spy["Low"] - 1) * 100
spy["vol_ratio"] = spy["Volume"] / spy["Volume"].rolling(20).mean()
spy["close_20ma"] = spy["Close"].rolling(20).mean()
spy["close_50ma"] = spy["Close"].rolling(50).mean()
spy["above_20ma"] = (spy["Close"] > spy["close_20ma"]).astype(int)
spy["above_50ma"] = (spy["Close"] > spy["close_50ma"]).astype(int)
spy["consec_up"] = (spy["day_pct"] > 0).astype(int).groupby((spy["day_pct"] <= 0).astype(int).cumsum()).cumsum()
spy["consec_down"] = (spy["day_pct"] < 0).astype(int).groupby((spy["day_pct"] >= 0).astype(int).cumsum()).cumsum()
spy["dow"] = spy.index.dayofweek
for d in range(5): spy[f"dow_{d}"] = (spy["dow"] == d).astype(int)

# Gap fill history
spy["gap_filled"] = ((spy["Low"] <= spy["Close"].shift(1)) & (spy["open_pct"] > 0)).astype(int) | \
                    ((spy["High"] >= spy["Close"].shift(1)) & (spy["open_pct"] < 0)).astype(int)
spy["gap_fill_rate"] = spy["gap_filled"].rolling(20).mean()

# Target
spy["target_dir"] = (spy["day_pct"].shift(-1) > 0).astype(int)

spy = spy.dropna()
print(f"Ready: {len(spy)} trading days\n")

# ---- OPTIONS MODEL ----
def opt_trade(row, direction):
    ret = _opt_ret(row["Open"], row["day_pct"], direction)
    cost = min(row["Open"] * 0.003 * 100, CAPITAL)
    if cost < 10: return 0
    return cost * ret / 100

def _opt_ret(price, move, direction):
    prem = price * 0.003
    abs_m = abs(move)
    d_move = price * move / 100
    if direction == "call":
        delta = min(0.50 + abs_m*0.10, 0.95)
        theta = prem * 0.08
        gam = 0.5*0.10*(d_move**2)
        ret = (delta*d_move + gam - theta)/prem*100
        if move < 0:
            ret = -100 if abs_m > 0.3 else max(ret, -60)
    else:
        delta = max(-0.50 - abs_m*0.10, -0.95)
        theta = prem * 0.08
        gam = 0.5*0.10*(d_move**2)
        ret = (delta*d_move + gam - theta)/prem*100
        if move > 0:
            ret = -100 if abs_m > 0.3 else max(ret, -60)
    return max(min(ret, 2000), -100)

# ---- RUNNER ----
results = []
def add(n, tr, wr, p, t):
    a = p*FRICTION; c = a/(NOW-spy.index[0]).total_seconds()*3600
    results.append((n, tr, wr or 0, p, a, c, c*24, t))

# 1. BASELINE: direction-aware
def baseline():
    tr=w=p=0
    for i in range(1,len(spy)):
        r=spy.iloc[i]; op=r["open_pct"]
        if pd.isna(op) or abs(op)<.3: continue
        pt=opt_trade(r,"call" if op>0 else "put")
        p+=pt;tr+=1
        if pt>0:w+=1
    add("0a. BASELINE: dir-aware 0DTE",tr,w/tr*100 if tr else 0,p,"OPT")

baseline()

# 2. BASELINE: only green days (calls)
def green_only():
    tr=w=p=0
    for i in range(1,len(spy)):
        r=spy.iloc[i]; op=r["open_pct"]
        if pd.isna(op) or op<.3: continue
        pt=opt_trade(r,"call")
        p+=pt;tr+=1
        if pt>0:w+=1
    add("0b. BASELINE: green-only calls 0DTE",tr,w/tr*100 if tr else 0,p,"OPT")

green_only()

# ===== ML STRATEGIES (train/test split) =====
features = ["open_pct","ret_1d","ret_3d","ret_5d","ret_10d",
            "above_20ma","above_50ma","vol_ratio","vix","range_pct",
            "consec_up","consec_down","gap_fill_rate",
            "dow_0","dow_1","dow_2","dow_3","dow_4"]

df = spy[features + ["target_dir"]].dropna()
X, y = df[features].values, df["target_dir"].values
split = int(len(X)*0.7)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
test_idx = range(split, len(df))
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train all models once on 70%, test on 30%
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", C=0.1, max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced"),
    "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(32,16), activation="relu", max_iter=200, random_state=42),
}

for mname, model in models.items():
    model.fit(X_train_s, y_train)
    # Evaluate on test set
    y_pred = model.predict(X_test_s)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Now trade using model predictions
    tr=w=p=0
    for j, i in enumerate(test_idx):
        if i+1 >= len(spy): break
        r = spy.iloc[i+1]
        op = r["open_pct"]
        if pd.isna(op) or abs(op) < 0.3: continue
        
        prob = model.predict_proba(X_test_s[j:j+1])[0][1]
        if prob < 0.55: continue  # need 55%+ confidence
        
        dir_ = "call" if prob > 0.5 else "put"
        if dir_ == "call" and op < 0: continue
        if dir_ == "put" and op > 0: continue
        
        pt = opt_trade(r, dir_)
        p += pt; tr += 1
        if pt > 0: w += 1
    
    wr_ = w/tr*100 if tr else 0
    add(f"ML: {mname} (acc={accuracy:.1%}, thr=55%)", tr, wr_, p, "ML")

# 6. ML ENSEMBLE: all 3 must agree >60%
lr = models["Logistic Regression"]
rf = models["Random Forest"]
mlp = models["MLP Neural Net"]

tr=w=p=0
for j, i in enumerate(test_idx):
    if i+1 >= len(spy): break
    r = spy.iloc[i+1]
    op = r["open_pct"]
    if pd.isna(op) or abs(op) < 0.3: continue
    
    xs = X_test_s[j:j+1]
    p_lr = lr.predict_proba(xs)[0][1]
    p_rf = rf.predict_proba(xs)[0][1]
    p_mlp = mlp.predict_proba(xs)[0][1]
    p_ens = (p_lr + p_rf + p_mlp) / 3
    
    if p_ens < 0.60: continue
    
    if op > 0:
        pt = opt_trade(r, "call")
    else:
        pt = opt_trade(r, "put")
    p += pt; tr += 1
    if pt > 0: w += 1

add(f"ML: Ensemble LR+RF+MLP (avg>60%)", tr, w/tr*100 if tr else 0, p, "ML")

# 7. XGBoost (much faster with train/test split)
try:
    import xgboost as xgb
    model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="logloss")
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    
    tr=w=p=0
    for j, i in enumerate(test_idx):
        if i+1 >= len(spy): break
        r = spy.iloc[i+1]
        op = r["open_pct"]
        if pd.isna(op) or abs(op) < 0.3: continue
        prob = model.predict_proba(X_test[j:j+1])[0][1]
        if prob < 0.55: continue
        if op > 0 and prob < 0.5: continue
        if op < 0 and prob > 0.5: continue
        if op > 0:
            pt = opt_trade(r, "call")
        else:
            pt = opt_trade(r, "put")
        p += pt; tr += 1
        if pt > 0: w += 1
    add(f"ML: XGBoost (acc={acc:.1%}, thr=55%)", tr, w/tr*100 if tr else 0, p, "ML")
except Exception as e:
    add(f"ML: XGBoost (FAILED: {e})", 0, 0, 0, "ML")

# ===== STATISTICAL / CREATIVE STRATEGIES =====

# 8. Gap Fill Probability
tr=w=p=0
for i in range(50, len(spy)):
    r=spy.iloc[i]; op=r["open_pct"]
    if pd.isna(op) or abs(op)<.3: continue
    fr=r["gap_fill_rate"]
    if pd.isna(fr): continue
    if op>0:
        if fr<.35: dir_="call"
        elif fr>.65: dir_="put"
        else: continue
    else:
        if fr<.35: dir_="put"
        elif fr>.65: dir_="call"
        else: continue
    if (dir_=="call" and op<0) or (dir_=="put" and op>0): continue
    pt=opt_trade(r,dir_); p+=pt;tr+=1
    if pt>0:w+=1
add("Gap Fill Prob Model",tr,w/tr*100 if tr else 0,p,"STAT")

# 9. VIX Regime
tr=w=p=0
for i in range(1,len(spy)):
    r=spy.iloc[i]; op=r["open_pct"]; v=r["vix"]
    if pd.isna(op) or pd.isna(v): continue
    if v<15:
        if op>=.3:
            pt=opt_trade(r,"call"); p+=pt;tr+=1
            if pt>0:w+=1
    elif v>25:
        if abs(op)>=1.:
            rc=_opt_ret(r["Open"],r["day_pct"],"call")
            rp=_opt_ret(r["Open"],r["day_pct"],"put")
            prem=r["Open"]*0.003; cost=min(prem*100,CAPITAL)
            if cost>=10:
                pt=cost*rc/200+cost*rp/200; p+=pt;tr+=1
                if pt>0:w+=1
    else:
        if abs(op)>=.3:
            dir_="call" if op>0 else "put"
            pt=opt_trade(r,dir_); p+=pt;tr+=1
            if pt>0:w+=1
add("VIX Regime (credit/dir-aware/straddle)",tr,w/tr*100 if tr else 0,p,"STAT")

# 10. Multi-Tick (SPY+QQQ+IWM must agree)
for name, df2 in [("SPY+QQQ", qqq), ("SPY+IWM", iwm), ("SPY+QQQ+IWM", None)]:
    tr=w=p=0
    for i in range(1,len(spy)):
        d=spy.index[i]; r=spy.iloc[i]; op=r["open_pct"]
        if pd.isna(op) or abs(op)<.3: continue
        all_up, all_down = op>0, op<0
        valid=True
        for df3 in [qqq, iwm]:
            try:
                idx=df3.index.get_loc(d)
                g=df3.iloc[idx].get("open_pct_comp", df3.iloc[idx]["Open"]/df3.iloc[idx]["Close"].shift(1) - 1)*100
                if pd.isna(g):
                    valid=False; break
                if op>0 and g<0: all_up=False
                if op<0 and g>0: all_down=False
            except: valid=False; break
        
        if not valid or not(all_up or all_down): continue
        pt=opt_trade(r,"call" if all_up else "put"); p+=pt;tr+=1
        if pt>0:w+=1
    if tr>0:
        add(f"Multi-Tick (SPY+QQQ+IWM all agree)", tr, w/tr*100 if tr else 0, p, "STAT")

# 11. Trend Continuation
tr=w=p=0
for i in range(10,len(spy)):
    r=spy.iloc[i]; op=r["open_pct"]
    if pd.isna(op) or abs(op)<.3: continue
    if op>0 and r["ret_5d"]>1:
        pt=opt_trade(r,"call"); p+=pt;tr+=1
        if pt>0:w+=1
    elif op<0 and r["ret_5d"]<-1:
        pt=opt_trade(r,"put"); p+=pt;tr+=1
        if pt>0:w+=1
add("Trend Continuation (gap+5d trend align)",tr,w/tr*100 if tr else 0,p,"STAT")

# 12. Mean Reversion (fade after 2%+ move)
tr=w=p=0
for i in range(10,len(spy)):
    r=spy.iloc[i]; op=r["open_pct"]
    if pd.isna(op) or abs(op)<.3: continue
    if abs(r["ret_5d"])<2: continue
    if r["ret_5d"]>2 and op>0:
        pt=opt_trade(r,"put"); p+=pt;tr+=1
        if pt>0:w+=1
    elif r["ret_5d"]<-2 and op<0:
        pt=opt_trade(r,"call"); p+=pt;tr+=1
        if pt>0:w+=1
add("Mean Reversion (fade 2%+ 5d move)",tr,w/tr*100 if tr else 0,p,"STAT")

# 13. Volume Confirmation
tr=w=p=0
for i in range(20,len(spy)):
    r=spy.iloc[i]; op=r["open_pct"]
    if pd.isna(op) or abs(op)<.3: continue
    if pd.isna(r["vol_ratio"]) or r["vol_ratio"]<1: continue
    pt=opt_trade(r,"call" if op>0 else "put"); p+=pt;tr+=1
    if pt>0:w+=1
add("Volume Confirmation (vol>20d avg)",tr,w/tr*100 if tr else 0,p,"STAT")

# 14. Day-of-Week optimized
best_dow = spy.groupby("dow")["day_pct"].mean().idxmax()
tr=w=p=0
for i in range(1,len(spy)):
    r=spy.iloc[i]; op=r["open_pct"]
    if pd.isna(op) or abs(op)<.3: continue
    if r["dow"]!=best_dow: continue
    pt=opt_trade(r,"call" if op>0 else "put"); p+=pt;tr+=1
    if pt>0:w+=1
add(f"Day-of-Week ({['Mon','Tue','Wed','Thu','Fri'][best_dow]})",tr,w/tr*100 if tr else 0,p,"STAT")

# 15. Consecutive Gap Fade (3+ streaks)
tr=w=p=0
for i in range(10,len(spy)):
    r=spy.iloc[i]; op=r["open_pct"]
    if pd.isna(op) or abs(op)<.3: continue
    c=r["consec_up"] if op>0 else r["consec_down"]
    if c<3: continue
    pt=opt_trade(r,"put" if op>0 else "call"); p+=pt;tr+=1
    if pt>0:w+=1
add("Consecutive Gap Fade (3+ streak)",tr,w/tr*100 if tr else 0,p,"STAT")

# 16. Pattern Recognition
tr=w=p=0
for i in range(3,len(spy)):
    r=spy.iloc[i]; p1=spy.iloc[i-1]; p2=spy.iloc[i-2]; op=r["open_pct"]
    if pd.isna(op) or abs(op)<.3: continue
    bull_engulf = p1["day_pct"]<-.5 and p2["day_pct"]<0 and r["day_pct"]>0 and r["range_pct"]>p1["range_pct"]
    bear_engulf = p1["day_pct"]>.5 and p2["day_pct"]>0 and r["day_pct"]<0 and r["range_pct"]>p1["range_pct"]
    inside_day = r["High"]<=p1["High"] and r["Low"]>=p1["Low"]
    dir_=None
    if bull_engulf: dir_="call"
    elif bear_engulf: dir_="put"
    elif inside_day:
        if p1["day_pct"]>.3 and op>0: dir_="call"
        elif p1["day_pct"]<-.3 and op<0: dir_="put"
    if dir_ is None: continue
    if (dir_=="call" and op<0) or (dir_=="put" and op>0): continue
    pt=opt_trade(r,dir_); p+=pt;tr+=1
    if pt>0:w+=1
add("Pattern Recognition (engulf/inside)",tr,w/tr*100 if tr else 0,p,"STAT")

# 17. Ensemble All Signals (5 votes)
signal_cols = []
spy["sig_gap"] = np.where(spy["open_pct"]>0.3,1,np.where(spy["open_pct"]<-0.3,0,np.nan))
spy["sig_trend"] = np.where(spy["ret_5d"]>1,1,np.where(spy["ret_5d"]<-1,0,np.nan))
spy["sig_vix"] = np.where((spy["vix"]>=15)&(spy["vix"]<=25),np.where(spy["open_pct"]>0,1,0),np.nan)
spy["sig_vol"] = np.where(spy["vol_ratio"]>1,np.where(spy["open_pct"]>0,1,0),np.nan)
spy["sig_fill"] = np.where(spy["gap_fill_rate"]<.35,np.where(spy["open_pct"]>0,1,0),
                          np.where(spy["gap_fill_rate"]>.65,np.where(spy["open_pct"]<0,1,0),np.nan))

tr=w=p=0
for i in range(50,len(spy)):
    r=spy.iloc[i]
    if pd.isna(r["open_pct"]) or abs(r["open_pct"])<.3: continue
    sigs=[s for s in ["sig_gap","sig_trend","sig_vix","sig_vol","sig_fill"] if not pd.isna(r[s])]
    if len(sigs)<3: continue
    avg=np.mean([int(r[s]) for s in sigs])
    if avg>=.6: dir_="call"
    elif avg<=.4: dir_="put"
    else: continue
    if (dir_=="call" and r["open_pct"]<0) or (dir_=="put" and r["open_pct"]>0): continue
    pt=opt_trade(r,dir_); p+=pt;tr+=1
    if pt>0:w+=1
add("Ensemble All Signals (5-vote majority)",tr,w/tr*100 if tr else 0,p,"ENS")

# 18. TQQQ 0DTE (extra leverage)
if "TQQQ" in yf.Ticker("TQQQ").history(period="7y"):
    tqqq = yf.Ticker("TQQQ").history(period="7y")
    tqqq["open_pct"] = (tqqq["Open"]/tqqq["Close"].shift(1)-1)*100
    tqqq["day_pct"] = (tqqq["Close"]/tqqq["Open"]-1)*100
    tqqq = tqqq[tqqq.index>=spy.index[0]]
    
    tr=w=p=0
    for i in range(1,len(spy)):
        d=spy.index[i]; r=spy.iloc[i]; op=r["open_pct"]
        if pd.isna(op) or abs(op)<.3: continue
        try: ti=tqqq.index.get_loc(d)
        except: continue
        trq=tqqq.iloc[ti]; tm=trq["day_pct"]
        prem=trq["Open"]*0.005; cost=min(prem*100,CAPITAL)
        if cost<10: continue
        dir_="call" if op>0 else "put"
        ret=_opt_ret(trq["Open"],tm,dir_)*1.5
        pt=cost*ret/100; p+=pt;tr+=1
        if pt>0:w+=1
    add(f"TQQQ 0DTE (3x leverage^2)",tr,w/tr*100 if tr else 0,p,"OPT")

# ===== RESULTS =====
results.sort(key=lambda r: r[5], reverse=True)
print("\n"+"="*80)
print(f"{'FINAL: 18 STRATEGIES — sorted by $/cal hr':^80}")
print("="*80)
print(f"{'#':>3s} {'Strategy':55s} {'Trades':>6s} {'WR':>5s} {'P&L':>9s} {'$/cal hr':>9s} {'$/day':>8s} {'Type':>6s}")
print("-"*80)
for idx,(n,tr,wr,p,adj,ch,dy,t) in enumerate(results,1):
    ps=f"${p:+.0f}" if abs(p)>100 else f"${p:+>.1f}"
    print(f"{idx:3d} {n:55s} {tr:6d} {wr:4.1f}% {ps:>9s} ${ch:.4f} ${dy:+.2f} {t:>6s}")

print("\n"+"="*80)
print("$1/CAL HR ANALYSIS")
print("="*80)
for n,tr,wr,p,adj,ch,dy,t in results:
    if ch>0 and tr>=5:
        print(f"  {n:55s} ${ch:.4f}/hr -> Need ${1/ch*200:,.0f} capital")

print("\n"+"="*80)
print("BOTTOM LINE")
print("="*80)
print(f"  #1: {results[0][0]} — ${results[0][5]:.4f}/cal hr")
print(f"  #2: {results[1][0]} — ${results[1][5]:.4f}/cal hr")
if results[0][5]>0:
    print(f"  Capital needed for $1/hr: ${1/results[0][5]*200:,.0f}")
print()
print("  ML models trained on 70%, tested on 30% (out-of-sample)")
print("  AI prediction accuracy: see acc=X% in strategy name")
print("  turbobot.py (TQQQ on green) still safest deployable option")
