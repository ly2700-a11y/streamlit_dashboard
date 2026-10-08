#!/usr/bin/env python3
"""Compare CLC RAD vs Yahoo Finance for 5 contracts, then full portfolio."""
import pandas as pd, numpy as np
from data_loader import load_clc_full
from config import ASSET_CLASSES, BP, TRADING_DAYS, PAPER_TABLE3, METRIC_NAMES
from metrics import compute_metrics

EWMA=60; ST=0.063; B=0.002
core = ['E(R)','std(R)','Sharpe','% +ve','Ave P/L']
yf_map = {'CC':'CC=F','LB':'LBS=F','JO':'OJ=F','ZO':'ZO=F','ZH':'HO=F'}

def load_yf(tk):
    df = pd.read_csv(f'data/yahoo/{tk}_yahoo.csv', header=[0,1], index_col=0, parse_dates=True)
    # Flatten multi-level columns
    df.columns = [c[0] for c in df.columns]
    p = pd.to_numeric(df['Close'], errors='coerce').values
    d = df.index.values
    mask = (d >= np.datetime64('2009-01-01')) & (d <= np.datetime64('2019-12-31'))
    p = p[mask]; d = d[mask]
    valid = np.isfinite(p) & (p > 0)
    return d[valid], p[valid]

def backtest(prices, dates):
    n = len(prices)
    rt = np.zeros(n); rt[1:] = prices[1:] - prices[:-1]
    sig = pd.Series(rt).ewm(span=EWMA, adjust=False).std().values
    ms = dates >= np.datetime64('2011-01-01'); me = dates <= np.datetime64('2019-12-31')
    if not ms.any() or not me.any(): return None, None
    t0 = max(np.argmax(ms), 252); t1 = n-1-np.argmax(me[::-1])
    Rt = np.zeros(n)
    for t in range(1,n):
        if sig[t-1]>0 and (t<2 or sig[t-2]>0):
            sp=ST/sig[t-1]; spp=ST/sig[t-2] if t>=2 else 0
            Rt[t]=sp*rt[t]-B*abs(prices[t-1])*abs(sp-spp)
    return Rt[t0:t1], np.mean(sig[t0:t1])

# Per-contract comparison
print(f"{'TK':4s} | {'CLC E(R)':>9s} {'CLC σ':>7s} | {'YF E(R)':>9s} {'YF σ':>7s} | {'YF_p':>9s} {'CLC_p':>9s}")
print("-"*75)
for tk in ['CC','LB','JO','ZO','ZH']:
    df = load_clc_full(tk)
    p_c = df['Close'].astype(float).values; d_c = df['Date'].values
    slc_c, sig_c = backtest(p_c, d_c)
    er_c = np.mean(slc_c)*252
    ms = d_c>=np.datetime64('2011-01-01'); me=d_c<=np.datetime64('2019-12-31')
    t0=max(np.argmax(ms),252); t1=len(d_c)-1-np.argmax(me[::-1])
    pmean_c = np.mean(p_c[t0:t1])

    d_y, p_y = load_yf(tk)
    slc_y, sig_y = backtest(p_y, d_y)
    er_y = np.mean(slc_y)*252
    ms2=d_y>=np.datetime64('2011-01-01'); me2=d_y<=np.datetime64('2019-12-31')
    t02=max(np.argmax(ms2),252); t12=len(d_y)-1-np.argmax(me2[::-1])
    pmean_y = np.mean(p_y[t02:t12])

    print(f"{tk:4s} | {er_c:+9.4f} {sig_c:7.2f} | {er_y:+9.4f} {sig_y:7.2f} | {pmean_y:9.1f} {pmean_c:9.1f}")

# Full portfolio: 5 YF + 45 CLC
print(f"\n{'='*75}")
print("Full portfolio: 5 Yahoo + 45 CLC RAD")
print(f"{'='*75}")
total_n10=total_n15=0
for ac_name in ['Commodity','Equity Index','Fixed Income','Forex']:
    tickers = ASSET_CLASSES[ac_name]
    port_ret = []
    for tk in tickers:
        if tk in yf_map:
            d,p = load_yf(tk)
        else:
            df = load_clc_full(tk)
            if df is None: continue
            p = df['Close'].astype(float).values; d = df['Date'].values
        if len(p)<500: continue
        slc, _ = backtest(p, d)
        if slc is None or len(slc)<100: continue
        ms=d>=np.datetime64('2011-01-01'); me=d<=np.datetime64('2019-12-31')
        t0=max(np.argmax(ms),252); t1=len(d)-1-np.argmax(me[::-1])
        dd=d[t0:t1][:len(slc)]
        port_ret.append(pd.Series(slc[:len(dd)], index=dd))

    df_p=pd.DataFrame(port_ret); Rp=df_p.T.mean(axis=1).values
    ma=compute_metrics(Rp,len(port_ret)); md=dict(zip(METRIC_NAMES,ma))
    paper=PAPER_TABLE3[ac_name]['Long']
    errs={k:abs((md[k]-paper[k])/abs(paper[k]))*100 if abs(paper[k])>1e-6 else 0 for k in core}
    n10=sum(1 for e in errs.values() if e<10); n15=sum(1 for e in errs.values() if e<15)
    total_n10+=n10; total_n15+=n15
    print(f"\n{ac_name} ({len(port_ret)}) n10={n10}/5 n15={n15}/5")
    for k in core:
        print(f"  {k:<10} {md[k]:>+8.3f} vs {paper[k]:>+8.3f}  err={errs[k]:.1f}%")

print(f"\nTOTAL: n10={total_n10}/20  n15={total_n15}/20")
