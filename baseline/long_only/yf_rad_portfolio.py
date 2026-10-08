#!/usr/bin/env python3
"""YF_RAD for LB/JO/ZO + CLC_RAD for CC/ZH + CLC_RAD for rest 20"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np
from data_loader import load_clc_full
from config import ASSET_CLASSES, BP, TRADING_DAYS, PAPER_TABLE3, METRIC_NAMES
from metrics import compute_metrics

EWMA=60; ST=0.063; B=0.002
core = ['E(R)','std(R)','Sharpe','% +ve','Ave P/L']

def load_yf_rad(tk):
    """Load Yahoo NON, apply CLC roll dates/ratios to generate YF_RAD"""
    # YF NON
    df = pd.read_csv(f'data/yahoo/{tk}_yahoo.csv', header=[0,1], index_col=0, parse_dates=True)
    df.columns = [c[0] for c in df.columns]
    p_yf = pd.to_numeric(df['Close'], errors='coerce').values.flatten()
    d_yf = df.index.values
    mask = (d_yf >= np.datetime64('2009-01-01')) & (d_yf <= np.datetime64('2019-12-31'))
    p_yf = p_yf[mask]; d_yf = d_yf[mask]
    valid = np.isfinite(p_yf) & (p_yf > 0)
    p_yf = p_yf[valid]; d_yf = d_yf[valid]

    # CLC rolls
    non = pd.read_csv(f'data/CLC/{tk}_NON.CSV', header=None, names=['D','O','H','L','C','V','OI'])
    rev = pd.read_csv(f'data/CLC/{tk}_REV.CSV', header=None, names=['D','O','H','L','C','V','OI'])
    non['D'] = pd.to_datetime(non['D'], format='%m/%d/%Y')
    rev['D'] = pd.to_datetime(rev['D'], format='%m/%d/%Y')
    m = non[['D','C']].merge(rev[['D','C']], on='D', suffixes=('_n','_r'))
    m = m.sort_values('D').reset_index(drop=True)
    p_n = m['C_n'].astype(float).values; p_r = m['C_r'].astype(float).values
    d_clc = m['D'].values
    cm = d_clc >= np.datetime64('2009-01-01')
    p_n=p_n[cm]; p_r=p_r[cm]; d_clc=d_clc[cm]
    adj = p_r - p_n; adj_diff = np.diff(adj)
    rolls = np.where(np.abs(adj_diff) > 1e-6)[0]

    # Map CLC roll dates to YF indices
    yf_date_map = {}
    for i, d in enumerate(d_yf):
        yf_date_map[pd.Timestamp(d)] = i

    cum = np.ones(len(p_yf))
    matched = 0
    for j in rolls:
        roll_date = pd.Timestamp(d_clc[j])
        if roll_date in yf_date_map:
            yj = yf_date_map[roll_date]
            ratio = p_n[j] / (p_n[j] - adj_diff[j])
            if yj+1 < len(cum): cum[yj+1:] *= ratio; matched += 1

    p_rad = p_yf * cum
    pos = p_rad > 0
    return d_yf[pos], p_rad[pos]

total_n10=total_n15=0
for ac_name in ['Commodity','Equity Index','Fixed Income','Forex']:
    tickers = ASSET_CLASSES[ac_name]
    port_ret = []
    for tk in tickers:
        if tk in ('LB','JO','ZO'):
            d, p = load_yf_rad(tk)
        else:
            df = load_clc_full(tk)
            if df is None: continue
            p = df['Close'].astype(float).values; d = df['Date'].values
        if len(p)<500: continue
        n=len(p); rt=np.zeros(n); rt[1:]=p[1:]-p[:-1]
        sig=pd.Series(rt).ewm(span=EWMA,adjust=False).std().values
        ms=d>=np.datetime64('2011-01-01'); me=d<=np.datetime64('2019-12-31')
        if not ms.any() or not me.any(): continue
        t0=max(np.argmax(ms),252); t1=n-1-np.argmax(me[::-1])
        Rt=np.zeros(n)
        for t in range(1,n):
            if sig[t-1]>0 and (t<2 or sig[t-2]>0):
                sp=ST/sig[t-1]; spp=ST/sig[t-2] if t>=2 else 0
                Rt[t]=sp*rt[t]-B*abs(p[t-1])*abs(sp-spp)
        slc=Rt[t0:t1]; dd=d[t0:t1][:len(slc)]
        if len(slc)>100: port_ret.append(pd.Series(slc[:len(dd)], index=dd))

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
