#!/usr/bin/env python3
"""Commodity Long baseline: all data source combinations for 5 contracts"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, numpy as np
from data_loader import load_clc_full
from config import ASSET_CLASSES, BP, TRADING_DAYS, PAPER_TABLE3, METRIC_NAMES
from metrics import compute_metrics

EWMA=60; ST=0.063; B=0.002
core = ['E(R)','std(R)','Sharpe','% +ve','Ave P/L']
neg5 = {'CC','LB','JO','ZO','ZH'}

def load_yf_non(tk):
    df = pd.read_csv(f'data/yahoo/{tk}_yahoo.csv', header=[0,1], index_col=0, parse_dates=True)
    df.columns = [c[0] for c in df.columns]
    p = pd.to_numeric(df['Close'], errors='coerce').values.flatten()
    d = df.index.values
    mask = (d >= np.datetime64('2009-01-01')) & (d <= np.datetime64('2019-12-31'))
    p=p[mask]; d=d[mask]
    valid = np.isfinite(p)&(p>0)
    return d[valid], p[valid]

def load_yf_rad(tk):
    d_yf, p_yf = load_yf_non(tk)
    # CLC rolls
    non = pd.read_csv(f'data/CLC/{tk}_NON.CSV', header=None, names=['D','O','H','L','C','V','OI'])
    rev = pd.read_csv(f'data/CLC/{tk}_REV.CSV', header=None, names=['D','O','H','L','C','V','OI'])
    non['D']=pd.to_datetime(non['D'],format='%m/%d/%Y'); rev['D']=pd.to_datetime(rev['D'],format='%m/%d/%Y')
    m=non[['D','C']].merge(rev[['D','C']],on='D',suffixes=('_n','_r'))
    m=m.sort_values('D').reset_index(drop=True)
    p_n=m['C_n'].astype(float).values; p_r=m['C_r'].astype(float).values; d_c=m['D'].values
    cm=d_c>=np.datetime64('2009-01-01'); p_n=p_n[cm]; p_r=p_r[cm]; d_c=d_c[cm]
    adj=p_r-p_n; adj_diff=np.diff(adj)
    rolls=np.where(np.abs(adj_diff)>1e-6)[0]
    yf_map={pd.Timestamp(d):i for i,d in enumerate(d_yf)}
    cum=np.ones(len(p_yf))
    for j in rolls:
        rd=pd.Timestamp(d_c[j])
        if rd in yf_map:
            yj=yf_map[rd]; ratio=p_n[j]/(p_n[j]-adj_diff[j])
            if yj+1<len(cum): cum[yj+1:]*=ratio
    p_rad=p_yf*cum
    pos=p_rad>0
    return d_yf[pos], p_rad[pos]

def load_clc_rev(tk):
    non=pd.read_csv(f'data/CLC/{tk}_NON.CSV',header=None,names=['D','O','H','L','C','V','OI'])
    rev=pd.read_csv(f'data/CLC/{tk}_REV.CSV',header=None,names=['D','O','H','L','C','V','OI'])
    non['D']=pd.to_datetime(non['D'],format='%m/%d/%Y'); rev['D']=pd.to_datetime(rev['D'],format='%m/%d/%Y')
    m=non[['D','C']].merge(rev[['D','C']],on='D',suffixes=('_n','_r'))
    m=m.sort_values('D').reset_index(drop=True)
    mask=(m['D']>='2009-01-01')&(m['D']<='2019-12-31')
    sub=m[mask].reset_index(drop=True)
    p=sub['C_r'].astype(float).values; d=sub['D'].values
    valid=np.isfinite(p)&(p!=0)
    return d[valid], p[valid]

def load_clc_non(tk):
    non=pd.read_csv(f'data/CLC/{tk}_NON.CSV',header=None,names=['D','O','H','L','C','V','OI'])
    non['D']=pd.to_datetime(non['D'],format='%m/%d/%Y')
    mask=(non['D']>='2009-01-01')&(non['D']<='2019-12-31')
    sub=mask[mask].reset_index(drop=True) if mask.sum()>0 else non
    # Simpler:
    sub=non[non['D']>='2009-01-01'].reset_index(drop=True)
    sub=sub[sub['D']<='2019-12-31'].reset_index(drop=True)
    p=sub['C'].astype(float).values; d=sub['D'].values
    valid=p>0
    return d[valid], p[valid]

def bt(prices, dates):
    n=len(prices); rt=np.zeros(n); rt[1:]=prices[1:]-prices[:-1]
    sig=pd.Series(rt).ewm(span=EWMA,adjust=False).std().values
    d=dates; ms=d>=np.datetime64('2011-01-01'); me=d<=np.datetime64('2019-12-31')
    if not ms.any() or not me.any(): return None
    t0=max(np.argmax(ms),252); t1=n-1-np.argmax(me[::-1])
    Rt=np.zeros(n)
    for t in range(1,n):
        if sig[t-1]>0 and (t<2 or sig[t-2]>0):
            sp=ST/sig[t-1]; spp=ST/sig[t-2] if t>=2 else 0
            Rt[t]=sp*rt[t]-B*abs(prices[t-1])*abs(sp-spp)
    return Rt[t0:t1]

def run_portfolio(five_source):
    """five_source: dict mapping tk in neg5 to source name"""
    loaders = {
        'CLC_NON': lambda tk: load_clc_non(tk),
        'CLC_REV': lambda tk: load_clc_rev(tk),
        'CLC_RAD': lambda tk: (load_clc_full(tk)['Date'].values, load_clc_full(tk)['Close'].astype(float).values) if load_clc_full(tk) is not None else (None,None),
        'YF_NON': lambda tk: load_yf_non(tk),
        'YF_RAD': lambda tk: load_yf_rad(tk),
    }
    
    port_ret = []
    for tk in ASSET_CLASSES['Commodity']:
        src = five_source.get(tk, 'CLC_RAD')
        d, p = loaders[src](tk)
        if d is None or len(p)<500: continue
        slc = bt(p, d)
        if slc is None or len(slc)<100: continue
        ms=d>=np.datetime64('2011-01-01'); me=d<=np.datetime64('2019-12-31')
        t0=max(np.argmax(ms),252); t1=len(d)-1-np.argmax(me[::-1])
        dd=d[t0:t1][:len(slc)]
        port_ret.append(pd.Series(slc[:len(dd)],index=dd))
    
    df_p=pd.DataFrame(port_ret); Rp=df_p.T.mean(axis=1).values
    ma=compute_metrics(Rp,len(port_ret)); md=dict(zip(METRIC_NAMES,ma))
    paper=PAPER_TABLE3['Commodity']['Long']
    errs={k:abs((md[k]-paper[k])/abs(paper[k]))*100 if abs(paper[k])>1e-6 else 0 for k in core}
    n10=sum(1 for e in errs.values() if e<10); n15=sum(1 for e in errs.values() if e<15)
    return md, errs, n10, n15, len(port_ret)

# Define all combos to test
combos = [
    ({}, "全CLC_RAD"),
    ({'CC':'YF_RAD','LB':'YF_RAD','JO':'YF_RAD','ZO':'YF_RAD','ZH':'YF_RAD'}, "全YF_RAD"),
    ({'CC':'YF_RAD','LB':'YF_RAD','JO':'YF_RAD','ZO':'YF_RAD','ZH':'CLC_RAD'}, "YF_RAD×4+CLC_RAD×1(ZH)"),
    ({'LB':'YF_RAD','JO':'YF_RAD','ZO':'YF_RAD','CC':'CLC_RAD','ZH':'CLC_RAD'}, "YF_RAD×3(LB/JO/ZO)+CLC_RAD×2"),
    ({'CC':'CLC_REV','LB':'CLC_REV','JO':'CLC_REV','ZO':'CLC_REV','ZH':'CLC_REV'}, "CLC_REV×5+CLC_RAD×20"),
    ({'CC':'CLC_REV','LB':'CLC_REV','JO':'CLC_REV','ZO':'CLC_REV','ZH':'CLC_RAD'}, "CLC_REV×4+CLC_RAD×21"),
    ({'LB':'CLC_REV','JO':'CLC_REV','ZO':'CLC_REV','CC':'CLC_RAD','ZH':'CLC_RAD'}, "CLC_REV×3(LB/JO/ZO)+CLC_RAD×22"),
]

print(f"{'方案':45s} | {'#':>2s} {'E(R)%':>7s} {'std%':>6s} {'Shp%':>6s} {'+ve%':>6s} {'P/L%':>6s} | {'n10':>3s} {'n15':>3s}")
print("-"*115)
for five_src, label in combos:
    md, errs, n10, n15, n = run_portfolio(five_src)
    print(f"{label:45s} | {n:2d} {errs['E(R)']:6.1f}% {errs['std(R)']:5.1f}% {errs['Sharpe']:5.1f}% {errs['% +ve']:5.1f}% {errs['Ave P/L']:5.1f}% | {n10:3d} {n15:3d}")
