import numpy as np
import pandas as pd


def pivots(df: pd.DataFrame, left=3, right=3):
    """Confirmed pivots. A pivot becomes usable only `right` bars after it forms."""
    d=df.copy(); n=len(d); ph=np.full(n,np.nan); pl=np.full(n,np.nan)
    h=d.high.to_numpy(); l=d.low.to_numpy()
    for i in range(left,n-right):
        if h[i] == np.max(h[i-left:i+right+1]): ph[i+right]=h[i]
        if l[i] == np.min(l[i-left:i+right+1]): pl[i+right]=l[i]
    return pd.Series(ph,index=d.index).ffill(), pd.Series(pl,index=d.index).ffill()


def market_structure(df: pd.DataFrame, left=3, right=3):
    d=df.copy(); last_hi,last_lo=pivots(d,left,right)
    d["swing_high"]=last_hi; d["swing_low"]=last_lo
    prev_hi=last_hi.shift(1); prev_lo=last_lo.shift(1)
    bull_break=(d.close>prev_hi)&(d.close.shift(1)<=prev_hi.shift(1))
    bear_break=(d.close<prev_lo)&(d.close.shift(1)>=prev_lo.shift(1))
    state=0; bos=[]; choch=[]
    for bup,bdn in zip(bull_break.fillna(False),bear_break.fillna(False)):
        b=c=0
        if bup: c=1 if state<0 else 0; b=1 if state>=0 else 0; state=1
        elif bdn: c=-1 if state>0 else 0; b=-1 if state<=0 else 0; state=-1
        bos.append(b); choch.append(c)
    d["bos"]=bos; d["choch"]=choch; d["structure_bias"]=pd.Series(np.where(d.choch.ne(0),d.choch,np.where(d.bos.ne(0),d.bos,np.nan)),index=d.index).ffill().fillna(0)
    return d
