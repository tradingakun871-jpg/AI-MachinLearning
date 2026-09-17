import numpy as np
import pandas as pd


def order_blocks(df: pd.DataFrame, impulse_atr=1.2, search_back=6):
    """Causal OB candidate: last opposite candle before a displacement/structure break."""
    d=df.copy(); body=(d.close-d.open).abs(); atr=d.get("atr",(d.high-d.low).rolling(14).mean()); displacement=body>atr*impulse_atr
    direction=np.where((displacement)&(d.close>d.open),1,np.where((displacement)&(d.close<d.open),-1,0)); ob_dir=np.zeros(len(d),dtype=int); ob_low=np.full(len(d),np.nan); ob_high=np.full(len(d),np.nan)
    for i,side in enumerate(direction):
        if side==0: continue
        for j in range(i-1,max(-1,i-search_back-1),-1):
            opposite=(side==1 and d.close.iloc[j]<d.open.iloc[j]) or (side==-1 and d.close.iloc[j]>d.open.iloc[j])
            if opposite: ob_dir[i]=side; ob_low[i]=d.low.iloc[j]; ob_high[i]=d.high.iloc[j]; break
    d["ob_direction"]=ob_dir; d["ob_low"]=ob_low; d["ob_high"]=ob_high; return d
