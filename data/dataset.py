import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from data.feature_builder import add_market_features
from data.multitimeframe import attach_htf_context
from smc.engine import add_smc_features
from ml.labeling import triple_barrier_label


def load_csv(path):
    d=pd.read_csv(path)
    d["timestamp"]=pd.to_datetime(d.timestamp,utc=True)
    return d.sort_values("timestamp").reset_index(drop=True)


def _direction(row):
    for key in ("mss","bos","liquidity_sweep_direction"):
        value=int(row.get(key,0) or 0)
        if value != 0:
            return 1 if value > 0 else -1
    value=int(row.get("structure_bias",0) or 0)
    return 1 if value > 0 else -1 if value < 0 else 0


def build(m3,m5,m15,rr=3.0,horizon=80):
    d=add_smc_features(add_market_features(m3))
    d=attach_htf_context(d,m5,m15)
    d["trend_m3"]=d["structure_bias"]
    d["source_index"]=np.arange(len(d),dtype=int)

    labels=[]
    directions=[]
    candidates=[]

    for i,row in d.iterrows():
        direction=_direction(row)
        directions.append(direction)

        event=(
            int(row.get("mss",0) or 0)!=0
            or int(row.get("bos",0) or 0)!=0
            or int(row.get("liquidity_sweep",0) or 0)!=0
            or int(row.get("order_block",0) or 0)!=0
            or int(row.get("fvg",0) or 0)!=0
        )
        candidate=bool(direction and event)
        candidates.append(int(candidate))

        if not candidate or pd.isna(row.get("atr")) or float(row.get("atr",0) or 0)<=0:
            labels.append(None)
            continue

        future=d.iloc[i+1:i+1+horizon]
        if len(future)==0:
            labels.append(None)
            continue

        side="BUY" if direction>0 else "SELL"
        labels.append(
            triple_barrier_label(
                future.high,
                future.low,
                float(row.close),
                side,
                float(row.atr),
                rr,
            )
        )

    d["signal_direction"]=directions
    d["label"]=labels
    d["candidate"]=candidates

    required=["label","trend_m5","trend_m15","atr_norm","return_3","range_atr"]
    return d[d.candidate.eq(1)].dropna(subset=required).reset_index(drop=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--m3",required=True)
    p.add_argument("--m5",required=True)
    p.add_argument("--m15",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--rr",type=float,default=3.0)
    p.add_argument("--horizon",type=int,default=80)
    a=p.parse_args()
    out=build(load_csv(a.m3),load_csv(a.m5),load_csv(a.m15),a.rr,a.horizon)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(a.output,index=False)
    print({"rows":len(out),"output":a.output,"engine":"SMC-v0.9"})
