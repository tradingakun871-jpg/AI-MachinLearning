import argparse
from pathlib import Path
import pandas as pd
from data.feature_builder import add_market_features, add_structure_proxy
from data.multitimeframe import attach_htf_context
from ml.labeling import triple_barrier_label


def load_csv(path):
    d = pd.read_csv(path)
    d["timestamp"] = pd.to_datetime(d.timestamp, utc=True)
    return d.sort_values("timestamp").reset_index(drop=True)


def build(m3, m5, m15, rr=3.0, horizon=80):
    d = add_structure_proxy(add_market_features(m3))
    d = attach_htf_context(d, m5, m15)
    labels=[]
    for i,row in d.iterrows():
        side = "BUY" if row.mss > 0 else "SELL" if row.mss < 0 else None
        if not side or pd.isna(row.atr): labels.append(None); continue
        future=d.iloc[i+1:i+1+horizon]
        labels.append(triple_barrier_label(future.high, future.low, row.close, side, row.atr, rr) if len(future) else None)
    d["label"] = labels
    d["candidate"] = d.mss.ne(0).astype(int)
    return d[d.candidate.eq(1)].dropna(subset=["label"]).reset_index(drop=True)

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--m3",required=True); p.add_argument("--m5",required=True); p.add_argument("--m15",required=True); p.add_argument("--output",required=True); p.add_argument("--rr",type=float,default=3.0); p.add_argument("--horizon",type=int,default=80)
    a=p.parse_args(); out=build(load_csv(a.m3),load_csv(a.m5),load_csv(a.m15),a.rr,a.horizon); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print({"rows":len(out),"output":a.output})
