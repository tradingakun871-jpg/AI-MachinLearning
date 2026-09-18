import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from data.feature_builder import add_market_features
from data.multitimeframe import attach_htf_context
from smc.engine import add_smc_features
from ml.labeling import barrier_outcome
from ml.setup_features import add_session_name, add_setup_features, structural_stop_distance


def load_csv(path):
    d=pd.read_csv(path)
    d["timestamp"]=pd.to_datetime(d.timestamp,utc=True)
    return d.sort_values("timestamp").reset_index(drop=True)


def build(m3,m5,m15,rr=3.0,horizon=80):
    d=add_smc_features(add_market_features(m3))
    d=attach_htf_context(d,m5,m15)
    d["trend_m3"]=d["structure_bias"]
    d=add_setup_features(d)
    d=add_session_name(d)
    d["source_index"]=np.arange(len(d),dtype=int)

    labels=[]; entries=[]; stops=[]; tps=[]; bars_to_outcome=[]; reasons=[]
    stop_atr=[]

    for i,row in d.iterrows():
        if int(row.get("candidate",0) or 0)!=1:
            labels.append(None); entries.append(np.nan); stops.append(np.nan); tps.append(np.nan)
            bars_to_outcome.append(np.nan); reasons.append("NOT_CANDIDATE"); stop_atr.append(np.nan)
            continue
        if i+1>=len(d):
            labels.append(None); entries.append(np.nan); stops.append(np.nan); tps.append(np.nan)
            bars_to_outcome.append(np.nan); reasons.append("NO_NEXT_BAR"); stop_atr.append(np.nan)
            continue

        entry=float(d.iloc[i+1]["open"])
        distance=structural_stop_distance(row,entry)
        if distance is None:
            labels.append(None); entries.append(entry); stops.append(np.nan); tps.append(np.nan)
            bars_to_outcome.append(np.nan); reasons.append("INVALID_STOP"); stop_atr.append(np.nan)
            continue

        future=d.iloc[i+1:i+1+horizon]
        side="BUY" if int(row.signal_direction)>0 else "SELL"
        outcome=barrier_outcome(
            future.high,future.low,entry,side,distance,rr
        )

        labels.append(outcome["label"])
        entries.append(outcome.get("entry",entry))
        stops.append(outcome.get("sl",np.nan))
        tps.append(outcome.get("tp",np.nan))
        bars_to_outcome.append(outcome.get("bars_to_outcome",np.nan))
        reasons.append(outcome.get("reason","UNKNOWN"))
        atr=float(row.get("atr",np.nan))
        planned=structural_stop_distance(row,float(row.close))
        stop_atr.append(float(planned/atr) if planned is not None and np.isfinite(atr) and atr>0 else np.nan)

    d["label"]=labels
    d["entry_price"]=entries
    d["sl_price"]=stops
    d["tp_price"]=tps
    d["bars_to_outcome"]=bars_to_outcome
    d["label_reason"]=reasons
    d["stop_distance_atr"]=stop_atr

    required=[
        "label","trend_m5","trend_m15","atr_norm","return_3","range_atr",
        "smc_confluence","htf_alignment","stop_distance_atr"
    ]
    return (
        d[d.candidate.eq(1)]
        .dropna(subset=required)
        .reset_index(drop=True)
    )


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
    print({"rows":len(out),"output":a.output,"engine":"SMC-v0.11"})
