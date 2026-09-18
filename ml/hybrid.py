import numpy as np

from ml.performance import trading_metrics


def _score_mask(y,mask,rr):
    y=np.asarray(y,dtype=int)
    mask=np.asarray(mask,dtype=bool)
    trades=int(mask.sum())
    if trades==0:
        return {"trades":0,"coverage":0.0,"score":-999.0}
    metrics=trading_metrics(y[mask],np.ones(trades),threshold=0.0,rr=rr)
    coverage=float(trades/len(y)) if len(y) else 0.0
    expectancy=float(metrics.get("expectancy_r",-999) or -999)
    dd=abs(float(metrics.get("max_drawdown_r",0.0) or 0.0))
    score=expectancy*np.sqrt(trades)*np.sqrt(max(coverage,1e-9))-0.01*dd
    return {"coverage":coverage,"score":float(score),**metrics}


def learn_hybrid_gate(y,supervised_mask,linear_mask,rl_mask,rr=3.0,min_trades=None,min_coverage=0.03):
    y=np.asarray(y,dtype=int)
    sup=np.asarray(supervised_mask,dtype=bool)
    lin=np.asarray(linear_mask,dtype=bool)
    rl=np.asarray(rl_mask,dtype=bool)

    if min_trades is None:
        min_trades=max(10,int(len(y)*0.03))

    candidates={
        "SUPERVISED_ONLY":sup,
        "SUPERVISED_AND_LINEAR":sup & lin,
        "SUPERVISED_AND_RL":sup & rl,
        "SUPERVISED_PLUS_ONE_CONFIRM":sup & (lin | rl),
        "SUPERVISED_AND_BOTH_CONFIRM":sup & lin & rl,
    }

    rows=[]
    for mode,mask in candidates.items():
        metrics=_score_mask(y,mask,rr)
        metrics["mode"]=mode
        metrics["selected_rows"]=int(mask.sum())
        rows.append(metrics)

    viable=[
        row for row in rows
        if int(row.get("trades",0))>=min_trades
        and float(row.get("coverage",0.0))>=min_coverage
        and float(row.get("expectancy_r",-999))>0
        and row.get("profit_factor") is not None
        and float(row.get("profit_factor",0))>=1.05
    ]
    selected=max(viable,key=lambda x:(x["score"],x["trades"])) if viable else None
    return {
        "mode":"NONE" if selected is None else selected["mode"],
        "selection":selected,
        "candidates":rows,
    }


def apply_hybrid_gate(mode,supervised_mask,linear_mask,rl_mask):
    sup=np.asarray(supervised_mask,dtype=bool)
    lin=np.asarray(linear_mask,dtype=bool)
    rl=np.asarray(rl_mask,dtype=bool)

    if mode=="SUPERVISED_ONLY":
        return sup
    if mode=="SUPERVISED_AND_LINEAR":
        return sup & lin
    if mode=="SUPERVISED_AND_RL":
        return sup & rl
    if mode=="SUPERVISED_PLUS_ONE_CONFIRM":
        return sup & (lin | rl)
    if mode=="SUPERVISED_AND_BOTH_CONFIRM":
        return sup & lin & rl
    return np.zeros(len(sup),dtype=bool)
