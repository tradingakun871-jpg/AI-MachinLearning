import numpy as np

from ml.performance import trading_metrics


def _score_mask(y,mask,rr,target_win_rate=0.45):
    y=np.asarray(y,dtype=int)
    mask=np.asarray(mask,dtype=bool)
    trades=int(mask.sum())
    if trades==0:
        return {"trades":0,"coverage":0.0,"score":-999.0}
    metrics=trading_metrics(y[mask],np.ones(trades),threshold=0.0,rr=rr)
    coverage=float(trades/len(y)) if len(y) else 0.0
    expectancy=float(metrics.get("expectancy_r",-999) or -999)
    win_rate=float(metrics.get("win_rate",0.0) or 0.0)
    pf=metrics.get("profit_factor")
    pf_score=float(pf) if pf is not None and np.isfinite(pf) else 0.0
    dd=abs(float(metrics.get("max_drawdown_r",0.0) or 0.0))

    # V0.12 explicitly rewards precision. Coverage still matters, but less than
    # win rate because the goal is fewer, higher-quality RR 1:2 trades.
    precision_bonus=max(0.0,win_rate-float(target_win_rate))*8.0
    score=(
        2.5*win_rate
        +precision_bonus
        +0.8*expectancy
        +0.08*pf_score
        +0.10*np.sqrt(max(trades,1))*np.sqrt(max(coverage,1e-9))
        -0.015*dd
    )
    return {"coverage":coverage,"score":float(score),**metrics}


def learn_hybrid_gate(
    y,
    supervised_mask,
    linear_mask,
    rl_mask,
    rr=2.0,
    min_trades=None,
    min_coverage=0.02,
    target_win_rate=0.45,
):
    """Choose a calibration-only confirmation policy for higher precision.

    A candidate must remain profitable, but V0.12 first prefers candidates that
    reach a high calibration win rate at RR 1:2. If none reaches the target, it
    keeps the best profitable candidate and the final quality gate still blocks
    deployment unless the independent holdout satisfies the stricter target.
    """
    y=np.asarray(y,dtype=int)
    sup=np.asarray(supervised_mask,dtype=bool)
    lin=np.asarray(linear_mask,dtype=bool)
    rl=np.asarray(rl_mask,dtype=bool)

    if min_trades is None:
        min_trades=max(10,int(len(y)*0.025))

    candidates={
        "SUPERVISED_ONLY":sup,
        "SUPERVISED_AND_LINEAR":sup & lin,
        "SUPERVISED_AND_RL":sup & rl,
        "SUPERVISED_PLUS_ONE_CONFIRM":sup & (lin | rl),
        "SUPERVISED_AND_BOTH_CONFIRM":sup & lin & rl,
    }

    rows=[]
    for mode,mask in candidates.items():
        metrics=_score_mask(y,mask,rr,target_win_rate=target_win_rate)
        metrics["mode"]=mode
        metrics["selected_rows"]=int(mask.sum())
        metrics["target_win_rate"]=float(target_win_rate)
        metrics["target_met"]=bool(float(metrics.get("win_rate",0.0) or 0.0)>=target_win_rate)
        rows.append(metrics)

    profitable=[
        row for row in rows
        if int(row.get("trades",0))>=min_trades
        and float(row.get("coverage",0.0))>=min_coverage
        and float(row.get("expectancy_r",-999))>0
        and row.get("profit_factor") is not None
        and float(row.get("profit_factor",0))>=1.15
    ]
    high_precision=[
        row for row in profitable
        if float(row.get("win_rate",0.0) or 0.0)>=float(target_win_rate)
    ]

    pool=high_precision if high_precision else profitable
    selected=max(pool,key=lambda x:(x["score"],x["win_rate"],x["trades"])) if pool else None
    return {
        "mode":"NONE" if selected is None else selected["mode"],
        "selection":selected,
        "target_win_rate":float(target_win_rate),
        "target_met_on_calibration":bool(selected and selected.get("target_met")),
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
