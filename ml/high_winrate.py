import math
import numpy as np

from ml.performance import trading_metrics
from ml.quality import break_even_probability, quality_gate_from_selection


TARGET_RR=2.0
CALIBRATION_WIN_RATE_TARGET=0.42
FINAL_WIN_RATE_TARGET=0.45
FINAL_EXPECTANCY_TARGET=0.20
FINAL_PF_TARGET=1.30


def _candidate_thresholds(p,rr=TARGET_RR):
    p=np.asarray(p,dtype=float)
    p=p[np.isfinite(p)]
    if len(p)==0:
        return []
    floor=max(break_even_probability(rr)+0.02,0.35)
    quantiles=np.quantile(p,[.35,.45,.55,.65,.72,.78,.84,.89,.93,.96,.98,.99])
    upper=min(.95,max(float(np.max(p)),floor))
    grid=np.arange(floor,upper+1e-9,.005)
    values=np.r_[grid,quantiles,[floor]]
    return sorted({round(float(x),4) for x in values if np.isfinite(x) and x>=floor and x<=.95})


def _select_threshold(y,p,rr=TARGET_RR,min_trades=None,min_coverage=.02,target_win_rate=CALIBRATION_WIN_RATE_TARGET):
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    if min_trades is None:
        min_trades=max(10,int(len(y)*.025))

    rows=[]
    for threshold in _candidate_thresholds(p,rr):
        metrics=trading_metrics(y,p,threshold=float(threshold),rr=rr)
        trades=int(metrics.get("trades",0) or 0)
        coverage=float(trades/len(y)) if len(y) else 0.0
        if trades<min_trades or coverage<min_coverage:
            continue
        wr=float(metrics.get("win_rate",0.0) or 0.0)
        exp=float(metrics.get("expectancy_r",-999) or -999)
        pf=metrics.get("profit_factor")
        pf_value=float(pf) if pf is not None and np.isfinite(pf) else 0.0
        dd=abs(float(metrics.get("max_drawdown_r",0.0) or 0.0))
        precision_bonus=max(0.0,wr-target_win_rate)*10.0
        score=(3.0*wr+precision_bonus+0.8*exp+0.08*pf_value+0.08*math.sqrt(trades)-0.015*dd)
        rows.append({
            "threshold":float(threshold),
            "coverage":coverage,
            "score":float(score),
            "target_win_rate":float(target_win_rate),
            "target_met":bool(wr>=target_win_rate),
            **metrics,
        })

    profitable=[
        row for row in rows
        if float(row.get("expectancy_r",-999))>0.10
        and row.get("profit_factor") is not None
        and float(row.get("profit_factor",0))>=1.15
    ]
    high_precision=[row for row in profitable if float(row.get("win_rate",0.0) or 0.0)>=target_win_rate]
    pool=high_precision if high_precision else profitable
    selected=max(pool,key=lambda x:(x["score"],x["win_rate"],x["trades"])) if pool else None
    return selected,rows


def _group_thresholds(frame,probability,rr,column,min_fraction=.06):
    work=frame.copy()
    work["_p"]=np.asarray(probability,dtype=float)
    thresholds={}
    detail={}
    min_rows=max(50,int(len(work)*min_fraction))

    for key,group in work.groupby(column,dropna=False):
        name=str(key)
        result={"sample_rows":int(len(group)),"eligible":False,"selection":None,"selected_threshold":None}
        if len(group)>=min_rows and len(np.unique(group["label"]))>=2:
            selected,sweep=_select_threshold(
                group["label"],group["_p"],rr=rr,
                min_trades=max(8,int(len(group)*.03)),min_coverage=.025,
            )
            result["sweep_count"]=len(sweep)
            result["selection"]=selected
            if selected is not None:
                thresholds[name]=float(selected["threshold"])
                result["eligible"]=True
                result["selected_threshold"]=float(selected["threshold"])
        detail[name]=result
    return thresholds,detail


def learn_high_winrate_policy(frame,probability,rr=TARGET_RR):
    work=frame.copy()
    if "session_name" not in work:
        work["session_name"]="unknown"
    work["joint_context"]=work["regime"].astype(str)+"|"+work["session_name"].astype(str)

    global_selected,global_sweep=_select_threshold(
        work["label"],probability,rr=rr,
        min_trades=max(12,int(len(work)*.025)),min_coverage=.025,
    )
    regime_thresholds,regime_detail=_group_thresholds(work,probability,rr,"regime",.07)
    session_thresholds,session_detail=_group_thresholds(work,probability,rr,"session_name",.08)
    joint_thresholds,joint_detail=_group_thresholds(work,probability,rr,"joint_context",.045)

    contextual=bool(regime_thresholds or session_thresholds or joint_thresholds)
    mode="CONTEXTUAL_HIGH_WINRATE" if contextual else "GLOBAL_HIGH_WINRATE" if global_selected else "NONE"
    return {
        "mode":mode,
        "objective":"HIGH_WINRATE_RR_1_2",
        "rr":float(rr),
        "break_even_probability":break_even_probability(rr),
        "calibration_win_rate_target":CALIBRATION_WIN_RATE_TARGET,
        "global_threshold":None if global_selected is None else float(global_selected["threshold"]),
        "global_selection":global_selected,
        "global_sweep_count":len(global_sweep),
        "regime_thresholds":regime_thresholds,
        "session_thresholds":session_thresholds,
        "joint_thresholds":joint_thresholds,
        "allowed_regimes":sorted(regime_thresholds.keys()),
        "allowed_sessions":sorted(session_thresholds.keys()),
        "allowed_joint_contexts":sorted(joint_thresholds.keys()),
        "regime_detail":regime_detail,
        "session_detail":session_detail,
        "joint_detail":joint_detail,
    }


def high_winrate_quality_gate(
    test_frame,
    test_probability,
    selected_mask,
    applied_thresholds,
    rr,
    policy_status="HIGH_WINRATE_RR_1_2",
    temporal_stability=None,
):
    gate=quality_gate_from_selection(
        test_frame,
        test_probability,
        selected_mask,
        applied_thresholds,
        rr,
        policy_status="HIGH_WINRATE_RR_1_2",
        temporal_stability=temporal_stability,
    )
    trading=gate.get("trading") or {}
    win_rate=float(trading.get("win_rate",0.0) or 0.0)
    expectancy=float(trading.get("expectancy_r",-999) or -999)
    pf=trading.get("profit_factor")
    pf_value=float(pf) if pf is not None and np.isfinite(pf) else 0.0

    gate["checks"]["high_win_rate"]={
        "passed":win_rate>=FINAL_WIN_RATE_TARGET,
        "value":win_rate,
        "minimum":FINAL_WIN_RATE_TARGET,
    }
    gate["checks"]["high_expectancy"]={
        "passed":expectancy>=FINAL_EXPECTANCY_TARGET,
        "value":expectancy,
        "minimum":FINAL_EXPECTANCY_TARGET,
    }
    gate["checks"]["high_profit_factor"]={
        "passed":pf_value>=FINAL_PF_TARGET,
        "value":pf,
        "minimum":FINAL_PF_TARGET,
    }
    gate["checks"]["rr_target"]={
        "passed":abs(float(rr)-2.0)<1e-9,
        "value":float(rr),
        "required":2.0,
    }
    gate["passed"]=all(item.get("passed",False) for item in gate["checks"].values())
    gate["status"]="PASSED" if gate["passed"] else "FAILED"
    gate["objective"]={
        "name":"HIGH_WINRATE_RR_1_2",
        "rr":2.0,
        "final_win_rate_target":FINAL_WIN_RATE_TARGET,
        "final_expectancy_target_r":FINAL_EXPECTANCY_TARGET,
        "final_profit_factor_target":FINAL_PF_TARGET,
    }
    return gate
