import math
import numpy as np

from ml.meta_precision import wilson_lower_bound
from ml.performance import trading_metrics
from ml.quality import break_even_probability, quality_gate_from_selection
from ml.stability import evaluate_temporal_stability


TARGET_RR=2.0
CALIBRATION_WIN_RATE_TARGET=0.42
FINAL_WIN_RATE_TARGET=0.45
FINAL_EXPECTANCY_TARGET=0.20
FINAL_PF_TARGET=1.30
FINAL_WILSON_LCB_MIN=0.34


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
        min_trades=max(12,int(len(y)*.025))

    rows=[]
    break_even=break_even_probability(rr)
    for threshold in _candidate_thresholds(p,rr):
        metrics=trading_metrics(y,p,threshold=float(threshold),rr=rr)
        trades=int(metrics.get("trades",0) or 0)
        coverage=float(trades/len(y)) if len(y) else 0.0
        if trades<min_trades or coverage<min_coverage:
            continue
        wr=float(metrics.get("win_rate",0.0) or 0.0)
        wins=int(round(wr*trades))
        lcb=wilson_lower_bound(wins,trades)
        exp=float(metrics.get("expectancy_r",-999) or -999)
        pf=metrics.get("profit_factor")
        pf_value=float(pf) if pf is not None and np.isfinite(pf) else 0.0
        dd=abs(float(metrics.get("max_drawdown_r",0.0) or 0.0))
        precision_bonus=max(0.0,wr-target_win_rate)*8.0
        score=(
            4.0*lcb
            +2.0*wr
            +precision_bonus
            +0.8*exp
            +0.08*pf_value
            +0.06*math.sqrt(trades)
            -0.015*dd
        )
        rows.append({
            "threshold":float(threshold),
            "coverage":coverage,
            "score":float(score),
            "wilson_lcb_95":lcb,
            "target_win_rate":float(target_win_rate),
            "target_met":bool(wr>=target_win_rate),
            **metrics,
        })

    profitable=[
        row for row in rows
        if float(row.get("expectancy_r",-999))>0.10
        and row.get("profit_factor") is not None
        and float(row.get("profit_factor",0))>=1.15
        and float(row.get("wilson_lcb_95",0.0))>=break_even
    ]
    high_precision=[row for row in profitable if float(row.get("win_rate",0.0) or 0.0)>=target_win_rate]
    pool=high_precision if high_precision else profitable
    selected=max(pool,key=lambda x:(x["score"],x["wilson_lcb_95"],x["trades"])) if pool else None
    return selected,rows


def _group_thresholds(frame,probability,rr,column,min_fraction=.06):
    work=frame.copy()
    work["_p"]=np.asarray(probability,dtype=float)
    thresholds={}
    detail={}
    min_rows=max(60,int(len(work)*min_fraction))

    for key,group in work.groupby(column,dropna=False):
        name=str(key)
        result={"sample_rows":int(len(group)),"eligible":False,"selection":None,"selected_threshold":None}
        if len(group)>=min_rows and len(np.unique(group["label"]))>=2:
            selected,sweep=_select_threshold(
                group["label"],group["_p"],rr=rr,
                min_trades=max(10,int(len(group)*.035)),min_coverage=.03,
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
        min_trades=max(15,int(len(work)*.03)),min_coverage=.03,
    )
    regime_thresholds,regime_detail=_group_thresholds(work,probability,rr,"regime",.07)
    session_thresholds,session_detail=_group_thresholds(work,probability,rr,"session_name",.08)
    joint_thresholds,joint_detail=_group_thresholds(work,probability,rr,"joint_context",.05)

    contextual=bool(regime_thresholds or session_thresholds or joint_thresholds)
    mode="CONTEXTUAL_HIGH_WINRATE" if contextual else "GLOBAL_HIGH_WINRATE" if global_selected else "NONE"
    return {
        "mode":mode,
        "objective":"HIGH_WINRATE_RR_1_2",
        "rr":float(rr),
        "break_even_probability":break_even_probability(rr),
        "calibration_win_rate_target":CALIBRATION_WIN_RATE_TARGET,
        "selection_statistic":"WILSON_LOWER_BOUND_95",
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


def evaluate_high_winrate_stability(
    development,
    base_features,
    rr=TARGET_RR,
    horizon=80,
    folds=3,
    min_side_train=120,
    n_estimators=140,
):
    result=evaluate_temporal_stability(
        development,
        base_features,
        rr=rr,
        horizon=horizon,
        folds=folds,
        min_side_train=min_side_train,
        n_estimators=n_estimators,
    )
    fold_rows=result.get("folds") or []
    win_rates=[]
    stable_winrate_folds=0
    pooled_wins=0
    pooled_trades=0
    for row in fold_rows:
        trade=row.get("trading") or {}
        trades=int(trade.get("trades",0) or 0)
        wr=float(trade.get("win_rate",0.0) or 0.0)
        if trades>0:
            win_rates.append(wr)
            pooled_trades+=trades
            pooled_wins+=int(round(wr*trades))
        if trades>=15 and wr>=0.36 and float(trade.get("expectancy_r",-999) or -999)>0:
            stable_winrate_folds+=1

    median_wr=float(np.median(win_rates)) if win_rates else None
    pooled_lcb=wilson_lower_bound(pooled_wins,pooled_trades) if pooled_trades else 0.0
    base_pass=bool(result.get("passed",False))
    high_precision_pass=bool(
        len(fold_rows)>=3
        and stable_winrate_folds>=2
        and median_wr is not None and median_wr>=0.38
        and pooled_lcb>=break_even_probability(rr)
    )
    result["base_stability_passed"]=base_pass
    result["high_winrate_stability"]={
        "passed":high_precision_pass,
        "stable_winrate_folds":int(stable_winrate_folds),
        "median_win_rate":median_wr,
        "pooled_trades":int(pooled_trades),
        "pooled_wilson_lcb_95":float(pooled_lcb),
        "minimum_median_win_rate":0.38,
        "minimum_stable_folds":2,
    }
    result["passed"]=bool(base_pass and high_precision_pass)
    result["status"]="PASSED" if result["passed"] else "FAILED"
    return result


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
    trades=int(trading.get("trades",0) or 0)
    win_rate=float(trading.get("win_rate",0.0) or 0.0)
    wins=int(round(win_rate*trades))
    win_lcb=wilson_lower_bound(wins,trades)
    expectancy=float(trading.get("expectancy_r",-999) or -999)
    pf=trading.get("profit_factor")
    pf_value=float(pf) if pf is not None and np.isfinite(pf) else 0.0

    gate["checks"]["high_win_rate"]={
        "passed":win_rate>=FINAL_WIN_RATE_TARGET,
        "value":win_rate,
        "minimum":FINAL_WIN_RATE_TARGET,
    }
    gate["checks"]["win_rate_confidence"]={
        "passed":win_lcb>=FINAL_WILSON_LCB_MIN,
        "value":win_lcb,
        "minimum":FINAL_WILSON_LCB_MIN,
        "statistic":"Wilson lower bound 95%",
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
        "final_win_rate_wilson_lcb_min":FINAL_WILSON_LCB_MIN,
        "final_expectancy_target_r":FINAL_EXPECTANCY_TARGET,
        "final_profit_factor_target":FINAL_PF_TARGET,
    }
    return gate
