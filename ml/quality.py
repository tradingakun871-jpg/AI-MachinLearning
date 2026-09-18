import math
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml.performance import probability_metrics, trading_metrics


def break_even_probability(rr=3.0):
    rr=float(rr)
    if rr<=0:
        raise ValueError("rr must be positive")
    return 1.0/(1.0+rr)


def probability_diagnostics(y,p,frame=None):
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    finite=np.isfinite(p)
    y=y[finite]; p=p[finite]
    out={"rows":int(len(p))}
    if not len(p):
        return out

    qs=[0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99]
    out["quantiles"]={str(q):float(np.quantile(p,q)) for q in qs}
    out["mean"]=float(np.mean(p))
    out["std"]=float(np.std(p))
    for label,name in ((1,"wins"),(0,"losses")):
        vals=p[y==label]
        out[name]={
            "rows":int(len(vals)),
            "mean":float(np.mean(vals)) if len(vals) else None,
            "median":float(np.median(vals)) if len(vals) else None,
            "p90":float(np.quantile(vals,.90)) if len(vals) else None,
        }
    if out["wins"]["mean"] is not None and out["losses"]["mean"] is not None:
        out["mean_probability_gap"]=float(out["wins"]["mean"]-out["losses"]["mean"])

    if frame is not None and "signal_direction" in frame:
        directions=np.asarray(frame.loc[finite,"signal_direction"],dtype=int)
        out["by_side"]={}
        for direction,name in ((1,"BUY"),(-1,"SELL")):
            mask=directions==direction
            vals=p[mask]
            labs=y[mask]
            out["by_side"][name]={
                "rows":int(len(vals)),
                "base_rate":float(np.mean(labs)) if len(vals) else None,
                "mean_probability":float(np.mean(vals)) if len(vals) else None,
                "p90":float(np.quantile(vals,.90)) if len(vals) else None,
            }
    return out


def select_features_by_importance(importance,frame,min_features=14,max_features=28,cumulative=0.90):
    """Training-only feature pruning using native ensemble importance plus variance checks."""
    usable=[]
    for item in importance:
        name=item["feature"]
        if name not in frame.columns:
            continue
        series=pd.to_numeric(frame[name],errors="coerce")
        if series.nunique(dropna=True)<=1:
            continue
        usable.append(item)

    if not usable:
        return []

    chosen=[]
    total=sum(max(float(x.get("importance",0.0)),0.0) for x in usable) or 1.0
    running=0.0
    for item in usable:
        chosen.append(item["feature"])
        running+=max(float(item.get("importance",0.0)),0.0)/total
        if len(chosen)>=min_features and running>=cumulative:
            break
        if len(chosen)>=max_features:
            break

    return chosen[:max_features]


def _threshold_candidates(p,rr=3.0,margin=0.005):
    p=np.asarray(p,dtype=float)
    p=p[np.isfinite(p)]
    if len(p)==0:
        return []

    floor=min(0.95,max(0.01,break_even_probability(rr)+float(margin)))
    quantiles=np.quantile(p,[.20,.30,.40,.50,.60,.70,.78,.84,.89,.93,.96,.98,.99])
    upper=min(0.95,max(float(np.max(p)),floor))
    grid=np.arange(floor,upper+1e-9,0.0025)
    vals=np.r_[grid,quantiles,[floor]]
    vals=np.clip(vals,floor,0.95)
    return sorted({round(float(x),4) for x in vals if np.isfinite(x)})


def threshold_sweep_adaptive(y,p,rr=3.0,min_trades=None,min_coverage=0.025,margin=0.005):
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    if min_trades is None:
        min_trades=max(12,int(len(y)*0.025))

    rows=[]
    for t in _threshold_candidates(p,rr=rr,margin=margin):
        m=trading_metrics(y,p,float(t),rr)
        trades=int(m.get("trades",0) or 0)
        coverage=float(trades/len(y)) if len(y) else 0.0
        if trades<int(min_trades) or coverage<float(min_coverage):
            continue

        dd=abs(float(m.get("max_drawdown_r",0.0) or 0.0))
        expectancy=float(m.get("expectancy_r",-999) or -999)
        pf=m.get("profit_factor")
        pf_score=float(pf) if pf is not None and np.isfinite(pf) else 0.0
        score=expectancy*math.sqrt(trades)*math.sqrt(max(coverage,1e-9))+0.05*pf_score-0.01*dd

        rows.append({
            "threshold":float(t),
            "coverage":coverage,
            "score":float(score),
            "break_even_probability":break_even_probability(rr),
            **m,
        })
    return rows


def select_threshold(y,p,rr=3.0,min_trades=None,min_coverage=0.025):
    rows=threshold_sweep_adaptive(
        y,p,rr=rr,min_trades=min_trades,min_coverage=min_coverage
    )
    viable=[
        x for x in rows
        if float(x.get("expectancy_r",-999))>0
        and float(x.get("net_r",-999))>0
        and x.get("profit_factor") is not None
        and float(x.get("profit_factor",0))>=1.05
    ]
    selected=max(viable,key=lambda x:(x["score"],x["trades"],x["coverage"])) if viable else None
    return selected,rows


def _learn_group_thresholds(frame,probability,rr,group_col,min_rows_fraction=0.06):
    work=frame.copy()
    work["_p"]=np.asarray(probability,dtype=float)
    detail={}
    thresholds={}

    min_rows=max(50,int(len(work)*min_rows_fraction))
    for key,g in work.groupby(group_col,dropna=False):
        name=str(key)
        sample_rows=int(len(g))
        result={
            "sample_rows":sample_rows,
            "eligible":False,
            "selected_threshold":None,
            "selection":None,
        }
        if sample_rows>=min_rows and len(np.unique(g["label"]))>=2:
            min_trades=max(8,int(sample_rows*0.035))
            selected,sweep=select_threshold(
                g["label"],g["_p"],rr=rr,
                min_trades=min_trades,min_coverage=0.035,
            )
            result["sweep_count"]=len(sweep)
            result["selection"]=selected
            if selected is not None:
                threshold=float(selected["threshold"])
                thresholds[name]=threshold
                result["eligible"]=True
                result["selected_threshold"]=threshold
        detail[name]=result
    return thresholds,detail


def learn_threshold_policy(frame,probability,rr=3.0):
    """Calibration-only threshold learning: global + regime + session + regime×session."""
    work=frame.copy()
    if "session_name" not in work:
        work["session_name"]="unknown"
    work["joint_context"]=work["regime"].astype(str)+"|"+work["session_name"].astype(str)

    global_min=max(15,int(len(work)*0.03))
    global_selected,global_sweep=select_threshold(
        work["label"],probability,rr=rr,
        min_trades=global_min,min_coverage=0.03,
    )

    regime_thresholds,regime_detail=_learn_group_thresholds(
        work,probability,rr,"regime",min_rows_fraction=0.07
    )
    session_thresholds,session_detail=_learn_group_thresholds(
        work,probability,rr,"session_name",min_rows_fraction=0.08
    )
    joint_thresholds,joint_detail=_learn_group_thresholds(
        work,probability,rr,"joint_context",min_rows_fraction=0.045
    )

    any_policy=bool(global_selected or regime_thresholds or session_thresholds or joint_thresholds)
    contextual=bool(regime_thresholds or session_thresholds or joint_thresholds)
    mode="CONTEXTUAL" if contextual else "GLOBAL" if global_selected else "NONE"

    return {
        "mode":mode,
        "rr":float(rr),
        "break_even_probability":break_even_probability(rr),
        "margin_above_break_even":0.005,
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
        "has_any_policy":any_policy,
    }


def threshold_for_context(regime,session,policy):
    """Use the most specific learned threshold; fall back to global instead of hard intersections."""
    if not policy or policy.get("mode")=="NONE":
        return None

    regime=str(regime)
    session=str(session)
    joint_key=f"{regime}|{session}"

    joint=policy.get("joint_thresholds") or {}
    if joint_key in joint:
        return float(joint[joint_key])

    contextual=[]
    rmap=policy.get("regime_thresholds") or {}
    smap=policy.get("session_thresholds") or {}
    if regime in rmap:
        contextual.append(float(rmap[regime]))
    if session in smap:
        contextual.append(float(smap[session]))

    global_t=policy.get("global_threshold")
    if contextual:
        if global_t is not None:
            contextual.append(float(global_t))
        return max(contextual)

    return None if global_t is None else float(global_t)


def apply_threshold_policy(frame,probability,policy):
    p=np.asarray(probability,dtype=float)
    take=np.zeros(len(frame),dtype=bool)
    thresholds=np.full(len(frame),np.nan,dtype=float)
    sessions=frame["session_name"].astype(str) if "session_name" in frame else pd.Series(["unknown"]*len(frame),index=frame.index)
    regimes=frame["regime"].astype(str)

    for i,(regime,session,prob) in enumerate(zip(regimes,sessions,p)):
        t=threshold_for_context(regime,session,policy)
        if t is None:
            continue
        thresholds[i]=t
        take[i]=bool(prob>=t)
    return take,thresholds


def quality_gate_from_selection(
    test_frame,
    test_probability,
    selected_mask,
    applied_thresholds,
    rr,
    policy_status="CUSTOM",
    temporal_stability=None,
):
    y=np.asarray(test_frame["label"],dtype=int)
    p=np.asarray(test_probability,dtype=float)
    selected_mask=np.asarray(selected_mask,dtype=bool)
    applied_thresholds=np.asarray(applied_thresholds,dtype=float)

    prob=probability_metrics(y,p)
    prob["pr_auc"]=float(average_precision_score(y,p)) if len(np.unique(y))>1 else None

    filtered_y=y[selected_mask]
    filtered_p=p[selected_mask]
    trade=trading_metrics(filtered_y,filtered_p,threshold=0.0,rr=rr) if len(filtered_y) else {"trades":0}
    trades=int(trade.get("trades",0) or 0)
    coverage=float(trades/len(y)) if len(y) else 0.0

    base_rate=float(np.mean(y)) if len(y) else 0.0
    base_brier=float(base_rate*(1-base_rate))
    auc=prob.get("auc")
    pr_auc=prob.get("pr_auc")
    pf=trade.get("profit_factor")
    expectancy=float(trade.get("expectancy_r",-999) or -999)
    max_dd=float(trade.get("max_drawdown_r",0.0) or 0.0)

    stability_passed=bool(
        temporal_stability
        and temporal_stability.get("passed",False)
    )

    checks={
        "test_rows":{"passed":len(y)>=500,"value":int(len(y)),"minimum":500},
        "test_trades":{"passed":trades>=30,"value":trades,"minimum":30},
        "coverage":{"passed":coverage>=0.03,"value":coverage,"minimum":0.03},
        "auc":{"passed":auc is not None and float(auc)>=0.52,"value":auc,"minimum":0.52},
        "pr_auc_vs_base":{
            "passed":pr_auc is not None and float(pr_auc)>=base_rate*1.02,
            "value":pr_auc,"minimum":base_rate*1.02,
        },
        "brier_vs_base":{
            "passed":prob.get("brier") is not None and float(prob["brier"])<=base_brier*0.99,
            "value":prob.get("brier"),"maximum":base_brier*0.99,
        },
        "expectancy_r":{"passed":expectancy>0,"value":expectancy,"minimum_exclusive":0.0},
        "profit_factor":{
            "passed":pf is not None and float(pf)>=1.10,
            "value":pf,"minimum":1.10,
        },
        "drawdown_r":{"passed":max_dd>=-25.0,"value":max_dd,"minimum":-25.0},
        "calibration_policy":{
            "passed":policy_status not in (None,"NONE"),
            "value":policy_status,
        },
        "temporal_stability":{
            "passed":stability_passed,
            "value":None if not temporal_stability else temporal_stability.get("status"),
        },
    }

    passed=all(item["passed"] for item in checks.values())
    finite_thresholds=applied_thresholds[np.isfinite(applied_thresholds)]

    return {
        "passed":bool(passed),
        "status":"PASSED" if passed else "FAILED",
        "checks":checks,
        "probability":prob,
        "probability_diagnostics":probability_diagnostics(y,p,test_frame),
        "trading":trade,
        "coverage":coverage,
        "base_rate":base_rate,
        "filtered_test_rows":trades,
        "threshold_policy_mode":policy_status,
        "temporal_stability":temporal_stability,
        "applied_threshold_min":float(np.min(finite_thresholds)) if len(finite_thresholds) else None,
        "applied_threshold_max":float(np.max(finite_thresholds)) if len(finite_thresholds) else None,
    }


def quality_gate(test_frame,test_probability,rr,policy,temporal_stability=None):
    mask,thresholds=apply_threshold_policy(test_frame,test_probability,policy)
    return quality_gate_from_selection(
        test_frame,
        test_probability,
        mask,
        thresholds,
        rr,
        policy_status=None if not policy else policy.get("mode"),
        temporal_stability=temporal_stability,
    )
