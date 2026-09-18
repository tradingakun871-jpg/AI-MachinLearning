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


def _threshold_candidates(p,rr=3.0,margin=0.01):
    """Adaptive probability grid anchored to RR break-even and observed calibration probabilities."""
    p=np.asarray(p,dtype=float)
    p=p[np.isfinite(p)]
    if len(p)==0:
        return []
    floor=min(0.95,max(0.01,break_even_probability(rr)+float(margin)))
    quantiles=np.quantile(p,[.35,.45,.55,.65,.75,.82,.88,.92,.95,.97,.99])
    upper=min(0.95,max(float(np.max(p)),floor))
    grid=np.arange(floor,upper+1e-9,0.005)
    vals=np.r_[grid,quantiles,[floor]]
    vals=np.clip(vals,floor,0.95)
    return sorted({round(float(x),4) for x in vals if np.isfinite(x)})


def threshold_sweep_adaptive(y,p,rr=3.0,min_trades=None,min_coverage=0.03,margin=0.01):
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    if min_trades is None:
        min_trades=max(15,int(len(y)*0.03))
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
        # Reward stable expectancy and sample size; penalize deep drawdown.
        score=expectancy*math.sqrt(trades)*math.sqrt(max(coverage,1e-9)) + 0.05*pf_score - 0.01*dd
        rows.append({
            "threshold":float(t),
            "coverage":coverage,
            "score":float(score),
            "break_even_probability":break_even_probability(rr),
            **m,
        })
    return rows


def select_threshold(y,p,rr=3.0,min_trades=None,min_coverage=0.03):
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


def _learn_group_thresholds(frame,probability,rr,group_col):
    work=frame.copy()
    work["_p"]=np.asarray(probability,dtype=float)
    detail={}
    thresholds={}
    allowed=[]

    min_rows=max(70,int(len(work)*0.08))
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
            min_trades=max(10,int(sample_rows*0.04))
            selected,sweep=select_threshold(
                g["label"],g["_p"],rr=rr,
                min_trades=min_trades,min_coverage=0.04,
            )
            result["sweep_count"]=len(sweep)
            result["selection"]=selected
            if selected is not None:
                threshold=float(selected["threshold"])
                thresholds[name]=threshold
                allowed.append(name)
                result["eligible"]=True
                result["selected_threshold"]=threshold
        detail[name]=result
    return thresholds,allowed,detail


def learn_threshold_policy(frame,probability,rr=3.0):
    """Learn global and per-regime/session thresholds using calibration data only."""
    work=frame.copy()
    if "session_name" not in work:
        work["session_name"]="unknown"

    global_min=max(20,int(len(work)*0.04))
    global_selected,global_sweep=select_threshold(
        work["label"],probability,rr=rr,
        min_trades=global_min,min_coverage=0.04,
    )

    regime_thresholds,allowed_regimes,regime_detail=_learn_group_thresholds(
        work,probability,rr,"regime"
    )
    session_thresholds,allowed_sessions,session_detail=_learn_group_thresholds(
        work,probability,rr,"session_name"
    )

    contextual=bool(regime_thresholds or session_thresholds)
    global_ok=global_selected is not None
    mode="CONTEXTUAL" if contextual else "GLOBAL" if global_ok else "NONE"

    return {
        "mode":mode,
        "rr":float(rr),
        "break_even_probability":break_even_probability(rr),
        "margin_above_break_even":0.01,
        "global_threshold":None if global_selected is None else float(global_selected["threshold"]),
        "global_selection":global_selected,
        "global_sweep_count":len(global_sweep),
        "regime_thresholds":regime_thresholds,
        "session_thresholds":session_thresholds,
        "allowed_regimes":allowed_regimes,
        "allowed_sessions":allowed_sessions,
        "regime_detail":regime_detail,
        "session_detail":session_detail,
    }


def threshold_for_context(regime,session,policy):
    """Return the threshold learned for this live/test context, or None when the context is blocked."""
    if not policy or policy.get("mode")=="NONE":
        return None

    candidates=[]
    global_t=policy.get("global_threshold")
    if global_t is not None:
        candidates.append(float(global_t))

    regime_map=policy.get("regime_thresholds") or {}
    session_map=policy.get("session_thresholds") or {}

    if regime_map:
        key=str(regime)
        if key not in regime_map:
            return None
        candidates.append(float(regime_map[key]))

    if session_map:
        key=str(session)
        if key not in session_map:
            return None
        candidates.append(float(session_map[key]))

    if not candidates:
        return None
    # Conservative composition when more than one calibration policy applies.
    return max(candidates)


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


def quality_gate(test_frame,test_probability,rr,policy):
    y=np.asarray(test_frame["label"],dtype=int)
    p=np.asarray(test_probability,dtype=float)
    prob=probability_metrics(y,p)
    prob["pr_auc"]=float(average_precision_score(y,p)) if len(np.unique(y))>1 else None

    selected_mask,applied_thresholds=apply_threshold_policy(test_frame,p,policy)
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

    # V0.11.1 keeps the independent final-test gate strict. Adaptive threshold
    # learning changes calibration policy, not pass criteria.
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
            "passed":bool(policy and policy.get("mode")!="NONE"),
            "value":None if not policy else policy.get("mode"),
        },
    }
    passed=all(item["passed"] for item in checks.values())
    finite_thresholds=applied_thresholds[np.isfinite(applied_thresholds)]

    return {
        "passed":bool(passed),
        "status":"PASSED" if passed else "FAILED",
        "checks":checks,
        "probability":prob,
        "trading":trade,
        "coverage":coverage,
        "base_rate":base_rate,
        "filtered_test_rows":trades,
        "threshold_policy_mode":None if not policy else policy.get("mode"),
        "applied_threshold_min":float(np.min(finite_thresholds)) if len(finite_thresholds) else None,
        "applied_threshold_max":float(np.max(finite_thresholds)) if len(finite_thresholds) else None,
    }
