import math
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ml.performance import grouped_performance, probability_metrics, trading_metrics


def threshold_sweep_realistic(y,p,rr=3.0,start=0.30,end=0.85,step=0.01,min_trades=30,min_coverage=0.03):
    y=np.asarray(y,dtype=int)
    p=np.asarray(p,dtype=float)
    rows=[]
    for t in np.arange(start,end+1e-9,step):
        m=trading_metrics(y,p,float(t),rr)
        trades=int(m.get("trades",0) or 0)
        coverage=float(trades/len(y)) if len(y) else 0.0
        if trades<min_trades or coverage<min_coverage:
            continue
        dd=abs(float(m.get("max_drawdown_r",0.0) or 0.0))
        expectancy=float(m.get("expectancy_r",-999) or -999)
        pf=m.get("profit_factor")
        pf_score=float(pf) if pf is not None and np.isfinite(pf) else 0.0
        # Penalize tiny samples and deep drawdown. This avoids selecting an
        # extreme threshold because of only a few lucky trades.
        score=expectancy*math.sqrt(trades) + 0.05*pf_score - 0.01*dd
        rows.append({
            "threshold":round(float(t),3),
            "coverage":coverage,
            "score":float(score),
            **m,
        })
    return rows


def select_threshold(y,p,rr=3.0):
    min_trades=max(30,int(len(y)*0.05))
    rows=threshold_sweep_realistic(
        y,p,rr=rr,min_trades=min_trades,min_coverage=0.05
    )
    viable=[
        x for x in rows
        if float(x.get("expectancy_r",-999))>0
        and float(x.get("net_r",-999))>0
        and (x.get("profit_factor") is None or float(x.get("profit_factor",0))>=1.05)
    ]
    selected=max(viable,key=lambda x:(x["score"],x["trades"])) if viable else None
    return selected,rows


def _group_rules(frame,probability,threshold,rr,group_col,min_group_rows=80,min_group_trades=15):
    work=frame.copy()
    work["probability"]=np.asarray(probability,dtype=float)
    results=grouped_performance(work,group_col,threshold,rr)
    allowed=[]
    detail={}
    for key,g in work.groupby(group_col,dropna=False):
        name=str(key)
        metrics=results.get(name,{"trades":0})
        sample_rows=int(len(g))
        trades=int(metrics.get("trades",0) or 0)
        exp=float(metrics.get("expectancy_r",-999) or -999)
        pf=metrics.get("profit_factor")
        pf_ok=pf is None or float(pf)>=1.0
        eligible=sample_rows>=min_group_rows and trades>=min_group_trades and exp>0 and pf_ok
        detail[name]={"sample_rows":sample_rows,"eligible":eligible,**metrics}
        if eligible:
            allowed.append(name)
    return allowed,detail


def learn_regime_session_filters(frame,probability,threshold,rr=3.0):
    work=frame.copy()
    if "session_name" not in work:
        work["session_name"]="unknown"
    regimes,regime_detail=_group_rules(
        work,probability,threshold,rr,"regime",
        min_group_rows=max(60,int(len(work)*0.05)),
        min_group_trades=max(10,int(len(work)*0.015)),
    )
    sessions,session_detail=_group_rules(
        work,probability,threshold,rr,"session_name",
        min_group_rows=max(80,int(len(work)*0.08)),
        min_group_trades=max(12,int(len(work)*0.02)),
    )
    return {
        "allowed_regimes":regimes,
        "allowed_sessions":sessions,
        "regime_detail":regime_detail,
        "session_detail":session_detail,
    }


def apply_learned_filters(frame,probability,filters):
    mask=np.ones(len(frame),dtype=bool)
    regimes=filters.get("allowed_regimes") or []
    sessions=filters.get("allowed_sessions") or []
    if regimes:
        mask &= frame["regime"].astype(str).isin(regimes).to_numpy()
    if sessions and "session_name" in frame:
        mask &= frame["session_name"].astype(str).isin(sessions).to_numpy()
    return mask


def quality_gate(test_frame,test_probability,threshold,rr,filters,calibration_selection=None):
    y=np.asarray(test_frame["label"],dtype=int)
    p=np.asarray(test_probability,dtype=float)
    prob=probability_metrics(y,p)
    prob["pr_auc"]=float(average_precision_score(y,p)) if len(np.unique(y))>1 else None

    group_mask=apply_learned_filters(test_frame,p,filters)
    selected_mask=(p>=float(threshold)) & group_mask
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

    checks={
        "test_rows": {"passed":len(y)>=500,"value":int(len(y)),"minimum":500},
        "test_trades": {"passed":trades>=30,"value":trades,"minimum":30},
        "coverage": {"passed":coverage>=0.03,"value":coverage,"minimum":0.03},
        "auc": {"passed":auc is not None and float(auc)>=0.52,"value":auc,"minimum":0.52},
        "pr_auc_vs_base": {
            "passed":pr_auc is not None and float(pr_auc)>=base_rate*1.02,
            "value":pr_auc,"minimum":base_rate*1.02,
        },
        "brier_vs_base": {
            "passed":prob.get("brier") is not None and float(prob["brier"])<=base_brier*0.99,
            "value":prob.get("brier"),"maximum":base_brier*0.99,
        },
        "expectancy_r": {"passed":expectancy>0,"value":expectancy,"minimum_exclusive":0.0},
        "profit_factor": {
            "passed":pf is not None and float(pf)>=1.10,
            "value":pf,"minimum":1.10,
        },
        "drawdown_r": {"passed":max_dd>=-25.0,"value":max_dd,"minimum":-25.0},
        "calibration_threshold": {
            "passed":calibration_selection is not None,
            "value":None if calibration_selection is None else calibration_selection.get("threshold"),
        },
    }
    passed=all(item["passed"] for item in checks.values())
    return {
        "passed":bool(passed),
        "status":"PASSED" if passed else "FAILED",
        "checks":checks,
        "probability":prob,
        "trading":trade,
        "coverage":coverage,
        "base_rate":base_rate,
        "filtered_test_rows":trades,
    }
