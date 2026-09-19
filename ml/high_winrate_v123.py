import numpy as np

from ml.high_winrate import (
    TARGET_RR,
    CALIBRATION_WIN_RATE_TARGET,
    FINAL_EXPECTANCY_TARGET,
    FINAL_PF_TARGET,
    FINAL_WILSON_LCB_MIN,
    FINAL_WIN_RATE_TARGET,
    evaluate_high_winrate_stability as _base_stability,
    high_winrate_quality_gate as _base_quality_gate,
    learn_high_winrate_policy as _base_policy,
)
from ml.meta_precision import wilson_lower_bound
from ml.performance import trading_metrics


def _segment_rows(work,parts=3):
    if "timestamp" in work.columns:
        ordered=work.sort_values("timestamp")
    else:
        ordered=work.copy()
    return [segment for segment in np.array_split(ordered,parts) if len(segment)]


def _finite(value,default=None):
    try:
        number=float(value)
    except (TypeError,ValueError):
        return default
    return number if np.isfinite(number) else default


def _median(values):
    clean=[number for number in (_finite(value) for value in values) if number is not None]
    return float(np.median(clean)) if clean else None


def _stable_threshold(work,probability,threshold,rr=TARGET_RR,group_col=None,group_key=None):
    probe=work.copy()
    probe["_p_v124"]=np.asarray(probability,dtype=float)
    segments=_segment_rows(probe,3)
    rows=[]

    for segment in segments:
        group_mask=np.ones(len(segment),dtype=bool)
        if group_col is not None:
            group_mask=segment[group_col].astype(str).to_numpy()==str(group_key)
        take=group_mask & (segment["_p_v124"].to_numpy(dtype=float)>=float(threshold))
        trades=int(np.sum(take))
        metrics=(
            trading_metrics(
                segment.loc[take,"label"],
                segment.loc[take,"_p_v124"],
                threshold=0.0,
                rr=rr,
            )
            if trades else {"trades":0}
        )
        rows.append({
            "trades":trades,
            "win_rate":metrics.get("win_rate"),
            "expectancy_r":metrics.get("expectancy_r"),
            "profit_factor":metrics.get("profit_factor"),
            "net_r":metrics.get("net_r"),
        })

    if group_col is None:
        pooled_group=np.ones(len(probe),dtype=bool)
    else:
        pooled_group=probe[group_col].astype(str).to_numpy()==str(group_key)
    pooled_take=pooled_group & (probe["_p_v124"].to_numpy(dtype=float)>=float(threshold))
    pooled=(
        trading_metrics(
            probe.loc[pooled_take,"label"],
            probe.loc[pooled_take,"_p_v124"],
            threshold=0.0,
            rr=rr,
        )
        if np.any(pooled_take) else {"trades":0}
    )

    usable=[row for row in rows if int(row.get("trades",0) or 0)>=4]
    positive=sum(
        1 for row in usable
        if (_finite(row.get("expectancy_r"),-999.0) or -999.0)>0
        and (_finite(row.get("win_rate"),0.0) or 0.0)>=0.36
    )
    pooled_trades=int(pooled.get("trades",0) or 0)
    pooled_wr=_finite(pooled.get("win_rate"),0.0) or 0.0
    pooled_exp=_finite(pooled.get("expectancy_r"),-999.0)
    pooled_pf_value=_finite(pooled.get("profit_factor"),0.0) or 0.0

    passed=bool(
        len(usable)>=2
        and positive>=2
        and pooled_trades>=15
        and pooled_wr>=0.38
        and pooled_exp is not None and pooled_exp>0
        and pooled_pf_value>=1.05
    )
    return {
        "passed":passed,
        "threshold":float(threshold),
        "usable_segments":int(len(usable)),
        "positive_segments":int(positive),
        "segments":rows,
        "pooled":pooled,
        "requirements":{
            "minimum_usable_segments":2,
            "minimum_positive_segments":2,
            "minimum_pooled_trades":15,
            "minimum_pooled_win_rate":0.38,
            "minimum_pooled_expectancy_r":0.0,
            "minimum_pooled_profit_factor":1.05,
        },
    }


def _prune_context_map(work,probability,rr,column,thresholds):
    kept={}
    stability={}
    for key,threshold in (thresholds or {}).items():
        check=_stable_threshold(
            work,probability,threshold,rr=rr,group_col=column,group_key=key
        )
        stability[str(key)]=check
        if check["passed"]:
            kept[str(key)]=float(threshold)
    return kept,stability


def learn_high_winrate_policy(frame,probability,rr=TARGET_RR):
    """Calibration-only context policy with chronological stability pruning."""
    policy=_base_policy(frame,probability,rr=rr)
    work=frame.copy()
    if "session_name" not in work:
        work["session_name"]="unknown"
    work["joint_context"]=work["regime"].astype(str)+"|"+work["session_name"].astype(str)

    global_threshold=policy.get("global_threshold")
    global_stability=None
    if global_threshold is not None:
        global_stability=_stable_threshold(
            work,probability,global_threshold,rr=rr
        )
        if not global_stability["passed"]:
            global_threshold=None

    regime_thresholds,regime_stability=_prune_context_map(
        work,probability,rr,"regime",policy.get("regime_thresholds") or {}
    )
    session_thresholds,session_stability=_prune_context_map(
        work,probability,rr,"session_name",policy.get("session_thresholds") or {}
    )
    joint_thresholds,joint_stability=_prune_context_map(
        work,probability,rr,"joint_context",policy.get("joint_thresholds") or {}
    )

    contextual=bool(regime_thresholds or session_thresholds or joint_thresholds)
    mode=(
        "CONTEXTUAL_STABLE_HIGH_WINRATE" if contextual
        else "GLOBAL_STABLE_HIGH_WINRATE" if global_threshold is not None
        else "NONE"
    )
    policy.update({
        "mode":mode,
        "version":"V0.12.5",
        "selection_statistic":"STRICT_FIRST_RECOVERY_PLUS_CALIBRATION_SUBWINDOW_STABILITY",
        "global_threshold":global_threshold,
        "regime_thresholds":regime_thresholds,
        "session_thresholds":session_thresholds,
        "joint_thresholds":joint_thresholds,
        "allowed_regimes":sorted(regime_thresholds.keys()),
        "allowed_sessions":sorted(session_thresholds.keys()),
        "allowed_joint_contexts":sorted(joint_thresholds.keys()),
        "context_stability":{
            "global":global_stability,
            "regime":regime_stability,
            "session":session_stability,
            "joint":joint_stability,
        },
        "stable_context_counts":{
            "regime":int(len(regime_thresholds)),
            "session":int(len(session_thresholds)),
            "joint":int(len(joint_thresholds)),
            "global":int(global_threshold is not None),
        },
        "has_any_policy":bool(contextual or global_threshold is not None),
    })
    return policy


def _directional_stability(result,rr):
    """Aggregate BUY/SELL walk-forward evidence using development folds only.

    A side must exist in at least three chronological validation folds and must
    show positive expectancy in at least two of them. This prevents a late
    regime pocket from activating a direction merely because the untouched
    holdout later happens to look good.
    """
    out={}
    break_even=1.0/(1.0+float(rr))
    for side in ("BUY","SELL"):
        rows=[]
        for fold in result.get("folds") or []:
            info=(fold.get("sides") or {}).get(side) or {}
            trade=info.get("trading") or {}
            trades=int(trade.get("trades",0) or 0)
            rows.append({
                "fold":fold.get("fold"),
                "policy_mode":info.get("policy_mode"),
                "trades":trades,
                "win_rate":trade.get("win_rate"),
                "expectancy_r":trade.get("expectancy_r"),
                "profit_factor":trade.get("profit_factor"),
                "max_drawdown_r":trade.get("max_drawdown_r"),
            })

        usable=[row for row in rows if int(row.get("trades",0) or 0)>=15]
        positive=[
            row for row in usable
            if (_finite(row.get("expectancy_r"),-999.0) or -999.0)>0
            and (_finite(row.get("win_rate"),0.0) or 0.0)>=0.36
        ]
        pooled_trades=sum(int(row.get("trades",0) or 0) for row in usable)
        pooled_wins=sum(
            int(round((_finite(row.get("win_rate"),0.0) or 0.0)*int(row.get("trades",0) or 0)))
            for row in usable
        )
        pooled_lcb=wilson_lower_bound(pooled_wins,pooled_trades) if pooled_trades else 0.0
        median_wr=_median([row.get("win_rate") for row in usable])
        median_exp=_median([row.get("expectancy_r") for row in usable])
        median_pf=_median([row.get("profit_factor") for row in usable])

        passed=bool(
            len(usable)>=3
            and len(positive)>=2
            and median_wr is not None and median_wr>=0.36
            and median_exp is not None and median_exp>0
            and median_pf is not None and median_pf>=1.05
            and pooled_lcb>=break_even
        )
        out[side]={
            "passed":passed,
            "usable_folds":int(len(usable)),
            "positive_folds":int(len(positive)),
            "median_win_rate":median_wr,
            "median_expectancy_r":median_exp,
            "median_profit_factor":median_pf,
            "pooled_trades":int(pooled_trades),
            "pooled_wins":int(pooled_wins),
            "pooled_wilson_lcb_95":float(pooled_lcb),
            "folds":rows,
            "requirements":{
                "minimum_usable_folds":3,
                "minimum_positive_folds":2,
                "minimum_median_win_rate":0.36,
                "minimum_median_expectancy_r":0.0,
                "minimum_median_profit_factor":1.05,
                "minimum_pooled_wilson_lcb_95":break_even,
                "minimum_trades_per_usable_fold":15,
            },
        }
    return out


def evaluate_high_winrate_stability(
    development,
    base_features,
    rr=TARGET_RR,
    horizon=80,
    folds=3,
    min_side_train=120,
    n_estimators=140,
):
    # Deep-history mode uses four expanding chronological folds whenever possible.
    result=_base_stability(
        development,
        base_features,
        rr=rr,
        horizon=horizon,
        folds=max(4,int(folds)),
        min_side_train=min_side_train,
        n_estimators=n_estimators,
    )
    directional=_directional_stability(result,rr)
    result["directional_stability"]=directional
    stable_sides=[side for side,info in directional.items() if info.get("passed")]
    result["stable_sides"]=stable_sides
    result["directional_gate_passed"]=bool(stable_sides)
    result["passed"]=bool(result.get("passed",False) and stable_sides)
    result["status"]="PASSED" if result["passed"] else "FAILED"
    return result


def high_winrate_quality_gate(
    test_frame,
    test_probability,
    selected_mask,
    applied_thresholds,
    rr,
    policy_status="HIGH_WINRATE_RR_1_2_DIRECTIONAL_STABILITY",
    temporal_stability=None,
):
    gate=_base_quality_gate(
        test_frame,
        test_probability,
        selected_mask,
        applied_thresholds,
        rr,
        policy_status=policy_status,
        temporal_stability=temporal_stability,
    )
    objective=gate.get("objective") or {}
    objective.update({
        "version":"V0.12.5",
        "context_policy":"CALIBRATION_SUBWINDOW_PLUS_DIRECTIONAL_WALK_FORWARD_STABILITY",
        "walk_forward_folds_target":4,
        "directional_gate":"DEVELOPMENT_ONLY_NO_HOLDOUT_TUNING",
        "deep_history_mode":True,
        "feature_schema":"SMC_PA_ORDERFLOW_REGIME_V1",
        "final_win_rate_target":FINAL_WIN_RATE_TARGET,
        "final_win_rate_wilson_lcb_min":FINAL_WILSON_LCB_MIN,
        "final_expectancy_target_r":FINAL_EXPECTANCY_TARGET,
        "final_profit_factor_target":FINAL_PF_TARGET,
    })
    gate["objective"]=objective
    gate["directional_stability"]=(temporal_stability or {}).get("directional_stability")
    return gate
