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
from ml.performance import trading_metrics


def _segment_rows(work,parts=3):
    if "timestamp" in work.columns:
        ordered=work.sort_values("timestamp")
    else:
        ordered=work.copy()
    return [segment for segment in np.array_split(ordered,parts) if len(segment)]


def _stable_threshold(work,probability,threshold,rr=TARGET_RR,group_col=None,group_key=None):
    probe=work.copy()
    probe["_p_v123"]=np.asarray(probability,dtype=float)
    segments=_segment_rows(probe,3)
    rows=[]
    pooled_mask=np.zeros(len(probe),dtype=bool)

    for segment in segments:
        group_mask=np.ones(len(segment),dtype=bool)
        if group_col is not None:
            group_mask=segment[group_col].astype(str).to_numpy()==str(group_key)
        take=group_mask & (segment["_p_v123"].to_numpy(dtype=float)>=float(threshold))
        trades=int(np.sum(take))
        metrics=(
            trading_metrics(
                segment.loc[take,"label"],
                segment.loc[take,"_p_v123"],
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
    pooled_take=pooled_group & (probe["_p_v123"].to_numpy(dtype=float)>=float(threshold))
    pooled=(
        trading_metrics(
            probe.loc[pooled_take,"label"],
            probe.loc[pooled_take,"_p_v123"],
            threshold=0.0,
            rr=rr,
        )
        if np.any(pooled_take) else {"trades":0}
    )

    usable=[row for row in rows if int(row.get("trades",0) or 0)>=4]
    positive=sum(
        1 for row in usable
        if float(row.get("expectancy_r",-999) or -999)>0
        and float(row.get("win_rate",0.0) or 0.0)>=0.36
    )
    pooled_trades=int(pooled.get("trades",0) or 0)
    pooled_wr=float(pooled.get("win_rate",0.0) or 0.0)
    pooled_exp=float(pooled.get("expectancy_r",-999) or -999)
    pooled_pf=pooled.get("profit_factor")
    pooled_pf_value=float(pooled_pf) if pooled_pf is not None else 0.0

    passed=bool(
        len(usable)>=2
        and positive>=2
        and pooled_trades>=15
        and pooled_wr>=0.38
        and pooled_exp>0
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
    """V0.12.3 calibration policy with subwindow context stability.

    Thresholds are still learned only from calibration data. V0.12.3 then asks
    whether the same regime/session edge survives multiple chronological slices
    before that context is allowed to reach the untouched holdout.
    """
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
        "version":"V0.12.3",
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
    return _base_stability(
        development,
        base_features,
        rr=rr,
        horizon=horizon,
        folds=max(4,int(folds)),
        min_side_train=min_side_train,
        n_estimators=n_estimators,
    )


def high_winrate_quality_gate(
    test_frame,
    test_probability,
    selected_mask,
    applied_thresholds,
    rr,
    policy_status="HIGH_WINRATE_RR_1_2_STABLE_CONTEXT",
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
        "version":"V0.12.3",
        "context_policy":"CALIBRATION_SUBWINDOW_STABILITY",
        "walk_forward_folds_target":4,
        "deep_history_mode":True,
        "final_win_rate_target":FINAL_WIN_RATE_TARGET,
        "final_win_rate_wilson_lcb_min":FINAL_WILSON_LCB_MIN,
        "final_expectancy_target_r":FINAL_EXPECTANCY_TARGET,
        "final_profit_factor_target":FINAL_PF_TARGET,
    })
    gate["objective"]=objective
    return gate
