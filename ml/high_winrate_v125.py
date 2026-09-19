from ml.high_winrate_v123 import (
    TARGET_RR,
    evaluate_high_winrate_stability,
    high_winrate_quality_gate as _base_quality_gate,
    learn_high_winrate_policy as _v125_policy,
)


MIN_CALIBRATION_WIN_RATE=0.42
MIN_CALIBRATION_EXPECTANCY_R=0.20
MIN_CALIBRATION_PROFIT_FACTOR=1.25
MIN_CALIBRATION_TRADES=20


def _strict_selection(selection):
    selection=selection or {}
    return bool(
        selection.get("confidence_tier")=="STRICT"
        and int(selection.get("trades",0) or 0)>=MIN_CALIBRATION_TRADES
        and float(selection.get("win_rate",0.0) or 0.0)>=MIN_CALIBRATION_WIN_RATE
        and float(selection.get("expectancy_r",-999) or -999)>=MIN_CALIBRATION_EXPECTANCY_R
        and selection.get("profit_factor") is not None
        and float(selection.get("profit_factor",0.0))>=MIN_CALIBRATION_PROFIT_FACTOR
    )


def _filter_contexts(thresholds,detail):
    kept={}
    audit={}
    for key,threshold in (thresholds or {}).items():
        info=(detail or {}).get(str(key)) or {}
        selection=info.get("selection") or {}
        passed=_strict_selection(selection)
        audit[str(key)]={
            "passed":passed,
            "threshold":float(threshold),
            "confidence_tier":selection.get("confidence_tier"),
            "trades":selection.get("trades"),
            "win_rate":selection.get("win_rate"),
            "expectancy_r":selection.get("expectancy_r"),
            "profit_factor":selection.get("profit_factor"),
        }
        if passed:
            kept[str(key)]=float(threshold)
    return kept,audit


def learn_high_winrate_policy(frame,probability,rr=TARGET_RR):
    """V0.12.6 deep-history precision-first policy.

    V0.12.5 proved that recovery thresholds can create broad but unstable
    contexts. With deep history now available, V0.12.6 keeps only calibration
    selections that clear the STRICT Wilson-confidence tier plus explicit
    win-rate, expectancy, profit-factor and sample-size requirements. The final
    holdout remains untouched and is still the promotion authority.
    """
    policy=_v125_policy(frame,probability,rr=rr)

    global_selection=policy.get("global_selection") or {}
    global_pass=bool(
        policy.get("global_threshold") is not None
        and _strict_selection(global_selection)
    )
    global_threshold=(
        float(policy["global_threshold"])
        if global_pass else None
    )

    regime,regime_audit=_filter_contexts(
        policy.get("regime_thresholds") or {},
        policy.get("regime_detail") or {},
    )
    session,session_audit=_filter_contexts(
        policy.get("session_thresholds") or {},
        policy.get("session_detail") or {},
    )
    joint,joint_audit=_filter_contexts(
        policy.get("joint_thresholds") or {},
        policy.get("joint_detail") or {},
    )

    contextual=bool(regime or session or joint)
    mode=(
        "CONTEXTUAL_STRICT_PRECISION" if contextual
        else "GLOBAL_STRICT_PRECISION" if global_threshold is not None
        else "NONE"
    )

    policy.update({
        "mode":mode,
        "version":"V0.12.6",
        "objective":"STRICT_PRECISION_HIGH_WINRATE_RR_1_2",
        "selection_statistic":"STRICT_WILSON_ONLY_PLUS_SUBWINDOW_STABILITY",
        "recovery_used":False,
        "recovery_allowed":False,
        "global_threshold":global_threshold,
        "regime_thresholds":regime,
        "session_thresholds":session,
        "joint_thresholds":joint,
        "allowed_regimes":sorted(regime.keys()),
        "allowed_sessions":sorted(session.keys()),
        "allowed_joint_contexts":sorted(joint.keys()),
        "stable_context_counts":{
            "regime":int(len(regime)),
            "session":int(len(session)),
            "joint":int(len(joint)),
            "global":int(global_threshold is not None),
        },
        "strict_precision_requirements":{
            "confidence_tier":"STRICT",
            "minimum_trades":MIN_CALIBRATION_TRADES,
            "minimum_win_rate":MIN_CALIBRATION_WIN_RATE,
            "minimum_expectancy_r":MIN_CALIBRATION_EXPECTANCY_R,
            "minimum_profit_factor":MIN_CALIBRATION_PROFIT_FACTOR,
        },
        "strict_precision_audit":{
            "global":{
                "passed":global_pass,
                "confidence_tier":global_selection.get("confidence_tier"),
                "trades":global_selection.get("trades"),
                "win_rate":global_selection.get("win_rate"),
                "expectancy_r":global_selection.get("expectancy_r"),
                "profit_factor":global_selection.get("profit_factor"),
            },
            "regime":regime_audit,
            "session":session_audit,
            "joint":joint_audit,
        },
        "has_any_policy":bool(contextual or global_threshold is not None),
    })
    return policy


def high_winrate_quality_gate(
    test_frame,
    test_probability,
    selected_mask,
    applied_thresholds,
    rr,
    policy_status="STRICT_PRECISION_HIGH_WINRATE_RR_1_2",
    temporal_stability=None,
):
    """V0.12.6 final gate with development-only side-stability enforcement."""
    gate=_base_quality_gate(
        test_frame,
        test_probability,
        selected_mask,
        applied_thresholds,
        rr,
        policy_status=policy_status,
        temporal_stability=temporal_stability,
    )
    temporal=temporal_stability or {}
    stable_sides=list(temporal.get("stable_sides") or [])
    directional=temporal.get("directional_stability") or {}
    gate.setdefault("checks",{})["directional_stability"]={
        "passed":bool(stable_sides),
        "value":stable_sides,
        "required":"at least one BUY/SELL side stable across development walk-forward folds",
        "source":"development_only_no_holdout_tuning",
    }
    gate["directional_stability"]=directional
    gate["stable_sides"]=stable_sides
    objective=gate.get("objective") or {}
    objective.update({
        "version":"V0.12.6",
        "name":"STRICT_PRECISION_HIGH_WINRATE_RR_1_2",
        "calibration_recovery_allowed":False,
        "minimum_calibration_trades":MIN_CALIBRATION_TRADES,
        "minimum_calibration_win_rate":MIN_CALIBRATION_WIN_RATE,
        "minimum_calibration_expectancy_r":MIN_CALIBRATION_EXPECTANCY_R,
        "minimum_calibration_profit_factor":MIN_CALIBRATION_PROFIT_FACTOR,
        "directional_stability_required":True,
    })
    gate["objective"]=objective
    gate["passed"]=all(item.get("passed",False) for item in gate["checks"].values())
    gate["status"]="PASSED" if gate["passed"] else "FAILED"
    return gate
