from ml.high_winrate_v123 import (
    TARGET_RR,
    evaluate_high_winrate_stability,
    high_winrate_quality_gate as _base_quality_gate,
    learn_high_winrate_policy,
)


def high_winrate_quality_gate(
    test_frame,
    test_probability,
    selected_mask,
    applied_thresholds,
    rr,
    policy_status="HIGH_WINRATE_RR_1_2_DIRECTIONAL_STABILITY",
    temporal_stability=None,
):
    """V0.12.5 final gate with an explicit development-only side-stability check."""
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
    gate["passed"]=all(item.get("passed",False) for item in gate["checks"].values())
    gate["status"]="PASSED" if gate["passed"] else "FAILED"
    return gate
