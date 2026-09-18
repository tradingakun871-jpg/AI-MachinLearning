import os
from typing import Any


def _quality_passed(evidence: dict[str, Any]) -> bool:
    ml = evidence.get("machine_learning") or {}
    model = ml.get("model") or {}
    if bool(model.get("quality_passed")):
        return True
    training = ml.get("training") or {}
    quality = training.get("quality") or {}
    return bool(quality.get("passed"))


def evaluate_gate(evidence: dict[str, Any], consensus: dict[str, Any]) -> dict[str, Any]:
    min_conf = float(os.getenv("MIN_REASONING_CONFIDENCE", "70"))
    votes = consensus.get("votes") or []
    providers_ok = len(votes) == 4 and all(bool(v.get("raw_valid")) for v in votes)
    no_veto = providers_ok and not any(bool(v.get("veto")) for v in votes)
    unanimous = consensus.get("status") == "CONSENSUS" and consensus.get("agreement") == "4/4"
    actionable = consensus.get("decision") in ("BUY", "SELL")
    confidence_ok = float(consensus.get("confidence", 0)) >= min_conf
    ml_quality = _quality_passed(evidence)
    nlp_ready = (evidence.get("nlp_sentiment") or {}).get("status") == "READY"
    dl_ready = (evidence.get("deep_forecast") or {}).get("status") == "READY"

    research_allowed = bool(unanimous and actionable and providers_ok and no_veto and confidence_ok and ml_quality)
    live_switch = os.getenv("ALLOW_MULTI_AI_EXECUTION", "false").lower() == "true"
    trading_mode = os.getenv("TRADING_MODE", "SHADOW").upper()
    execution_allowed = bool(
        research_allowed
        and nlp_ready
        and dl_ready
        and live_switch
        and trading_mode == "LIVE"
        and evidence.get("execution_requested")
    )

    blockers = []
    if not unanimous: blockers.append("STRICT_4_OF_4_CONSENSUS_NOT_MET")
    if not actionable: blockers.append("DECISION_NOT_ACTIONABLE")
    if not providers_ok: blockers.append("ONE_OR_MORE_REASONING_PROVIDERS_UNAVAILABLE")
    if not no_veto: blockers.append("AI_VETO_PRESENT")
    if not confidence_ok: blockers.append("REASONING_CONFIDENCE_BELOW_THRESHOLD")
    if not ml_quality: blockers.append("ML_QUALITY_GATE_NOT_PASSED")
    if not nlp_ready: blockers.append("FINANCIAL_NLP_NOT_PRODUCTION_READY")
    if not dl_ready: blockers.append("DEEP_FORECAST_NOT_PRODUCTION_READY")
    if trading_mode != "LIVE": blockers.append("TRADING_MODE_NOT_LIVE")
    if not live_switch: blockers.append("MULTI_AI_LIVE_SWITCH_OFF")
    if not evidence.get("execution_requested"): blockers.append("EXECUTION_NOT_REQUESTED")

    return {
        "research_signal_allowed": research_allowed,
        "execution_allowed": execution_allowed,
        "minimum_reasoning_confidence": min_conf,
        "checks": {
            "unanimous_4_of_4": unanimous,
            "actionable_direction": actionable,
            "providers_ok": providers_ok,
            "no_veto": no_veto,
            "confidence_ok": confidence_ok,
            "ml_quality_passed": ml_quality,
            "nlp_ready": nlp_ready,
            "deep_forecast_ready": dl_ready,
            "trading_mode": trading_mode,
            "live_switch": live_switch,
        },
        "blockers": blockers,
    }
