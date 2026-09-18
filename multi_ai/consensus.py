import json
import os
from datetime import datetime, timezone
from typing import Any

from multi_ai.providers import ask_all

SYSTEM_RULES = """
You are one member of a four-model trading reasoning council.
You receive exactly the same evidence as the other models.
Your job is NOT to invent missing market data and NOT to place orders.

Return ONLY one JSON object:
{
  "decision": "BUY" | "SELL" | "NO_TRADE",
  "confidence": 0-100,
  "veto": true | false,
  "reason": "short evidence-grounded reason",
  "checks": {
    "structure": true | false,
    "ml": true | false,
    "sentiment": true | false,
    "forecast": true | false,
    "risk": true | false
  }
}

Rules:
- Missing/contradictory critical evidence => NO_TRADE.
- Never convert provider availability problems into BUY/SELL.
- Use only supplied evidence.
- Confidence is calibration, not permission to trade.
- A veto means you found a critical reason execution should be blocked.
""".strip()


def _agreement(votes: list[dict[str, Any]]) -> tuple[bool, str | None]:
    if len(votes) != 4 or any(not v.get("raw_valid") for v in votes):
        return False, None
    decisions = [v.get("decision") for v in votes]
    if len(set(decisions)) == 1:
        return True, decisions[0]
    return False, None


def _confidence(votes: list[dict[str, Any]]) -> float:
    vals = [float(v.get("confidence", 0)) for v in votes if v.get("raw_valid")]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def _single_voice(decision: str, confidence: float, votes: list[dict[str, Any]], round_no: int) -> str:
    if decision == "NO_TRADE":
        return f"NO TRADE. Konsensus 4/4 tercapai pada ronde {round_no}; evidence belum mendukung eksekusi dengan aman."
    common = [v.get("reason", "") for v in votes if v.get("reason")]
    seed = common[0][:320] if common else "Evidence mendukung arah yang sama."
    return f"{decision}. Konsensus 4/4 tercapai pada ronde {round_no} dengan confidence gabungan {confidence:.1f}%. {seed}"


async def run_consensus(evidence: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"), default=str)
    first_prompt = f"{SYSTEM_RULES}\n\nROUND=1\nEVIDENCE={payload}"
    round1 = await ask_all(first_prompt)
    ok1, decision1 = _agreement(round1)
    if ok1:
        conf = _confidence(round1)
        return {
            "status": "CONSENSUS",
            "decision": decision1,
            "confidence": conf,
            "rounds": 1,
            "agreement": "4/4",
            "single_voice": _single_voice(decision1, conf, round1, 1),
            "votes": round1,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    conflict = [
        {
            "provider": v.get("provider"),
            "decision": v.get("decision"),
            "confidence": v.get("confidence"),
            "veto": v.get("veto"),
            "reason": v.get("reason"),
            "raw_valid": v.get("raw_valid"),
        }
        for v in round1
    ]
    second_prompt = (
        f"{SYSTEM_RULES}\n\nROUND=2 FINAL RE-EVALUATION\n"
        f"EVIDENCE={payload}\n"
        f"ROUND1_CONFLICT={json.dumps(conflict, ensure_ascii=False, separators=(',', ':'))}\n"
        "Re-evaluate the evidence and the conflict. Do not compromise merely to agree. "
        "If any critical uncertainty remains, choose NO_TRADE."
    )
    round2 = await ask_all(second_prompt)
    ok2, decision2 = _agreement(round2)
    if ok2:
        conf = _confidence(round2)
        return {
            "status": "CONSENSUS",
            "decision": decision2,
            "confidence": conf,
            "rounds": 2,
            "agreement": "4/4",
            "single_voice": _single_voice(decision2, conf, round2, 2),
            "votes": round2,
            "round1": conflict,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    return {
        "status": "NO_CONSENSUS",
        "decision": "NO_TRADE",
        "confidence": 0.0,
        "rounds": 2,
        "agreement": "FAILED",
        "single_voice": "NO TRADE. Empat reasoning AI tidak mencapai satu suara setelah dua ronde.",
        "votes": round2,
        "round1": conflict,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
