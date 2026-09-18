import json
import os
import time

from app import NLPRequest, analyze


def main():
    token = os.getenv("INTERNAL_SERVICE_TOKEN", "").strip()
    started = time.perf_counter()
    result = {"ok": False, "error": "internal token unavailable"}
    if token:
        try:
            req = NLPRequest(
                symbol="XAUUSD",
                texts=["US CPI inflation comes in above forecast while Treasury yields rise."],
                event={"actual": 3.4, "forecast": 3.1},
            )
            data = analyze(req, token)
            result = {
                "ok": data.get("status") == "READY",
                "status": data.get("status"),
                "model": data.get("model"),
                "label": data.get("label"),
                "score": data.get("score"),
                "event_type": (data.get("event_impact") or {}).get("event_type"),
                "event_direction": (data.get("event_impact") or {}).get("direction"),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        except Exception as exc:
            result = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
    print("FINBERT_SELFTEST=" + json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
