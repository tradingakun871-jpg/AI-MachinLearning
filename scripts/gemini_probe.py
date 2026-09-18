import asyncio
import json
import time

from multi_ai.providers import call_gemini


async def main():
    prompt = (
        'Connectivity check. Return ONLY JSON: '
        '{"decision":"NO_TRADE","confidence":100,"veto":false,'
        '"reason":"gemini connectivity ok","checks":{"risk":true}}'
    )
    started = time.perf_counter()
    try:
        result = await call_gemini(prompt)
        print("GEMINI_PROBE=" + json.dumps({
            "ok": bool(result.get("raw_valid")),
            "model": result.get("model"),
            "decision": result.get("decision"),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }, separators=(",", ":")), flush=True)
    except Exception as exc:
        print("GEMINI_PROBE=" + json.dumps({
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        }, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
