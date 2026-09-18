import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_ai.providers import CALLERS, provider_status

PROBE_TIMEOUT_SECONDS = 20

PROMPT = (
    'Connectivity self-test only. Return ONLY JSON: '
    '{"decision":"NO_TRADE","confidence":100,"veto":false,'
    '"reason":"provider self-test ok","checks":{"structure":false,"ml":false,'
    '"sentiment":false,"forecast":false,"risk":true}}'
)


async def probe_one(name, fn, statuses):
    info = statuses[name]
    if not info.get("configured"):
        return {
            "provider": name,
            "configured": False,
            "reachable": False,
            "model": info.get("model"),
            "decision": None,
            "error": "NOT_CONFIGURED",
        }
    try:
        vote = await asyncio.wait_for(fn(PROMPT), timeout=PROBE_TIMEOUT_SECONDS)
        return {
            "provider": name,
            "configured": True,
            "reachable": bool(vote.get("raw_valid")),
            "model": info.get("model"),
            "decision": vote.get("decision"),
            "error": None,
        }
    except asyncio.TimeoutError:
        return {
            "provider": name,
            "configured": True,
            "reachable": False,
            "model": info.get("model"),
            "decision": None,
            "error": "PROBE_TIMEOUT",
        }
    except Exception as exc:
        return {
            "provider": name,
            "configured": True,
            "reachable": False,
            "model": info.get("model"),
            "decision": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def main():
    statuses = provider_status()
    rows = await asyncio.gather(
        *(probe_one(name, fn, statuses) for name, fn in CALLERS.items())
    )
    safe = {
        "ready_4_of_4": all(row.get("reachable") for row in rows),
        "timeout_seconds": PROBE_TIMEOUT_SECONDS,
        "providers": rows,
    }
    print("PROVIDER_PROBE=" + json.dumps(safe, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
