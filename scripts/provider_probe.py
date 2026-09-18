import asyncio
import json
from multi_ai.providers import self_test_all


async def main():
    result = await self_test_all()
    safe = {
        "ready_4_of_4": result.get("ready_4_of_4", False),
        "providers": [
            {
                "provider": row.get("provider"),
                "configured": row.get("configured"),
                "reachable": row.get("reachable"),
                "model": row.get("model"),
                "decision": row.get("decision"),
                "error": row.get("error"),
            }
            for row in result.get("providers", [])
        ],
    }
    print("PROVIDER_PROBE=" + json.dumps(safe, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
