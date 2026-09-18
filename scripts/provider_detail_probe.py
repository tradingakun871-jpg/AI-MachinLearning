import asyncio
import json
import os
import re

import httpx


def clean(value):
    text = str(value or "")
    text = re.sub(r"sk-[A-Za-z0-9_\-]+", "[REDACTED]", text)
    text = re.sub(r"AQ\.[A-Za-z0-9_\-]+", "[REDACTED]", text)
    return text[:500]


def safe_error(resp):
    try:
        data = resp.json()
        err = data.get("error", data)
        if isinstance(err, dict):
            return {
                "type": clean(err.get("type") or err.get("status") or err.get("code")),
                "message": clean(err.get("message") or err.get("error") or err),
            }
        return {"message": clean(err)}
    except Exception:
        return {"message": clean(resp.text)}


async def post(name, url, headers, body, timeout=45):
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, headers=headers, json=body)
        result = {"provider": name, "http": r.status_code, "ok": 200 <= r.status_code < 300}
        if not result["ok"]:
            result["error"] = safe_error(r)
        else:
            result["response_shape"] = sorted(list((r.json() or {}).keys()))[:12]
        return result
    except httpx.TimeoutException:
        return {"provider": name, "ok": False, "error": {"type": "timeout"}}
    except Exception as exc:
        return {"provider": name, "ok": False, "error": {"type": type(exc).__name__, "message": clean(exc)}}


async def main():
    rows = []

    ck = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if ck:
        rows.append(await post(
            "claude",
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": ck, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            {"model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"), "max_tokens": 64, "messages": [{"role": "user", "content": "Reply only with the word OK."}]},
        ))
    else:
        rows.append({"provider": "claude", "ok": False, "error": {"type": "not_configured"}})

    gk = os.getenv("GEMINI_API_KEY", "").strip()
    if gk:
        gm = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        rows.append(await post(
            "gemini",
            f"https://generativelanguage.googleapis.com/v1beta/models/{gm}:generateContent",
            {"x-goog-api-key": gk, "content-type": "application/json"},
            {"contents": [{"parts": [{"text": "Reply only with the word OK."}]}]},
        ))
    else:
        rows.append({"provider": "gemini", "ok": False, "error": {"type": "not_configured"}})

    dk = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if dk:
        base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        dm = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
        rows.append(await post(
            "deepseek",
            f"{base}/chat/completions",
            {"authorization": f"Bearer {dk}", "content-type": "application/json"},
            {"model": dm, "messages": [{"role": "user", "content": "Reply only with the word OK."}], "stream": False},
            timeout=60,
        ))
    else:
        rows.append({"provider": "deepseek", "ok": False, "error": {"type": "not_configured"}})

    print("PROVIDER_DETAIL=" + json.dumps(rows, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
