import asyncio
import json
import os
import re

import httpx


def clean(value):
    text = str(value or "")
    text = re.sub(r"sk-[A-Za-z0-9_\-]+", "[REDACTED]", text)
    text = re.sub(r"AQ\.[A-Za-z0-9_\-]+", "[REDACTED]", text)
    return text[:700]


def safe_error(resp):
    try:
        data = resp.json()
        err = data.get("error", data) if isinstance(data, dict) else data
        if isinstance(err, dict):
            return {
                "type": clean(err.get("type") or err.get("status") or err.get("code")),
                "code": clean(err.get("code")),
                "message": clean(err.get("message") or err.get("error") or err),
            }
        return {"message": clean(err)}
    except Exception:
        return {"message": clean(resp.text)}


async def post(name, url, headers, body, timeout=20):
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


async def openai_probe():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return {"provider": "chatgpt", "ok": False, "error": {"type": "not_configured"}}
    return await post(
        "chatgpt",
        "https://api.openai.com/v1/responses",
        {"authorization": f"Bearer {key}", "content-type": "application/json"},
        {
            "model": os.getenv("OPENAI_MODEL", "gpt-5.6-sol"),
            "input": "Reply only with the word OK.",
            "reasoning": {"effort": "low"},
            "max_output_tokens": 32,
        },
    )


async def claude_probe():
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return {"provider": "claude", "ok": False, "error": {"type": "not_configured"}}
    return await post(
        "claude",
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        {
            "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Reply only with the word OK."}],
        },
    )


async def gemini_probe():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return {"provider": "gemini", "ok": False, "error": {"type": "not_configured"}}
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    return await post(
        "gemini",
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {"x-goog-api-key": key, "content-type": "application/json"},
        {
            "contents": [{"role": "user", "parts": [{"text": "Reply only with the word OK."}]}],
            "generationConfig": {"maxOutputTokens": 32},
        },
    )


async def deepseek_probe():
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        return {"provider": "deepseek", "ok": False, "error": {"type": "not_configured"}}
    base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    return await post(
        "deepseek",
        f"{base}/chat/completions",
        {"authorization": f"Bearer {key}", "content-type": "application/json"},
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply only with the word OK."}],
            "stream": False,
            "max_tokens": 32,
            "thinking": {"type": "disabled"},
        },
    )


async def bounded(coro, name):
    try:
        return await asyncio.wait_for(coro, timeout=22)
    except asyncio.TimeoutError:
        return {"provider": name, "ok": False, "error": {"type": "hard_timeout"}}


async def main():
    rows = await asyncio.gather(
        bounded(openai_probe(), "chatgpt"),
        bounded(claude_probe(), "claude"),
        bounded(gemini_probe(), "gemini"),
        bounded(deepseek_probe(), "deepseek"),
    )
    print("PROVIDER_DETAIL=" + json.dumps(rows, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
