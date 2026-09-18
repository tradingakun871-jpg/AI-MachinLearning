import asyncio
import json
import os
import re
from typing import Any

import httpx

VALID_DECISIONS = {"BUY", "SELL", "NO_TRADE"}
RETRYABLE_HTTP = {500, 502, 503, 504}
TRANSIENT_PROVIDER_ERRORS = {
    "provider_timeout",
    "provider_network_error",
    "provider_http_500",
    "provider_http_502",
    "provider_http_503",
    "provider_http_504",
}


def _json_from_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty model response")
    try:
        return json.loads(text)
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    if fenced:
        return json.loads(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("model did not return valid JSON")


def normalize_vote(provider: str, payload: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    decision = str(payload.get("decision", "NO_TRADE")).upper().replace(" ", "_")
    if decision == "WAIT":
        decision = "NO_TRADE"
    if decision not in VALID_DECISIONS:
        decision = "NO_TRADE"
    try:
        confidence = float(payload.get("confidence", 0))
    except Exception:
        confidence = 0.0
    if 0 <= confidence <= 1:
        confidence *= 100
    confidence = max(0.0, min(100.0, confidence))
    checks = payload.get("checks") if isinstance(payload.get("checks"), dict) else {}
    row = {
        "provider": provider,
        "decision": decision,
        "confidence": round(confidence, 2),
        "veto": bool(payload.get("veto", False)),
        "reason": str(payload.get("reason", "")).strip()[:1200],
        "checks": checks,
        "raw_valid": True,
    }
    if model:
        row["model"] = model
    return row


def provider_status() -> dict[str, dict[str, Any]]:
    return {
        "chatgpt": {
            "configured": bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_MODEL")),
            "model": os.getenv("OPENAI_MODEL", "gpt-5.6-sol"),
        },
        "claude": {
            "configured": bool(os.getenv("ANTHROPIC_API_KEY") and os.getenv("ANTHROPIC_MODEL")),
            "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
        },
        "gemini": {
            "configured": bool(os.getenv("GEMINI_API_KEY") and os.getenv("GEMINI_MODEL")),
            "model": os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        },
        "deepseek": {
            "configured": bool(os.getenv("DEEPSEEK_API_KEY") and os.getenv("DEEPSEEK_MODEL") and os.getenv("DEEPSEEK_BASE_URL")),
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        },
    }


async def _post(
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, Any],
    timeout: float = 45.0,
    retries: int = 0,
):
    last_error: str | None = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, headers=headers, json=json_body)
            if r.status_code in RETRYABLE_HTTP and attempt < retries:
                last_error = f"provider_http_{r.status_code}"
                await asyncio.sleep(min(4.0, 0.75 * (2 ** attempt)))
                continue
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"provider_http_{exc.response.status_code}") from None
        except httpx.TimeoutException:
            last_error = "provider_timeout"
            if attempt < retries:
                await asyncio.sleep(min(4.0, 0.75 * (2 ** attempt)))
                continue
            raise RuntimeError(last_error) from None
        except httpx.RequestError:
            last_error = "provider_network_error"
            if attempt < retries:
                await asyncio.sleep(min(4.0, 0.75 * (2 ** attempt)))
                continue
            raise RuntimeError(last_error) from None
    raise RuntimeError(last_error or "provider_request_failed")


async def call_openai(prompt: str) -> dict[str, Any]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-sol").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not configured")
    data = await _post(
        os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json_body={
            "model": model,
            "input": prompt,
            "reasoning": {"effort": os.getenv("OPENAI_REASONING_EFFORT", "medium")},
            "max_output_tokens": int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1200")),
        },
        timeout=float(os.getenv("OPENAI_TIMEOUT", "45")),
    )
    text = data.get("output_text")
    if not text:
        chunks = []
        for item in data.get("output", []) or []:
            for c in item.get("content", []) or []:
                if c.get("type") in ("output_text", "text") and c.get("text"):
                    chunks.append(c["text"])
        text = "\n".join(chunks)
    return normalize_vote("chatgpt", _json_from_text(text or ""), model)


async def call_claude(prompt: str) -> dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not configured")
    data = await _post(
        os.getenv("ANTHROPIC_MESSAGES_URL", "https://api.anthropic.com/v1/messages"),
        headers={
            "x-api-key": key,
            "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
            "Content-Type": "application/json",
        },
        json_body={
            "model": model,
            "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "1200")),
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=float(os.getenv("ANTHROPIC_TIMEOUT", "45")),
    )
    text = "\n".join(x.get("text", "") for x in data.get("content", []) if x.get("type") == "text")
    return normalize_vote("claude", _json_from_text(text), model)


def _gemini_interactions_text(data: dict[str, Any]) -> str:
    chunks: list[str] = []
    for step in data.get("steps", []) or []:
        if step.get("type") != "model_output":
            continue
        for part in step.get("content", []) or []:
            if part.get("type") == "text" and part.get("text"):
                chunks.append(part["text"])
    if chunks:
        return "\n".join(chunks)
    if data.get("output_text"):
        return str(data["output_text"])
    return ""


def _gemini_generate_text(data: dict[str, Any]) -> str:
    chunks: list[str] = []
    for candidate in data.get("candidates", []) or []:
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            if part.get("text"):
                chunks.append(str(part["text"]))
    return "\n".join(chunks)


async def _call_gemini_interactions(prompt: str, key: str, model: str) -> dict[str, Any]:
    base = os.getenv("GEMINI_INTERACTIONS_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    data = await _post(
        f"{base}/interactions",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json_body={
            "model": model,
            "input": prompt,
            "store": False,
            "generation_config": {
                "thinking_level": os.getenv("GEMINI_THINKING_LEVEL", "low"),
                "max_output_tokens": int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1200")),
            },
        },
        timeout=float(os.getenv("GEMINI_INTERACTIONS_TIMEOUT", "15")),
        retries=int(os.getenv("GEMINI_INTERACTIONS_RETRIES", "0")),
    )
    return normalize_vote("gemini", _json_from_text(_gemini_interactions_text(data)), model)


async def _call_gemini_generate_content(prompt: str, key: str, model: str) -> dict[str, Any]:
    base = os.getenv("GEMINI_GENERATE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    data = await _post(
        f"{base}/models/{model}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json_body={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "thinkingConfig": {"thinkingLevel": os.getenv("GEMINI_THINKING_LEVEL", "low")},
                "maxOutputTokens": int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1200")),
            },
        },
        timeout=float(os.getenv("GEMINI_GENERATE_TIMEOUT", "20")),
        retries=int(os.getenv("GEMINI_GENERATE_RETRIES", "0")),
    )
    return normalize_vote("gemini", _json_from_text(_gemini_generate_text(data)), model)


async def call_gemini(prompt: str) -> dict[str, Any]:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    primary = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    fallback = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash").strip()
    mode = os.getenv("GEMINI_API_MODE", "auto").strip().lower()
    if not key:
        raise RuntimeError("GEMINI_API_KEY not configured")

    last_error: Exception | None = None
    if mode in {"auto", "interactions"}:
        try:
            return await _call_gemini_interactions(prompt, key, primary)
        except RuntimeError as exc:
            last_error = exc
            if mode == "interactions" or str(exc) not in TRANSIENT_PROVIDER_ERRORS:
                raise

    if mode in {"auto", "generatecontent", "generate_content"}:
        try:
            return await _call_gemini_generate_content(prompt, key, primary)
        except RuntimeError as exc:
            last_error = exc
            if fallback and fallback != primary and str(exc) in TRANSIENT_PROVIDER_ERRORS:
                return await _call_gemini_generate_content(prompt, key, fallback)
            raise

    raise RuntimeError(str(last_error or "invalid GEMINI_API_MODE"))


async def call_deepseek(prompt: str) -> dict[str, Any]:
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro").strip()
    base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not configured")
    data = await _post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json_body={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "reasoning_effort": os.getenv("DEEPSEEK_REASONING_EFFORT", "high"),
            "thinking": {"type": os.getenv("DEEPSEEK_THINKING", "enabled")},
            "response_format": {"type": "json_object"},
        },
        timeout=float(os.getenv("DEEPSEEK_TIMEOUT", "45")),
    )
    text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    return normalize_vote("deepseek", _json_from_text(text), model)


CALLERS = {
    "chatgpt": call_openai,
    "claude": call_claude,
    "gemini": call_gemini,
    "deepseek": call_deepseek,
}


async def ask_all(prompt: str) -> list[dict[str, Any]]:
    async def one(name: str, fn):
        try:
            timeout = float(os.getenv("PROVIDER_HARD_TIMEOUT", "55"))
            return await asyncio.wait_for(fn(prompt), timeout=timeout)
        except asyncio.TimeoutError:
            return {
                "provider": name,
                "decision": "NO_TRADE",
                "confidence": 0.0,
                "veto": True,
                "reason": "PROVIDER_ERROR: provider_hard_timeout",
                "checks": {},
                "raw_valid": False,
            }
        except Exception as exc:
            return {
                "provider": name,
                "decision": "NO_TRADE",
                "confidence": 0.0,
                "veto": True,
                "reason": f"PROVIDER_ERROR: {type(exc).__name__}: {exc}",
                "checks": {},
                "raw_valid": False,
            }

    return await asyncio.gather(*(one(name, fn) for name, fn in CALLERS.items()))


async def self_test_all() -> dict[str, Any]:
    statuses = provider_status()
    prompt = (
        'Connectivity self-test only. Return ONLY JSON: '
        '{"decision":"NO_TRADE","confidence":100,"veto":false,'
        '"reason":"provider self-test ok","checks":{"structure":false,"ml":false,'
        '"sentiment":false,"forecast":false,"risk":true}}'
    )

    async def one(name: str, fn):
        if not statuses[name]["configured"]:
            return {"provider": name, "configured": False, "reachable": False, "model": statuses[name]["model"], "error": "NOT_CONFIGURED"}
        try:
            timeout = float(os.getenv("PROVIDER_SELFTEST_TIMEOUT", "45"))
            vote = await asyncio.wait_for(fn(prompt), timeout=timeout)
            return {
                "provider": name,
                "configured": True,
                "reachable": bool(vote.get("raw_valid")),
                "model": vote.get("model") or statuses[name]["model"],
                "decision": vote.get("decision"),
            }
        except asyncio.TimeoutError:
            return {
                "provider": name,
                "configured": True,
                "reachable": False,
                "model": statuses[name]["model"],
                "error": "provider_hard_timeout",
            }
        except Exception as exc:
            return {
                "provider": name,
                "configured": True,
                "reachable": False,
                "model": statuses[name]["model"],
                "error": f"{type(exc).__name__}: {exc}",
            }

    rows = await asyncio.gather(*(one(name, fn) for name, fn in CALLERS.items()))
    return {
        "ready_4_of_4": all(x.get("reachable") for x in rows),
        "providers": rows,
    }
