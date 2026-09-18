import json
import os
from typing import Any

import psycopg


DDL = """
CREATE TABLE IF NOT EXISTS multi_ai_decisions(
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    decision TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    consensus_status TEXT NOT NULL,
    rounds INTEGER NOT NULL DEFAULT 0,
    single_voice TEXT,
    research_signal_allowed BOOLEAN NOT NULL DEFAULT FALSE,
    execution_allowed BOOLEAN NOT NULL DEFAULT FALSE,
    evidence JSONB NOT NULL,
    consensus JSONB NOT NULL,
    gate JSONB NOT NULL
)
"""


def _url() -> str:
    return os.getenv("DATABASE_URL", "").strip()


def init_store() -> bool:
    if not _url():
        return False
    with psycopg.connect(_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
        conn.commit()
    return True


def save_decision(evidence: dict[str, Any], consensus: dict[str, Any], gate: dict[str, Any]) -> int | None:
    if not _url():
        return None
    sql = """
    INSERT INTO multi_ai_decisions(
      symbol,timeframe,decision,confidence,consensus_status,rounds,single_voice,
      research_signal_allowed,execution_allowed,evidence,consensus,gate
    ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)
    RETURNING id
    """
    with psycopg.connect(_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (
                evidence.get("symbol","XAUUSD"),
                evidence.get("timeframe","M3"),
                consensus.get("decision","NO_TRADE"),
                float(consensus.get("confidence",0)),
                consensus.get("status","UNKNOWN"),
                int(consensus.get("rounds",0)),
                consensus.get("single_voice",""),
                bool(gate.get("research_signal_allowed")),
                bool(gate.get("execution_allowed")),
                json.dumps(evidence, default=str),
                json.dumps(consensus, default=str),
                json.dumps(gate, default=str),
            ))
            row = cur.fetchone()
        conn.commit()
    return int(row[0]) if row else None


def latest_decision() -> dict[str, Any] | None:
    if not _url():
        return None
    sql = """
    SELECT id,created_at,symbol,timeframe,decision,confidence,consensus_status,rounds,
           single_voice,research_signal_allowed,execution_allowed,evidence,consensus,gate
    FROM multi_ai_decisions ORDER BY id DESC LIMIT 1
    """
    with psycopg.connect(_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            row = cur.fetchone()
    if not row:
        return None
    keys = [
        "id","created_at","symbol","timeframe","decision","confidence","consensus_status","rounds",
        "single_voice","research_signal_allowed","execution_allowed","evidence","consensus","gate"
    ]
    out = dict(zip(keys,row))
    out["created_at"] = out["created_at"].isoformat()
    return out
