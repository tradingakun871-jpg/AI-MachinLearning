import json
import os

import psycopg


def _iso(value):
    return value.isoformat() if hasattr(value, "isoformat") else value


def _side_summary(metrics, side):
    info=(metrics.get("side_models") or {}).get(side) or {}
    policy=info.get("threshold_policy") or {}
    meta=info.get("meta_precision") or {}
    return {
        "test_trading":info.get("test_trading"),
        "test_selected_rows":info.get("test_selected_rows"),
        "test_coverage":info.get("test_coverage"),
        "selected_features":info.get("selected_features"),
        "threshold_mode":policy.get("mode"),
        "stable_context_counts":policy.get("stable_context_counts"),
        "allowed_regimes":policy.get("allowed_regimes"),
        "allowed_sessions":policy.get("allowed_sessions"),
        "allowed_joint_contexts":policy.get("allowed_joint_contexts"),
        "recovery_used":policy.get("recovery_used"),
        "hybrid_mode":(info.get("hybrid_gate") or {}).get("mode"),
        "meta_mode":meta.get("mode"),
        "meta_policy_mode":(meta.get("policy") or {}).get("mode"),
    }


def _fold_side(item, side):
    info=(item.get("sides") or {}).get(side) or {}
    trade=info.get("trading") or {}
    return {
        "selected_rows":info.get("selected_rows"),
        "policy_mode":info.get("policy_mode"),
        "trades":trade.get("trades"),
        "win_rate":trade.get("win_rate"),
        "expectancy_r":trade.get("expectancy_r"),
        "profit_factor":trade.get("profit_factor"),
        "max_drawdown_r":trade.get("max_drawdown_r"),
    }


def _training(cur, symbol):
    cur.execute(
        """
        SELECT status, version, dataset_rows, error, recorded_at, metrics
        FROM training_states
        WHERE symbol=%s
        ORDER BY id DESC
        LIMIT 1
        """,
        (symbol,),
    )
    row=cur.fetchone()
    if not row:
        return None
    metrics=row[5] if isinstance(row[5],dict) else {}
    trading=metrics.get("trading") or {}
    temporal=metrics.get("temporal_stability") or {}
    gate=metrics.get("quality_gate") or {}
    checks=gate.get("checks") or {}
    fold_rows=[]
    for item in (temporal.get("folds") or []):
        trade=item.get("trading") or {}
        fold_rows.append({
            "fold":item.get("fold"),
            "passed":item.get("passed"),
            "trades":trade.get("trades"),
            "win_rate":trade.get("win_rate"),
            "expectancy_r":trade.get("expectancy_r"),
            "profit_factor":trade.get("profit_factor"),
            "max_drawdown_r":trade.get("max_drawdown_r"),
            "sides":{
                "BUY":_fold_side(item,"BUY"),
                "SELL":_fold_side(item,"SELL"),
            },
        })
    importance=[]
    for item in (metrics.get("feature_importance") or [])[:25]:
        importance.append({
            "feature":item.get("feature"),
            "importance":item.get("importance"),
        })
    return {
        "status":row[0],
        "version":row[1],
        "dataset_rows":row[2],
        "error":row[3],
        "recorded_at":_iso(row[4]),
        "rr":metrics.get("rr"),
        "trading":{
            "trades":trading.get("trades"),
            "win_rate":trading.get("win_rate"),
            "expectancy_r":trading.get("expectancy_r"),
            "profit_factor":trading.get("profit_factor"),
            "max_drawdown_r":trading.get("max_drawdown_r"),
        },
        "quality_gate":{
            "status":gate.get("status"),
            "passed":gate.get("passed"),
            "failed_checks":[name for name,item in checks.items() if not item.get("passed",False)],
            "checks":checks,
        },
        "temporal_stability":{
            "status":temporal.get("status"),
            "passed":temporal.get("passed"),
            "summary":temporal.get("summary"),
            "high_winrate_stability":temporal.get("high_winrate_stability"),
            "directional_stability":temporal.get("directional_stability"),
            "folds":fold_rows,
        },
        "feature_importance":importance,
        "sides":{
            "BUY":_side_summary(metrics,"BUY"),
            "SELL":_side_summary(metrics,"SELL"),
        },
    }


def main():
    url=os.environ["DATABASE_URL"]
    with psycopg.connect(url,connect_timeout=8) as conn:
        conn.autocommit=True
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT timeframe, COUNT(*), MIN(timestamp), MAX(timestamp)
                FROM market_candles
                WHERE symbol='XAUUSD'
                GROUP BY timeframe
                ORDER BY timeframe
                """
            )
            history={
                tf:{"rows":int(rows),"first_timestamp":_iso(first),"last_timestamp":_iso(last)}
                for tf,rows,first,last in cur.fetchall()
            }
            payload={
                "xau_history":history,
                "training":{
                    "XAUUSD":_training(cur,"XAUUSD"),
                    "BTCUSD":_training(cur,"BTCUSD"),
                },
            }
    print("HISTORY_PROBE="+json.dumps(payload,default=str),flush=True)


if __name__=="__main__":
    main()
