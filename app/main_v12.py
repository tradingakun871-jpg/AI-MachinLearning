from pathlib import Path

from fastapi.responses import Response

import app.main as base
from live.inference_v12 import LiveInference
from ml.auto_train_v123 import (
    BTC_DEEP_HISTORY_DAYS,
    XAU_DEEP_HISTORY_REQUIRED,
    TrainingManager,
)
from ml.features import FEATURE_COLUMNS


VERSION="0.12.4"
MODEL_VERSION="v0.12.4"
FEATURE_SCHEMA="SMC_PA_ORDERFLOW_REGIME_V1"

# New artifact namespace: V0.12.4 keeps the SMC + PA + Order Flow schema from
# V0.12.3, but requires deeper history and calibration-subwindow context stability.
base.VERSION=VERSION
base.trainer=TrainingManager(base.shadow,version=MODEL_VERSION)
base.shadow.inference=LiveInference(threshold=.34,rr=2.0)


def model_available(symbol,version=MODEL_VERSION):
    return Path("artifacts/models")/symbol.upper()/version/"model.joblib"


base.model_available=model_available
base.app.version=VERSION
base.app.title="AI Market Intelligence Agent V0.12.4 Deep History + Stable Context"
app=base.app


@app.get("/api/training/summary")
def v124_training_summary():
    out={}
    for symbol in ("XAUUSD","BTCUSD"):
        state=base.trainer.status().get(symbol,{})
        metrics=state.get("metrics") or {}
        quality=state.get("quality") or {}
        trading=metrics.get("trading") or {}
        side_models=metrics.get("side_models") or {}
        temporal=metrics.get("temporal_stability") or {}
        out[symbol]={
            "status":state.get("status"),
            "version":state.get("version"),
            "feature_schema":FEATURE_SCHEMA,
            "feature_count":len(FEATURE_COLUMNS),
            "orderflow_policy":"CONDITIONAL_PROXY_AWARE",
            "context_policy":"CALIBRATION_SUBWINDOW_STABILITY",
            "dataset_rows":state.get("dataset_rows"),
            "history":metrics.get("history") or state.get("history"),
            "required_history":(
                XAU_DEEP_HISTORY_REQUIRED if symbol=="XAUUSD"
                else {"days":BTC_DEEP_HISTORY_DAYS}
            ),
            "error":state.get("error"),
            "rr":metrics.get("rr"),
            "quality":quality.get("status"),
            "quality_passed":bool(quality.get("passed",False)),
            "failed_checks":[
                name for name,item in (quality.get("checks") or {}).items()
                if not item.get("passed",False)
            ],
            "trading":{
                "trades":trading.get("trades"),
                "win_rate":trading.get("win_rate"),
                "net_r":trading.get("net_r"),
                "expectancy_r":trading.get("expectancy_r"),
                "profit_factor":trading.get("profit_factor"),
                "max_drawdown_r":trading.get("max_drawdown_r"),
            },
            "coverage":quality.get("coverage"),
            "temporal_stability":temporal.get("status"),
            "temporal_summary":temporal.get("summary"),
            "high_winrate_stability":temporal.get("high_winrate_stability"),
            "sides":{
                side:{
                    "threshold_mode":((side_models.get(side) or {}).get("threshold_policy") or {}).get("mode"),
                    "stable_context_counts":((side_models.get(side) or {}).get("threshold_policy") or {}).get("stable_context_counts"),
                    "recovery_used":((side_models.get(side) or {}).get("threshold_policy") or {}).get("recovery_used"),
                    "hybrid_mode":((side_models.get(side) or {}).get("hybrid_gate") or {}).get("mode"),
                    "meta_mode":((side_models.get(side) or {}).get("meta_precision") or {}).get("mode"),
                    "test_selected_rows":(side_models.get(side) or {}).get("test_selected_rows"),
                    "test_coverage":(side_models.get(side) or {}).get("test_coverage"),
                }
                for side in ("BUY","SELL")
            },
        }
    return out


@app.middleware("http")
async def v124_runtime_labels(request,call_next):
    response=await call_next(request)
    content_type=response.headers.get("content-type","")

    # Legacy base endpoint contains an old literal model version. Correct the
    # presentation layer without changing the underlying research state.
    if request.url.path=="/api/live/status" and "application/json" in content_type:
        body=b""
        async for chunk in response.body_iterator:
            body+=chunk
        text=body.decode("utf-8",errors="replace")
        text=text.replace("v0.11.4",MODEL_VERSION)
        headers=dict(response.headers)
        headers.pop("content-length",None)
        return Response(
            content=text,
            status_code=response.status_code,
            headers=headers,
            media_type="application/json",
        )

    if request.url.path!="/" or "text/html" not in content_type:
        return response

    body=b""
    async for chunk in response.body_iterator:
        body+=chunk
    text=body.decode("utf-8",errors="replace")
    for old in ("V0.11.4","V0.12.0","V0.12.1","V0.12.2","V0.12.3"):
        text=text.replace(old,"V0.12.4")
    text=text.replace(
        "Hybrid BUY/SELL: LightGBM + XGBoost + Linear Regression + Offline Q-Learning.",
        "Deep-history RR 1:2: ML + SMC + Price Action + conditional Order Flow + Stable Regime/Session Context."
    )
    text=text.replace(
        "90-day Coinbase training with linear expected-R and offline reinforcement-learning confirmation.",
        "180-day Coinbase deep-history training with causal SMC, Price Action, proxy-aware Order Flow and stable-context validation."
    )
    text=text.replace(
        "90-day Coinbase training with causal SMC, Price Action, proxy-aware Order Flow, regime context and strict holdout validation.",
        "180-day Coinbase deep-history training with causal SMC, Price Action, proxy-aware Order Flow, four-fold temporal diagnostics and stable-context validation."
    )
    text=text.replace(
        "High-winrate RR 1:2: ML + SMC + Regime + Price Action + conditional Order Flow + Meta Precision.",
        "Deep-history RR 1:2: SMC + Price Action + conditional Order Flow + Stable Context + Meta Precision."
    )
    text=text.replace(
        "High-winrate RR 1:2: ML + SMC + Regime + Price Action + conditional Order Flow + Adaptive Coverage.",
        "Deep-history RR 1:2: SMC + Price Action + conditional Order Flow + Stable Context + Adaptive Coverage."
    )
    headers=dict(response.headers)
    headers.pop("content-length",None)
    return Response(
        content=text,
        status_code=response.status_code,
        headers=headers,
        media_type="text/html",
    )
