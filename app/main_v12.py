from pathlib import Path

from fastapi.responses import Response

import app.main as base
from live.inference_v12 import LiveInference
from ml.auto_train_v122 import TrainingManager
from ml.features import FEATURE_COLUMNS


VERSION="0.12.3"
MODEL_VERSION="v0.12.3"
FEATURE_SCHEMA="SMC_PA_ORDERFLOW_REGIME_V1"

# Replace the legacy runtime components before FastAPI lifespan starts.
# A new artifact version intentionally prevents restoring pre-PA/order-flow
# V0.12.2 model files after this feature-schema change.
base.VERSION=VERSION
base.trainer=TrainingManager(base.shadow,version=MODEL_VERSION)
base.shadow.inference=LiveInference(threshold=.34,rr=2.0)


def model_available(symbol,version=MODEL_VERSION):
    return Path("artifacts/models")/symbol.upper()/version/"model.joblib"


base.model_available=model_available
base.app.version=VERSION
base.app.title="AI Market Intelligence Agent V0.12.3 SMC + Price Action + Order Flow"
app=base.app


@app.get("/api/training/summary")
def v123_training_summary():
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
            "dataset_rows":state.get("dataset_rows"),
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
async def v12_dashboard_labels(request,call_next):
    response=await call_next(request)
    if request.url.path!="/":
        return response
    content_type=response.headers.get("content-type","")
    if "text/html" not in content_type:
        return response
    body=b""
    async for chunk in response.body_iterator:
        body+=chunk
    text=body.decode("utf-8",errors="replace")
    text=text.replace("V0.11.4","V0.12.3")
    text=text.replace("V0.12.0","V0.12.3")
    text=text.replace("V0.12.1","V0.12.3")
    text=text.replace("V0.12.2","V0.12.3")
    text=text.replace(
        "Hybrid BUY/SELL: LightGBM + XGBoost + Linear Regression + Offline Q-Learning.",
        "High-winrate RR 1:2: ML + SMC + Regime + Price Action + conditional Order Flow + Meta Precision."
    )
    text=text.replace(
        "90-day Coinbase training with linear expected-R and offline reinforcement-learning confirmation.",
        "90-day Coinbase training with causal SMC, Price Action, proxy-aware Order Flow, regime context and strict holdout validation."
    )
    text=text.replace(
        "High-winrate RR 1:2: LightGBM + XGBoost + Linear Regression + Offline Q-Learning + Meta Precision.",
        "High-winrate RR 1:2: ML + SMC + Regime + Price Action + conditional Order Flow + Meta Precision."
    )
    text=text.replace(
        "High-winrate RR 1:2: ML + Linear Regression + Offline Q-Learning + Meta Precision + Adaptive Coverage.",
        "High-winrate RR 1:2: ML + SMC + Regime + Price Action + conditional Order Flow + Adaptive Coverage."
    )
    headers=dict(response.headers)
    headers.pop("content-length",None)
    return Response(
        content=text,
        status_code=response.status_code,
        headers=headers,
        media_type="text/html",
    )
