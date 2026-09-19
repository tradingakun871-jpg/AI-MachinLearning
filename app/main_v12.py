from pathlib import Path

from fastapi.responses import Response

import app.main as base
from live.inference_v12 import LiveInference
from ml.auto_train_v12 import TrainingManager


VERSION="0.12.1"

# Replace the legacy runtime components before FastAPI lifespan starts.
base.VERSION=VERSION
base.trainer=TrainingManager(base.shadow,version="v0.12.1")
base.shadow.inference=LiveInference(threshold=.35,rr=2.0)


def model_available(symbol,version="v0.12.1"):
    return Path("artifacts/models")/symbol.upper()/version/"model.joblib"


base.model_available=model_available
base.app.version=VERSION
base.app.title="AI Market Intelligence Agent V0.12.1 High Winrate RR 1:2"
app=base.app


@app.get("/api/training/summary")
def v121_training_summary():
    out={}
    for symbol in ("XAUUSD","BTCUSD"):
        state=base.trainer.status().get(symbol,{})
        metrics=state.get("metrics") or {}
        quality=state.get("quality") or {}
        trading=metrics.get("trading") or {}
        side_models=metrics.get("side_models") or {}
        out[symbol]={
            "status":state.get("status"),
            "version":state.get("version"),
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
            "temporal_stability":(metrics.get("temporal_stability") or {}).get("status"),
            "meta_precision":{
                side:((side_models.get(side) or {}).get("meta_precision") or {}).get("mode")
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
    text=text.replace("V0.11.4","V0.12.1")
    text=text.replace("V0.12.0","V0.12.1")
    text=text.replace(
        "Hybrid BUY/SELL: LightGBM + XGBoost + Linear Regression + Offline Q-Learning.",
        "High-winrate RR 1:2: LightGBM + XGBoost + Linear Regression + Offline Q-Learning + Meta Precision."
    )
    text=text.replace(
        "90-day Coinbase training with linear expected-R and offline reinforcement-learning confirmation.",
        "90-day Coinbase training relabeled for TP 2R before SL 1R with confidence-aware high-precision filtering."
    )
    text=text.replace(
        "High-winrate RR 1:2: LightGBM + XGBoost + Linear Regression + Offline Q-Learning.",
        "High-winrate RR 1:2: LightGBM + XGBoost + Linear Regression + Offline Q-Learning + Meta Precision."
    )
    headers=dict(response.headers)
    headers.pop("content-length",None)
    return Response(
        content=text,
        status_code=response.status_code,
        headers=headers,
        media_type="text/html",
    )
