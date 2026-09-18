from pathlib import Path

from fastapi.responses import Response

import app.main as base
from live.inference_v12 import LiveInference
from ml.auto_train_v12 import TrainingManager


VERSION="0.12.0"

# Replace the V0.11.4 runtime components before FastAPI lifespan starts.
base.VERSION=VERSION
base.trainer=TrainingManager(base.shadow,version="v0.12.0")
base.shadow.inference=LiveInference(threshold=.35,rr=2.0)


def model_available(symbol,version="v0.12.0"):
    return Path("artifacts/models")/symbol.upper()/version/"model.joblib"


base.model_available=model_available
base.app.version=VERSION
base.app.title="AI Market Intelligence Agent V0.12 High Winrate RR 1:2"
app=base.app


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
    text=text.replace("V0.11.4","V0.12.0")
    text=text.replace(
        "Hybrid BUY/SELL: LightGBM + XGBoost + Linear Regression + Offline Q-Learning.",
        "High-winrate RR 1:2: LightGBM + XGBoost + Linear Regression + Offline Q-Learning."
    )
    text=text.replace(
        "90-day Coinbase training with linear expected-R and offline reinforcement-learning confirmation.",
        "90-day Coinbase training relabeled for TP 2R before SL 1R with high-precision filtering."
    )
    headers=dict(response.headers)
    headers.pop("content-length",None)
    return Response(
        content=text,
        status_code=response.status_code,
        headers=headers,
        media_type="text/html",
    )
