from fastapi import FastAPI,HTTPException,Depends
from fastapi.responses import HTMLResponse
from app.schemas import AnalysisRequest,AnalysisResponse
from app.engine import MarketAgent
from app.security import verify_bridge_token
from live.service import ShadowService

app=FastAPI(title="AI Market Intelligence Agent",version="0.7.0"); agent=MarketAgent(); shadow=ShadowService()

@app.get("/health")
def health(): return {"status":"ok","mode":"SHADOW_RESEARCH","version":"0.7.0"}
@app.get("/api/status")
def status(): return {**agent.status(),"live_mode":"SHADOW_RESEARCH","version":"0.7.0"}
@app.post("/api/analyze",response_model=AnalysisResponse)
def analyze(req:AnalysisRequest): return agent.analyze(req)
@app.post("/api/market/candle")
def candle(payload:dict): return agent.ingest(payload)
@app.post("/api/live/candle",dependencies=[Depends(verify_bridge_token)])
def live_candle(payload:dict):
    for k in ("symbol","timeframe","timestamp","open","high","low","close"):
        if k not in payload: raise HTTPException(422,f"missing field: {k}")
    if payload["timeframe"].upper() not in ("M3","M5","M15"): raise HTTPException(422,"timeframe must be M3, M5 or M15")
    candle={k:v for k,v in payload.items() if k not in ("symbol","timeframe")}; return shadow.ingest(payload["symbol"],payload["timeframe"],candle)
@app.get("/api/live/signals")
def live_signals(limit:int=100): return {"signals":shadow.recent(limit)}
@app.get("/api/signals")
def signals(): return {"signals":agent.signals[-100:]}
@app.get("/",response_class=HTMLResponse)
def dashboard(): return """<!doctype html><html><head><title>AI Market Agent V0.7</title><style>body{font-family:system-ui;background:#0b1020;color:#e8ecf5;margin:40px}.card{background:#151c31;padding:24px;border-radius:16px;max-width:820px}code{color:#8ee3ff}.ok{color:#77e29a}</style></head><body><div class='card'><h1>AI Market Intelligence Agent V0.7</h1><p class='ok'>API ONLINE</p><p>Mode: <b>SHADOW RESEARCH — NO BROKER ORDERS</b></p><p>MT5 Bridge: authenticated M3/M5/M15 closed-candle feed.</p><p>Pipeline: MT5 → M3/M5/M15 → SMC → Regime → LightGBM/XGBoost → Calibration → Expected R → Shadow Journal.</p><p>Docs: <code>/docs</code></p></div></body></html>"""
