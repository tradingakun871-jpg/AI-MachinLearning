from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.schemas import AnalysisRequest, AnalysisResponse
from app.engine import MarketAgent

app = FastAPI(title="AI Market Intelligence Agent", version="0.1.0")
agent = MarketAgent()

@app.get("/health")
def health():
    return {"status": "ok", "mode": agent.mode}

@app.get("/api/status")
def status():
    return agent.status()

@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze(req: AnalysisRequest):
    return agent.analyze(req)

@app.post("/api/market/candle")
def candle(payload: dict):
    return agent.ingest(payload)

@app.get("/api/signals")
def signals():
    return {"signals": agent.signals[-100:]}

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """<!doctype html><html><head><title>AI Market Agent</title><style>body{font-family:system-ui;background:#0b1020;color:#e8ecf5;margin:40px}.card{background:#151c31;padding:24px;border-radius:16px;max-width:760px}code{color:#8ee3ff}.ok{color:#77e29a}</style></head><body><div class='card'><h1>AI Market Intelligence Agent</h1><p class='ok'>API ONLINE</p><p>Default execution mode: <b>SHADOW</b></p><p>Pipeline: Market Data → Features → Structure/SMC → ML Probability → Expectancy → Risk → Decision.</p><p>API documentation: <code>/docs</code></p></div></body></html>"""
