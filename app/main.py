import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime,timezone
from pathlib import Path

from fastapi import FastAPI,HTTPException,Depends
from fastapi.responses import HTMLResponse

from app.schemas import AnalysisRequest,AnalysisResponse
from app.engine import MarketAgent
from app.security import verify_bridge_token
from app.database import database_health, init_database, market_history_status, upsert_market_candles
from live.service import ShadowService
from data.coinbase import CoinbaseBTCFeed
from ml.auto_train import TrainingManager

VERSION="0.11.2"
agent=MarketAgent()
shadow=ShadowService()
coinbase=CoinbaseBTCFeed(shadow,poll_seconds=int(os.getenv("COINBASE_POLL_SECONDS","60")))
trainer=TrainingManager(shadow,version="v0.11.2")

@asynccontextmanager
async def lifespan(app:FastAPI):
    try:
        init_database()
    except Exception:
        pass
    await coinbase.start()
    await trainer.start_btc(force=False)
    await trainer.start_xau(force=False)
    yield
    if trainer.task and not trainer.task.done():
        trainer.task.cancel()
    if trainer.xau_task and not trainer.xau_task.done():
        trainer.xau_task.cancel()
    await coinbase.stop()

app=FastAPI(title="AI Market Intelligence Agent",version=VERSION,lifespan=lifespan)

def bridge_configured():
    token=os.getenv("BRIDGE_TOKEN","")
    return bool(token and token!="change-me")

def model_available(symbol,version="v0.11.2"):
    return Path("artifacts/models")/symbol.upper()/version/"model.joblib"

@app.get("/health")
def health():
    return {"status":"ok","mode":"SHADOW_RESEARCH","version":VERSION}

@app.get("/api/status")
def status():
    return {**agent.status(),"live_mode":"SHADOW_RESEARCH","version":VERSION}

@app.get("/api/coinbase/status")
def coinbase_status():
    return coinbase.status()

@app.get("/api/database/status")
def db_status():
    return database_health()

@app.get("/api/history/status")
def history_status():
    return {"XAUUSD":market_history_status("XAUUSD")}

@app.get("/api/training/status")
def training_status():
    return trainer.status()

@app.get("/api/model/quality")
def model_quality():
    return {
        symbol:{
            "status":trainer.status().get(symbol,{}).get("status"),
            "version":trainer.status().get(symbol,{}).get("version"),
            "quality":trainer.status().get(symbol,{}).get("quality"),
            "feature_importance":((trainer.status().get(symbol,{}).get("metrics") or {}).get("feature_importance")),
            "learned_filters":((trainer.status().get(symbol,{}).get("metrics") or {}).get("learned_filters")),
            "threshold_policy":((trainer.status().get(symbol,{}).get("metrics") or {}).get("threshold_policy")),\n            "side_models":((trainer.status().get(symbol,{}).get("metrics") or {}).get("side_models")),\n            "probability_diagnostics":((trainer.status().get(symbol,{}).get("metrics") or {}).get("probability_diagnostics")),
        }
        for symbol in ("XAUUSD","BTCUSD")
    }

@app.post("/api/training/btc/retrain")
async def retrain_btc():
    await trainer.start_btc(force=True)
    return {"accepted":True,"status":trainer.status()["BTCUSD"]["status"]}

@app.post("/api/training/xau/start")
async def train_xau():
    await trainer.start_xau(force=True)
    return {"accepted":True,"status":trainer.status()["XAUUSD"]["status"],"history":market_history_status("XAUUSD")}

@app.post("/api/coinbase/btc/sync")
async def coinbase_sync():
    try:
        return await coinbase.sync()
    except Exception as exc:
        raise HTTPException(502,f"Coinbase sync failed: {exc}")

@app.get("/api/live/status")
def live_status():
    payload=shadow.status()
    model_status={}
    for symbol in ("XAUUSD","BTCUSD"):
        path=model_available(symbol)
        quality=trainer.status().get(symbol,{}).get("quality") or {}
        passed=bool(quality.get("passed",False))
        model_status[symbol]={
            "available":path.exists(),
            "version":"v0.11.2" if path.exists() else None,
            "quality_passed":passed,
            "state":("READY_QUALITY_PASSED" if passed else "READY_QUALITY_BLOCKED") if path.exists() else "MODEL_NOT_AVAILABLE",
        }
    recent=shadow.recent(1)
    return {
        "status":"ONLINE",
        "version":VERSION,
        "mode":"SHADOW_RESEARCH",
        "broker_orders_enabled":False,
        "sources":{
            "XAUUSD":"MT5",
            "BTCUSD":"Coinbase Exchange BTC-USD",
        },
        "bridge":{
            "configured":bridge_configured(),
            "endpoint":"/api/live/candle",
            "symbols":["XAUUSD"],
            "timeframes":["M3","M5","M15"],
        },
        "coinbase":coinbase.status(),
        "database":database_health(),
        "history":{"XAUUSD":market_history_status("XAUUSD")},
        "training":trainer.status(),
        "models":model_status,
        "buffer":payload,
        "last_signal":recent[-1] if recent else None,
        "server_time":datetime.now(timezone.utc).isoformat(),
        "performance":{
            "BTCUSD":((trainer.status().get("BTCUSD",{}).get("metrics") or {}).get("trading")),
            "XAUUSD":((trainer.status().get("XAUUSD",{}).get("metrics") or {}).get("trading")),
            "state":"MODEL_TEST_METRICS_ONLY_NOT_LIVE_OUTCOMES",
        },
    }

@app.post("/api/analyze",response_model=AnalysisResponse)
def analyze(req:AnalysisRequest):
    return agent.analyze(req)

@app.post("/api/market/candle")
def candle(payload:dict):
    return agent.ingest(payload)

@app.post("/api/history/batch",dependencies=[Depends(verify_bridge_token)])
async def history_batch(payload:dict):
    if str(payload.get("symbol","")).upper()!="XAUUSD":
        raise HTTPException(422,"historical backfill currently supports XAUUSD only")
    timeframe=str(payload.get("timeframe","")).upper()
    if timeframe not in ("M3","M5","M15"):
        raise HTTPException(422,"timeframe must be M3, M5 or M15")
    candles=payload.get("candles")
    if not isinstance(candles,list) or not candles:
        raise HTTPException(422,"candles must be a non-empty list")
    if len(candles)>500:
        raise HTTPException(422,"maximum 500 candles per batch")
    required=("timestamp","open","high","low","close")
    for i,candle in enumerate(candles):
        if not isinstance(candle,dict) or any(k not in candle for k in required):
            raise HTTPException(422,f"invalid candle at index {i}")
    try:
        stored=await asyncio.to_thread(upsert_market_candles,"XAUUSD",timeframe,candles)
    except Exception as exc:
        raise HTTPException(500,f"history persistence failed: {exc}")
    return {"accepted":True,"stored":stored,"timeframe":timeframe,"history":market_history_status("XAUUSD")}

@app.post("/api/history/complete",dependencies=[Depends(verify_bridge_token)])
async def history_complete(payload:dict):
    if str(payload.get("symbol","")).upper()!="XAUUSD":
        raise HTTPException(422,"historical completion currently supports XAUUSD only")
    history=market_history_status("XAUUSD")
    await trainer.start_xau(force=True)
    return {"accepted":True,"status":trainer.status()["XAUUSD"]["status"],"history":history}

@app.post("/api/live/candle",dependencies=[Depends(verify_bridge_token)])
def live_candle(payload:dict):
    for k in ("symbol","timeframe","timestamp","open","high","low","close"):
        if k not in payload:
            raise HTTPException(422,f"missing field: {k}")
    if payload["symbol"].upper()!="XAUUSD":
        raise HTTPException(422,"MT5 live endpoint is reserved for XAUUSD; BTCUSD source is Coinbase")
    if payload["timeframe"].upper() not in ("M3","M5","M15"):
        raise HTTPException(422,"timeframe must be M3, M5 or M15")
    candle={k:v for k,v in payload.items() if k not in ("symbol","timeframe")}
    return shadow.ingest(payload["symbol"],payload["timeframe"],candle)

@app.get("/api/live/signals")
def live_signals(limit:int=100):
    return {"signals":shadow.recent(limit)}

@app.get("/api/signals")
def signals():
    return {"signals":agent.signals[-100:]}

@app.get("/",response_class=HTMLResponse)
def dashboard():
    return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>AI Market Intelligence V0.11.2</title>
<style>
:root{--bg:#080d1b;--panel:#11192b;--panel2:#172137;--text:#eef3ff;--muted:#8fa0bd;--line:#26324c;--green:#49e59a;--amber:#ffcb66;--red:#ff7184;--blue:#68a7ff}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(180deg,#070b16,#0a1020);color:var(--text);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1260px;margin:auto;padding:24px}.top{display:flex;gap:16px;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;margin-bottom:18px}
h1{font-size:28px;margin:0 0 6px}.sub{color:var(--muted)}.badge{display:inline-flex;align-items:center;gap:7px;padding:7px 10px;border:1px solid var(--line);border-radius:999px;background:#0d1528}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 14px var(--green)}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}
.card{background:rgba(17,25,43,.94);border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:0 12px 30px rgba(0,0,0,.18)}
.span3{grid-column:span 3}.span4{grid-column:span 4}.span6{grid-column:span 6}.span12{grid-column:span 12}
.k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.v{font-size:23px;font-weight:750;margin-top:6px}.small{font-size:12px;color:var(--muted)}
.ok{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}
.tf{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:13px}.tf div{background:#0c1426;border:1px solid var(--line);border-radius:10px;padding:10px}
table{width:100%;border-collapse:collapse;margin-top:10px}th,td{text-align:left;padding:10px 8px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-weight:600;font-size:12px}
.scroll{overflow:auto}.footer{color:var(--muted);font-size:12px;margin:18px 2px}.pill{padding:4px 8px;border-radius:999px;background:#0d1528;border:1px solid var(--line);font-size:12px}
a{color:#8bb8ff;text-decoration:none}
.train-head{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.train-title{font-size:18px;font-weight:800;margin-top:3px}.train-actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.btn{border:1px solid var(--line);background:#0d1528;color:#dce9ff;border-radius:10px;padding:8px 11px;cursor:pointer;font-weight:700}
.btn:hover{border-color:#4d8cff;box-shadow:0 0 0 1px rgba(77,140,255,.16),0 0 16px rgba(77,140,255,.12)}
.train-progress-wrap{display:grid;grid-template-columns:1fr auto;gap:12px;align-items:center;margin:4px 0 16px}
.train-progress{height:14px;border:1px solid var(--line);border-radius:999px;overflow:hidden;background:#09111f;position:relative}
.train-progress-fill{height:100%;width:0;border-radius:999px;background:linear-gradient(90deg,#4f8dff,#45d8ff,#49e59a);box-shadow:0 0 18px rgba(79,141,255,.35);transition:width .6s ease;position:relative}
.train-progress-fill:after{content:"";position:absolute;top:0;bottom:0;width:30%;left:-35%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.5),transparent);animation:trainSweep 1.6s linear infinite}
.train-pct{font-weight:800;color:#b9d7ff;min-width:46px;text-align:right}
.train-steps{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;position:relative}
.train-step{min-height:122px;padding:13px;border-radius:14px;border:1px solid var(--line);background:#0b1425;position:relative;overflow:hidden;transition:.3s ease}
.train-step:before{content:"";position:absolute;inset:0;background:linear-gradient(110deg,transparent 25%,rgba(91,154,255,.06) 45%,transparent 65%);transform:translateX(-120%)}
.train-step.active{border-color:#5d9dff;box-shadow:0 0 0 1px rgba(93,157,255,.2),0 0 24px rgba(54,121,255,.12);transform:translateY(-2px)}
.train-step.active:before{animation:trainShimmer 1.7s linear infinite}
.train-step.done{border-color:rgba(73,229,154,.55);background:linear-gradient(180deg,#0c1925,#0b1622)}
.train-step.failed{border-color:rgba(255,113,132,.6);box-shadow:0 0 18px rgba(255,113,132,.12)}
.train-step .step-no{width:31px;height:31px;border-radius:50%;display:grid;place-items:center;border:1px solid #35517b;background:#101d33;font-weight:800;margin-bottom:10px;color:#b8d4ff}
.train-step.done .step-no{color:#07130e;background:var(--green);border-color:var(--green);box-shadow:0 0 16px rgba(73,229,154,.35)}
.train-step.active .step-no{border-color:#66a7ff;box-shadow:0 0 16px rgba(102,167,255,.28);animation:trainPulse 1.4s ease-in-out infinite}
.step-name{font-weight:800;font-size:13px;letter-spacing:.02em}.step-desc{color:var(--muted);font-size:11px;margin-top:5px;line-height:1.35}
.train-flow{position:absolute;width:7px;height:7px;border-radius:50%;background:#65b6ff;box-shadow:0 0 14px #65b6ff;top:10px;right:10px;opacity:0}
.train-step.active .train-flow{opacity:1;animation:trainOrbit 1.8s linear infinite}
.train-meta{margin-top:14px;display:grid;grid-template-columns:repeat(5,1fr);gap:9px}
.meta-box{background:#0b1425;border:1px solid var(--line);border-radius:11px;padding:10px}.meta-box .mv{margin-top:4px;font-weight:700;word-break:break-word}
.ready-burst{display:none;align-items:center;gap:10px;margin-top:13px;padding:11px 12px;border:1px solid rgba(73,229,154,.35);border-radius:12px;background:rgba(73,229,154,.06)}
.ready-burst.show{display:flex}.ready-check{width:34px;height:34px;border-radius:50%;display:grid;place-items:center;background:rgba(73,229,154,.15);color:var(--green);font-weight:900;box-shadow:0 0 22px rgba(73,229,154,.22);animation:readyPulse 1.5s ease-in-out infinite}

.pipeline-board{margin-top:14px;position:relative;padding:14px;border:1px solid rgba(104,167,255,.16);border-radius:16px;background:radial-gradient(circle at 50% 0%,rgba(68,123,255,.08),transparent 38%),#0a1222;overflow:hidden}
.pipeline-grid{display:grid;grid-template-columns:1.25fr .9fr 1.15fr .85fr 1.15fr .85fr 1fr;gap:12px;align-items:stretch;position:relative;z-index:2}
.pipe-stage{min-height:148px;border:1px solid var(--line);background:linear-gradient(180deg,#0f1a2d,#0a1323);border-radius:15px;padding:14px;position:relative;overflow:hidden;display:flex;flex-direction:column;justify-content:space-between}
.pipe-stage:after{content:"";position:absolute;inset:-40%;background:radial-gradient(circle,rgba(92,160,255,.09),transparent 32%);transform:translateX(-45%);animation:pipeGlow 5.8s ease-in-out infinite}
.pipe-stage.ready{border-color:rgba(73,229,154,.42);box-shadow:0 0 18px rgba(73,229,154,.08)}
.pipe-stage.wait{border-color:rgba(255,203,102,.38)}
.pipe-stage .stage-icon{width:39px;height:39px;border-radius:12px;display:grid;place-items:center;background:#101e35;border:1px solid #31517a;font-size:18px;box-shadow:inset 0 0 14px rgba(92,160,255,.08)}
.pipe-stage.ready .stage-icon{border-color:rgba(73,229,154,.5);box-shadow:0 0 14px rgba(73,229,154,.12)}
.stage-name{font-size:13px;font-weight:850;letter-spacing:.025em;margin-top:10px}
.stage-sub{font-size:11px;color:var(--muted);line-height:1.35;margin-top:5px}
.stage-state{font-size:10px;display:inline-flex;align-items:center;gap:6px;margin-top:9px;color:#b7c8e3}
.stage-led{width:7px;height:7px;border-radius:50%;background:var(--blue);box-shadow:0 0 12px var(--blue);animation:ledPulse 1.35s ease-in-out infinite}
.pipe-stage.ready .stage-led{background:var(--green);box-shadow:0 0 12px var(--green)}
.pipe-stage.wait .stage-led{background:var(--amber);box-shadow:0 0 12px var(--amber)}
.pipe-link{position:absolute;top:50%;height:2px;background:linear-gradient(90deg,rgba(65,135,255,.08),rgba(87,174,255,.65),rgba(65,135,255,.08));z-index:1;overflow:visible}
.pipe-link:before,.pipe-link:after{content:"";position:absolute;top:50%;width:8px;height:8px;border-radius:50%;background:#66b4ff;box-shadow:0 0 16px #66b4ff;transform:translate(-50%,-50%);animation:pipeParticle 2.15s linear infinite}
.pipe-link:after{animation-delay:1.05s}
.pl1{left:15.2%;width:4.3%}.pl2{left:27.9%;width:4.3%}.pl3{left:44.1%;width:4.3%}.pl4{left:56.6%;width:4.3%}.pl5{left:72.6%;width:4.3%}.pl6{left:85%;width:4.3%}
.pipeline-streams{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:9px}
.source-chip{font-size:10px;padding:6px 7px;border:1px solid var(--line);border-radius:9px;background:#0a1425;display:flex;align-items:center;gap:6px}
.source-chip .mini-led{width:6px;height:6px;border-radius:50%;background:var(--amber);box-shadow:0 0 9px var(--amber)}
.source-chip.live .mini-led{background:var(--green);box-shadow:0 0 9px var(--green)}
.stage-bars{display:flex;align-items:flex-end;gap:4px;height:28px;margin-top:8px}.stage-bars i{display:block;width:5px;border-radius:4px;background:linear-gradient(180deg,#6ec5ff,#4f7dff);animation:barDance 1.4s ease-in-out infinite}.stage-bars i:nth-child(1){height:35%}.stage-bars i:nth-child(2){height:72%;animation-delay:.2s}.stage-bars i:nth-child(3){height:48%;animation-delay:.4s}.stage-bars i:nth-child(4){height:90%;animation-delay:.6s}.stage-bars i:nth-child(5){height:60%;animation-delay:.8s}
.signal-orb{width:47px;height:47px;border-radius:50%;border:2px solid rgba(73,229,154,.55);display:grid;place-items:center;margin-top:8px;box-shadow:0 0 22px rgba(73,229,154,.18),inset 0 0 18px rgba(73,229,154,.08);animation:orbPulse 1.7s ease-in-out infinite}
.pipeline-legend{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap;margin-top:12px;position:relative;z-index:2}
.pipe-live{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.pipe-live span{font-size:11px;color:var(--muted)}
.flow-caption{font-size:11px;color:#7892b9;letter-spacing:.08em;text-transform:uppercase}
@keyframes pipeParticle{0%{left:0;opacity:0;transform:translate(-50%,-50%) scale(.5)}10%{opacity:1}90%{opacity:1}100%{left:100%;opacity:0;transform:translate(-50%,-50%) scale(1.2)}}
@keyframes pipeGlow{0%,100%{transform:translateX(-45%)}50%{transform:translateX(45%)}}
@keyframes ledPulse{50%{opacity:.35;transform:scale(.82)}}
@keyframes barDance{50%{transform:scaleY(.55);opacity:.65}}
@keyframes orbPulse{50%{transform:scale(1.07);box-shadow:0 0 32px rgba(73,229,154,.28),inset 0 0 24px rgba(73,229,154,.12)}}
@media(max-width:1100px){.pipeline-grid{grid-template-columns:repeat(2,1fr)}.pipe-link{display:none}}
@media(max-width:700px){.pipeline-grid{grid-template-columns:1fr}}
@keyframes trainSweep{to{left:110%}}@keyframes trainShimmer{to{transform:translateX(120%)}}@keyframes trainPulse{50%{transform:scale(1.08);box-shadow:0 0 22px rgba(102,167,255,.45)}}@keyframes trainOrbit{50%{transform:translate(-18px,18px)}100%{transform:translate(0,0)}}@keyframes readyPulse{50%{transform:scale(1.06)}}@media(max-width:1100px){.train-steps{grid-template-columns:repeat(3,1fr)}.train-meta{grid-template-columns:repeat(2,1fr)}}@media(max-width:900px){.span3,.span4,.span6{grid-column:span 12}.wrap{padding:14px}h1{font-size:23px}.train-steps{grid-template-columns:1fr}.train-meta{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div><h1>AI Market Intelligence Agent V0.11.2</h1><div class="sub">XAUUSD: MT5 · BTCUSD: Coinbase BTC-USD · SMC · ML inference · Shadow journal</div></div>
    <div class="badge"><span class="dot"></span><span id="apiState">Checking API…</span></div>
  </div>

  <div class="grid">
    <div class="card span3"><div class="k">Mode</div><div class="v warn">SHADOW RESEARCH</div><div class="small">Broker orders are disabled.</div></div>
    <div class="card span3"><div class="k">XAU MT5 Bridge</div><div class="v" id="bridgeState">—</div><div class="small">XAUUSD M3/M5/M15 closed candles.</div></div>
    <div class="card span3"><div class="k">BTC Coinbase</div><div class="v" id="coinbaseState">—</div><div class="small">Public BTC-USD OHLCV feed.</div></div>
    <div class="card span3"><div class="k">Shadow Signals</div><div class="v" id="signalCount">0</div><div class="small">Research journal only.</div></div>

    <div class="card span6">
      <div class="k">XAUUSD · Source MT5</div><div class="v" id="xauPrice">WAITING FOR MT5</div>
      <div class="tf">
        <div><div class="k">M3</div><b id="xauM3">0 rows</b></div>
        <div><div class="k">M5</div><b id="xauM5">0 rows</b></div>
        <div><div class="k">M15</div><b id="xauM15">0 rows</b></div>
      </div>
      <p class="small" id="xauReady">Need 300 closed candles per timeframe before inference.</p>
    </div>

    <div class="card span6">
      <div class="k">BTCUSD · Source Coinbase BTC-USD</div><div class="v" id="btcPrice">CONNECTING TO COINBASE</div>
      <div class="tf">
        <div><div class="k">M3</div><b id="btcM3">0 rows</b></div>
        <div><div class="k">M5</div><b id="btcM5">0 rows</b></div>
        <div><div class="k">M15</div><b id="btcM15">0 rows</b></div>
      </div>
      <p class="small" id="btcReady">Coinbase M1 is aggregated to M3; M5/M15 use Coinbase candles.</p>
      <p class="small" id="coinbaseDetail">Waiting for first Coinbase sync…</p>
    </div>

    <div class="card span4"><div class="k">XAUUSD ML Model</div><div class="v" id="xauModel">—</div><div class="small">Separate BUY/SELL LightGBM + XGBoost models with feature pruning.</div></div>
    <div class="card span4"><div class="k">BTCUSD ML Model</div><div class="v" id="btcModel">—</div><div class="small">90-day Coinbase training, separate BUY/SELL models and contextual thresholds.</div></div>
    <div class="card span4"><div class="k">Performance</div><div class="v warn">N/A</div><div class="small">Win rate / PF / DD require validated outcomes.</div></div>


    <div class="card span12" id="btcTrainingCard">
      <div class="train-head">
        <div>
          <div class="k">BTCUSD ML MODEL TRAINING</div>
          <div class="train-title">Coinbase History → Feature Engineering → Regime → BUY/SELL Models → Context Policy → Quality Gate</div>
          <div class="small" id="trainSub">Live backend training state. Animation reflects the actual training pipeline.</div>
        </div>
        <div class="train-actions">
          <span class="pill" id="trainStatusPill">IDLE</span>
          <button class="btn" id="replayTrainingBtn" type="button">Replay animation</button>
        </div>
      </div>

      <div class="train-progress-wrap">
        <div class="train-progress"><div class="train-progress-fill" id="trainProgressFill"></div></div>
        <div class="train-pct" id="trainPct">0%</div>
      </div>

      <div class="train-steps">
        <div class="train-step" id="step0" data-num="1"><div class="train-flow"></div><div class="step-no">1</div><div class="step-name">DOWNLOADING HISTORY</div><div class="step-desc">Fetching closed BTC-USD candles from Coinbase.</div></div>
        <div class="train-step" id="step1" data-num="2"><div class="train-flow"></div><div class="step-no">2</div><div class="step-name">BUILDING DATASET</div><div class="step-desc">SMC features, MTF context, labels and purged splits.</div></div>
        <div class="train-step" id="step2" data-num="3"><div class="train-flow"></div><div class="step-no">3</div><div class="step-name">TRAINING REGIME</div><div class="step-desc">Learning market regime clusters from training-only data.</div></div>
        <div class="train-step" id="step3" data-num="4"><div class="train-flow"></div><div class="step-no">4</div><div class="step-name">BUY / SELL MODELS</div><div class="step-desc">Feature pruning, then separate LightGBM + XGBoost ensembles.</div></div>
        <div class="train-step" id="step4" data-num="5"><div class="train-flow"></div><div class="step-no">5</div><div class="step-name">CONTEXT + QUALITY</div><div class="step-desc">Regime×session thresholds, diagnostics and independent holdout gate.</div></div>
        <div class="train-step" id="step5" data-num="6"><div class="train-flow"></div><div class="step-no">6</div><div class="step-name">READY v0.11.2</div><div class="step-desc">Validated artifact available for shadow inference.</div></div>
      </div>

      <div class="train-meta">
        <div class="meta-box"><div class="k">STATUS</div><div class="mv" id="trainStatus">IDLE</div></div>
        <div class="meta-box"><div class="k">VERSION</div><div class="mv" id="trainVersion">v0.11</div></div>
        <div class="meta-box"><div class="k">DATASET ROWS</div><div class="mv" id="trainRows">0</div></div>
        <div class="meta-box"><div class="k">STARTED UTC</div><div class="mv" id="trainStarted">—</div></div>
        <div class="meta-box"><div class="k">FINISHED UTC</div><div class="mv" id="trainFinished">—</div></div>
      </div>

      <div class="ready-burst" id="readyBurst">
        <div class="ready-check">✓</div>
        <div><b>BTCUSD V0.11.2 training artifact is READY.</b><div class="small">Shadow signals remain blocked unless the independent quality gate passes.</div></div>
      </div>
    </div>

    <div class="card span12">
      <div class="k">Latest Shadow Signals</div>
      <div class="scroll"><table><thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Probability</th><th>Expected R</th><th>Regime</th><th>Model</th><th>Qualified</th></tr></thead><tbody id="signalRows"><tr><td colspan="8" class="small">No model signals yet.</td></tr></tbody></table></div>
    </div>

    <div class="card span12">
      <div class="train-head">
        <div>
          <div class="k">PIPELINE STATUS</div>
          <div class="train-title">Continuous Live Intelligence Flow</div>
          <div class="small">Animated continuously to show how market data moves through the research pipeline.</div>
        </div>
        <div class="badge"><span class="dot"></span><span>PIPELINE RUNNING</span></div>
      </div>

      <div class="pipeline-board">
        <div class="pipe-link pl1"></div><div class="pipe-link pl2"></div><div class="pipe-link pl3"></div>
        <div class="pipe-link pl4"></div><div class="pipe-link pl5"></div><div class="pipe-link pl6"></div>

        <div class="pipeline-grid">
          <div class="pipe-stage" id="pipeSources">
            <div>
              <div class="stage-icon">⇩</div>
              <div class="stage-name">MARKET DATA</div>
              <div class="stage-sub">Two independent live sources feed the research engine.</div>
              <div class="pipeline-streams">
                <div class="source-chip" id="pipeXauSource"><span class="mini-led"></span>XAU · MT5</div>
                <div class="source-chip" id="pipeBtcSource"><span class="mini-led"></span>BTC · Coinbase</div>
              </div>
            </div>
            <div class="stage-state"><span class="stage-led"></span><span id="pipeSourceState">INGESTING</span></div>
          </div>

          <div class="pipe-stage ready">
            <div>
              <div class="stage-icon">▥</div>
              <div class="stage-name">M3 / M5 / M15</div>
              <div class="stage-sub">Closed-candle alignment and multi-timeframe context.</div>
              <div class="stage-bars"><i></i><i></i><i></i><i></i><i></i></div>
            </div>
            <div class="stage-state"><span class="stage-led"></span>ALIGNMENT ACTIVE</div>
          </div>

          <div class="pipe-stage ready">
            <div>
              <div class="stage-icon">⌁</div>
              <div class="stage-name">SMC FEATURE ENGINE</div>
              <div class="stage-sub">BOS · CHOCH · MSS · liquidity sweep · OB · FVG.</div>
              <div class="stage-bars"><i></i><i></i><i></i><i></i><i></i></div>
            </div>
            <div class="stage-state"><span class="stage-led"></span>FEATURES STREAMING</div>
          </div>

          <div class="pipe-stage ready">
            <div>
              <div class="stage-icon">◈</div>
              <div class="stage-name">REGIME</div>
              <div class="stage-sub">Market state classifier transforms structure into regime context.</div>
            </div>
            <div class="stage-state"><span class="stage-led"></span>CLASSIFYING</div>
          </div>

          <div class="pipe-stage" id="pipeModel">
            <div>
              <div class="stage-icon">ML</div>
              <div class="stage-name">LIGHTGBM + XGBOOST</div>
              <div class="stage-sub">Calibrated ensemble probability for TP-before-SL outcomes.</div>
              <div class="stage-bars"><i></i><i></i><i></i><i></i><i></i></div>
            </div>
            <div class="stage-state"><span class="stage-led"></span><span id="pipeModelState">CHECKING MODEL</span></div>
          </div>

          <div class="pipe-stage ready">
            <div>
              <div class="stage-icon">R</div>
              <div class="stage-name">EXPECTED R</div>
              <div class="stage-sub">Probability × reward/risk converted into expected return.</div>
              <div class="signal-orb">R</div>
            </div>
            <div class="stage-state"><span class="stage-led"></span>SCORING</div>
          </div>

          <div class="pipe-stage ready">
            <div>
              <div class="stage-icon">✓</div>
              <div class="stage-name">SHADOW JOURNAL</div>
              <div class="stage-sub">Qualified research signals are recorded without broker execution.</div>
              <div class="signal-orb">✓</div>
            </div>
            <div class="stage-state"><span class="stage-led"></span><span id="pipeSignalState">MONITORING</span></div>
          </div>
        </div>

        <div class="pipeline-legend">
          <div class="pipe-live">
            <span id="pipeBtcInfo">BTC Coinbase: checking…</span>
            <span id="pipeXauInfo">XAU MT5: checking…</span>
            <span id="pipeModelInfo">Model: checking…</span>
            <span id="pipeSignalInfo">Shadow signals: 0</span>
          </div>
          <div class="flow-caption">Market Data → Structure → ML → Expected R → Shadow Research</div>
        </div>
      </div>

      <div class="small" style="margin-top:11px">API docs: <a href="/docs">/docs</a> · Live status: <a href="/api/live/status">/api/live/status</a> · Training status: <a href="/api/training/status">/api/training/status</a></div>
    </div>
  </div>
  <div class="footer">V0.11.2 research mode. BTCUSD is sourced from Coinbase BTC-USD. M3 is built causally from three closed 1-minute Coinbase candles. Broker orders remain disabled.</div>
</div>
<script>
const $=id=>document.getElementById(id);

const trainOrder=["DOWNLOADING_HISTORY","BUILDING_DATASET","TRAINING_REGIME","TRAINING_ENSEMBLE","VALIDATING","READY"];
const trainPctMap={DOWNLOADING_HISTORY:10,BUILDING_DATASET:30,TRAINING_REGIME:50,TRAINING_ENSEMBLE:70,VALIDATING:90,READY:100};
let trainReplay=false;
let lastTraining=null;

function fmtTime(v){return v?String(v).replace("T"," ").replace("Z","").slice(0,19):"—"}

function drawTrainingState(status, meta={}, visualOnly=false){
  const visualMap={
    FEATURE_PRUNING:"TRAINING_ENSEMBLE",
    TRAINING_SIDE_MODELS:"TRAINING_ENSEMBLE",
    LEARNING_CONTEXT_POLICY:"VALIDATING",
    VALIDATING_QUALITY:"VALIDATING"
  };
  const visualStatus=visualMap[status]||status;
  const idx=trainOrder.indexOf(visualStatus);
  const pct=trainPctMap[visualStatus] ?? (status==="FAILED"?0:0);
  for(let i=0;i<6;i++){
    const el=$("step"+i), no=el.querySelector(".step-no");
    el.classList.remove("done","active","failed");
    no.textContent=el.dataset.num;
    if(status==="FAILED"){
      if(i===Math.max(0,idx)) el.classList.add("failed");
    }else if(idx>=0){
      if(i<idx){el.classList.add("done");no.textContent="✓"}
      else if(i===idx) el.classList.add(visualStatus==="READY"?"done":"active");
      if(visualStatus==="READY" && i===idx) no.textContent="✓";
    }
  }
  $("trainProgressFill").style.width=pct+"%";
  $("trainPct").textContent=pct+"%";
  $("trainStatusPill").textContent=status.replaceAll("_"," ");
  $("trainStatus").textContent=status.replaceAll("_"," ");
  $("readyBurst").classList.toggle("show",visualStatus==="READY");
  if(!visualOnly){
    $("trainVersion").textContent=meta.version||"v0.11.2";
    $("trainRows").textContent=(meta.dataset_rows||0).toLocaleString();
    $("trainStarted").textContent=fmtTime(meta.started_at);
    $("trainFinished").textContent=fmtTime(meta.finished_at);
    if(meta.error){
      $("trainSub").textContent="Training error: "+meta.error;
      $("trainSub").className="small bad";
    }else{
      $("trainSub").textContent="Live backend training state. Animation reflects the actual training pipeline.";
      $("trainSub").className="small";
    }
  }
}

async function replayTraining(){
  if(trainReplay)return;
  trainReplay=true;
  $("replayTrainingBtn").disabled=true;
  $("trainSub").textContent="Replaying completed training pipeline — visualization only; model is not being retrained.";
  $("trainSub").className="small warn";
  for(const st of trainOrder){
    drawTrainingState(st,lastTraining||{},true);
    await new Promise(r=>setTimeout(r,620));
  }
  trainReplay=false;
  $("replayTrainingBtn").disabled=false;
  if(lastTraining)drawTrainingState(lastTraining.status||"IDLE",lastTraining,false);
}

function updatePipeline(d){
  const cb=d.coinbase||{};
  const xs=symbolData(d,"XAUUSD");
  const btc=symbolData(d,"BTCUSD");
  const btcLive=!!cb.last_success;
  const xauLive=!!(xs && ((xs.timeframes?.M3?.rows||0)+(xs.timeframes?.M5?.rows||0)+(xs.timeframes?.M15?.rows||0)>0));
  const modelReady=!!d.models?.BTCUSD?.available;

  $("pipeBtcSource").classList.toggle("live",btcLive);
  $("pipeXauSource").classList.toggle("live",xauLive);
  $("pipeSources").classList.toggle("ready",btcLive||xauLive);
  $("pipeSources").classList.toggle("wait",!(btcLive||xauLive));
  $("pipeSourceState").textContent=(btcLive||xauLive)?"LIVE FEED":"WAITING FOR DATA";

  $("pipeModel").classList.toggle("ready",modelReady);
  $("pipeModel").classList.toggle("wait",!modelReady);
  $("pipeModelState").textContent=modelReady?"MODEL READY":"MODEL NOT AVAILABLE";

  $("pipeBtcInfo").textContent="BTC Coinbase: "+(btcLive?"LIVE":"WAITING");
  $("pipeXauInfo").textContent="XAU MT5: "+(xauLive?"LIVE":"WAITING");
  $("pipeModelInfo").textContent="BTC Model: "+(modelReady?(d.models.BTCUSD.version||"READY"):"NOT AVAILABLE");
  $("pipeSignalInfo").textContent="Shadow signals: "+(d.buffer?.shadow_signal_count||0);
  $("pipeSignalState").textContent=(d.buffer?.shadow_signal_count||0)>0?"RECORDING":"MONITORING";
}
function symbolData(data,name){return (data.buffer?.symbols||[]).find(x=>x.symbol===name)}
function paintSymbol(prefix,s,waiting){
  if(!s)return;
  const m3=s.timeframes?.M3||{},m5=s.timeframes?.M5||{},m15=s.timeframes?.M15||{};
  $(prefix+"M3").textContent=(m3.rows||0)+" rows";
  $(prefix+"M5").textContent=(m5.rows||0)+" rows";
  $(prefix+"M15").textContent=(m15.rows||0)+" rows";
  const price=m3.last_close ?? m5.last_close ?? m15.last_close;
  $(prefix+"Price").textContent=price==null?waiting:Number(price).toLocaleString(undefined,{maximumFractionDigits:5});
  $(prefix+"Ready").className=s.ready?"small ok":"small";
}
function modelText(el,m){
  if(m?.available && m?.quality_passed){el.textContent="READY "+(m.version||"")+" · QUALITY PASSED";el.className="v ok"}
  else if(m?.available){el.textContent=(m.version||"")+" · QUALITY BLOCKED";el.className="v warn"}
  else{el.textContent="MODEL NOT AVAILABLE";el.className="v warn"}
}
async function refresh(){
  try{
    const r=await fetch("/api/live/status",{cache:"no-store"}); const d=await r.json();
    $("apiState").textContent=d.status+" · V"+d.version;
    $("bridgeState").textContent=d.bridge?.configured?"CONFIGURED":"TOKEN NOT SET";
    $("bridgeState").className=d.bridge?.configured?"v ok":"v warn";
    const cb=d.coinbase||{};
    $("coinbaseState").textContent=cb.last_success?"LIVE":(cb.last_error?"ERROR":"CONNECTING");
    $("coinbaseState").className=cb.last_success?"v ok":(cb.last_error?"v bad":"v warn");
    $("coinbaseDetail").textContent=cb.last_error?cb.last_error:(cb.last_success?"Last sync: "+cb.last_success:"Waiting for first Coinbase sync…");
    $("signalCount").textContent=d.buffer?.shadow_signal_count||0;
    paintSymbol("xau",symbolData(d,"XAUUSD"),"WAITING FOR MT5");
    paintSymbol("btc",symbolData(d,"BTCUSD"),"CONNECTING TO COINBASE");
    const bs=symbolData(d,"BTCUSD"); if(bs)$("btcReady").textContent=bs.ready?"Coinbase buffer ready for inference.":"Coinbase is filling 300 closed candles per timeframe.";
    const xs=symbolData(d,"XAUUSD"); if(xs)$("xauReady").textContent=xs.ready?"MT5 buffer ready for inference.":"Need 300 closed candles per timeframe before inference.";
    modelText($("xauModel"),d.models?.XAUUSD);
    modelText($("btcModel"),d.models?.BTCUSD);
    lastTraining=d.training?.BTCUSD||null;
    if(lastTraining && !trainReplay)drawTrainingState(lastTraining.status||"IDLE",lastTraining,false);
    updatePipeline(d);
  }catch(e){$("apiState").textContent="API ERROR"}

  try{
    const r=await fetch("/api/live/signals?limit=20",{cache:"no-store"}); const d=await r.json(); const rows=$("signalRows"); rows.innerHTML="";
    const sig=(d.signals||[]).slice().reverse();
    if(!sig.length){rows.innerHTML='<tr><td colspan="8" class="small">No model signals yet.</td></tr>';return}
    for(const s of sig){
      const tr=document.createElement("tr");
      const vals=[s.timestamp||s.recorded_at||"—",s.symbol||"—",s.side||"—",s.probability==null?"—":Number(s.probability).toFixed(3),s.expected_r==null?"—":Number(s.expected_r).toFixed(3),s.regime??"—",s.model_version||"—",s.qualified===true?"YES":s.qualified===false?"NO":"—"];
      for(const v of vals){const td=document.createElement("td");td.textContent=v;tr.appendChild(td)} rows.appendChild(tr)
    }
  }catch(e){}
}
$("replayTrainingBtn").addEventListener("click",replayTraining);
refresh();setInterval(refresh,5000);
</script>
</body></html>"""

