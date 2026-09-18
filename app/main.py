import os
from contextlib import asynccontextmanager
from datetime import datetime,timezone
from pathlib import Path

from fastapi import FastAPI,HTTPException,Depends
from fastapi.responses import HTMLResponse

from app.schemas import AnalysisRequest,AnalysisResponse
from app.engine import MarketAgent
from app.security import verify_bridge_token
from live.service import ShadowService
from data.coinbase import CoinbaseBTCFeed
from ml.auto_train import TrainingManager

VERSION="0.9.1"
agent=MarketAgent()
shadow=ShadowService()
coinbase=CoinbaseBTCFeed(shadow,poll_seconds=int(os.getenv("COINBASE_POLL_SECONDS","60")))
trainer=TrainingManager(shadow,version="v0.9")

@asynccontextmanager
async def lifespan(app:FastAPI):
    await coinbase.start()
    await trainer.start_btc(force=False)
    yield
    if trainer.task and not trainer.task.done():
        trainer.task.cancel()
    await coinbase.stop()

app=FastAPI(title="AI Market Intelligence Agent",version=VERSION,lifespan=lifespan)

def bridge_configured():
    token=os.getenv("BRIDGE_TOKEN","")
    return bool(token and token!="change-me")

def model_available(symbol,version="v0.9"):
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

@app.get("/api/training/status")
def training_status():
    return trainer.status()

@app.post("/api/training/btc/retrain")
async def retrain_btc():
    await trainer.start_btc(force=True)
    return {"accepted":True,"status":trainer.status()["BTCUSD"]["status"]}

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
        model_status[symbol]={
            "available":path.exists(),
            "version":"v0.9" if path.exists() else None,
            "state":"READY" if path.exists() else "MODEL_NOT_AVAILABLE",
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
        "training":trainer.status(),
        "models":model_status,
        "buffer":payload,
        "last_signal":recent[-1] if recent else None,
        "server_time":datetime.now(timezone.utc).isoformat(),
        "performance":{
            "win_rate":None,
            "profit_factor":None,
            "max_drawdown_r":None,
            "net_r":None,
            "state":"WAITING_FOR_VALIDATED_OUTCOMES",
        },
    }

@app.post("/api/analyze",response_model=AnalysisResponse)
def analyze(req:AnalysisRequest):
    return agent.analyze(req)

@app.post("/api/market/candle")
def candle(payload:dict):
    return agent.ingest(payload)

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
<title>AI Market Intelligence V0.9.1</title>
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
@keyframes trainSweep{to{left:110%}}@keyframes trainShimmer{to{transform:translateX(120%)}}@keyframes trainPulse{50%{transform:scale(1.08);box-shadow:0 0 22px rgba(102,167,255,.45)}}@keyframes trainOrbit{50%{transform:translate(-18px,18px)}100%{transform:translate(0,0)}}@keyframes readyPulse{50%{transform:scale(1.06)}}@media(max-width:1100px){.train-steps{grid-template-columns:repeat(3,1fr)}.train-meta{grid-template-columns:repeat(2,1fr)}}@media(max-width:900px){.span3,.span4,.span6{grid-column:span 12}.wrap{padding:14px}h1{font-size:23px}.train-steps{grid-template-columns:1fr}.train-meta{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div><h1>AI Market Intelligence Agent V0.9.1</h1><div class="sub">XAUUSD: MT5 · BTCUSD: Coinbase BTC-USD · SMC · ML inference · Shadow journal</div></div>
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

    <div class="card span4"><div class="k">XAUUSD ML Model</div><div class="v" id="xauModel">—</div><div class="small">LightGBM/XGBoost ensemble artifact.</div></div>
    <div class="card span4"><div class="k">BTCUSD ML Model</div><div class="v" id="btcModel">—</div><div class="small">Background training uses Coinbase history with purged time splits.</div></div>
    <div class="card span4"><div class="k">Performance</div><div class="v warn">N/A</div><div class="small">Win rate / PF / DD require validated outcomes.</div></div>


    <div class="card span12" id="btcTrainingCard">
      <div class="train-head">
        <div>
          <div class="k">BTCUSD ML MODEL TRAINING</div>
          <div class="train-title">Coinbase History → Feature Engineering → Regime → Ensemble → Validation → READY</div>
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
        <div class="train-step" id="step3" data-num="4"><div class="train-flow"></div><div class="step-no">4</div><div class="step-name">TRAINING ENSEMBLE</div><div class="step-desc">LightGBM + XGBoost probability ensemble.</div></div>
        <div class="train-step" id="step4" data-num="5"><div class="train-flow"></div><div class="step-no">5</div><div class="step-name">VALIDATING</div><div class="step-desc">Calibration, threshold sweep and held-out test.</div></div>
        <div class="train-step" id="step5" data-num="6"><div class="train-flow"></div><div class="step-no">6</div><div class="step-name">READY v0.9</div><div class="step-desc">Validated artifact available for shadow inference.</div></div>
      </div>

      <div class="train-meta">
        <div class="meta-box"><div class="k">STATUS</div><div class="mv" id="trainStatus">IDLE</div></div>
        <div class="meta-box"><div class="k">VERSION</div><div class="mv" id="trainVersion">v0.9</div></div>
        <div class="meta-box"><div class="k">DATASET ROWS</div><div class="mv" id="trainRows">0</div></div>
        <div class="meta-box"><div class="k">STARTED UTC</div><div class="mv" id="trainStarted">—</div></div>
        <div class="meta-box"><div class="k">FINISHED UTC</div><div class="mv" id="trainFinished">—</div></div>
      </div>

      <div class="ready-burst" id="readyBurst">
        <div class="ready-check">✓</div>
        <div><b>BTCUSD model is READY for shadow inference.</b><div class="small">The animation can be replayed without retraining the model.</div></div>
      </div>
    </div>

    <div class="card span12">
      <div class="k">Latest Shadow Signals</div>
      <div class="scroll"><table><thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Probability</th><th>Expected R</th><th>Regime</th><th>Model</th><th>Qualified</th></tr></thead><tbody id="signalRows"><tr><td colspan="8" class="small">No model signals yet.</td></tr></tbody></table></div>
    </div>

    <div class="card span12">
      <div class="k">Pipeline Status</div>
      <p><span class="pill">XAU: MT5</span> + <span class="pill">BTC: Coinbase</span> → <span class="pill">M3/M5/M15</span> → <span class="pill">SMC BOS/CHOCH/MSS + OB/FVG</span> → <span class="pill">Regime</span> → <span class="pill">LightGBM/XGBoost</span> → <span class="pill">Expected R</span> → <span class="pill">Shadow Journal</span></p>
      <div class="small">API docs: <a href="/docs">/docs</a> · Live status: <a href="/api/live/status">/api/live/status</a> · Coinbase status: <a href="/api/coinbase/status">/api/coinbase/status</a></div>
    </div>
  </div>
  <div class="footer">V0.9.1 research mode. BTCUSD is sourced from Coinbase BTC-USD. M3 is built causally from three closed 1-minute Coinbase candles. Broker orders remain disabled.</div>
</div>
<script>
const $=id=>document.getElementById(id);

const trainOrder=["DOWNLOADING_HISTORY","BUILDING_DATASET","TRAINING_REGIME","TRAINING_ENSEMBLE","VALIDATING","READY"];
const trainPctMap={DOWNLOADING_HISTORY:10,BUILDING_DATASET:30,TRAINING_REGIME:50,TRAINING_ENSEMBLE:70,VALIDATING:90,READY:100};
let trainReplay=false;
let lastTraining=null;

function fmtTime(v){return v?String(v).replace("T"," ").replace("Z","").slice(0,19):"—"}

function drawTrainingState(status, meta={}, visualOnly=false){
  const idx=trainOrder.indexOf(status);
  const pct=trainPctMap[status] ?? (status==="FAILED"?0:0);
  for(let i=0;i<6;i++){
    const el=$("step"+i), no=el.querySelector(".step-no");
    el.classList.remove("done","active","failed");
    no.textContent=el.dataset.num;
    if(status==="FAILED"){
      if(i===Math.max(0,idx)) el.classList.add("failed");
    }else if(idx>=0){
      if(i<idx){el.classList.add("done");no.textContent="✓"}
      else if(i===idx) el.classList.add(status==="READY"?"done":"active");
      if(status==="READY" && i===idx) no.textContent="✓";
    }
  }
  $("trainProgressFill").style.width=pct+"%";
  $("trainPct").textContent=pct+"%";
  $("trainStatusPill").textContent=status.replaceAll("_"," ");
  $("trainStatus").textContent=status.replaceAll("_"," ");
  $("readyBurst").classList.toggle("show",status==="READY");
  if(!visualOnly){
    $("trainVersion").textContent=meta.version||"v0.9";
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
  if(m?.available){el.textContent="READY "+(m.version||"");el.className="v ok"}
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

