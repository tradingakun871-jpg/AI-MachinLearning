import os
from datetime import datetime,timezone
from pathlib import Path

from fastapi import FastAPI,HTTPException,Depends
from fastapi.responses import HTMLResponse

from app.schemas import AnalysisRequest,AnalysisResponse
from app.engine import MarketAgent
from app.security import verify_bridge_token
from live.service import ShadowService

VERSION="0.8.0"
app=FastAPI(title="AI Market Intelligence Agent",version=VERSION)
agent=MarketAgent()
shadow=ShadowService()

def bridge_configured():
    token=os.getenv("BRIDGE_TOKEN","")
    return bool(token and token!="change-me")

def model_available(symbol,version="v0.5"):
    return Path("artifacts/models")/symbol.upper()/version/"model.joblib"

@app.get("/health")
def health():
    return {"status":"ok","mode":"SHADOW_RESEARCH","version":VERSION}

@app.get("/api/status")
def status():
    return {**agent.status(),"live_mode":"SHADOW_RESEARCH","version":VERSION}

@app.get("/api/live/status")
def live_status():
    payload=shadow.status()
    model_status={}
    for symbol in ("XAUUSD","BTCUSD","BTCUSDT"):
        path=model_available(symbol)
        model_status[symbol]={
            "available":path.exists(),
            "version":"v0.5" if path.exists() else None,
            "state":"READY" if path.exists() else "MODEL_NOT_AVAILABLE",
        }
    recent=shadow.recent(1)
    return {
        "status":"ONLINE",
        "version":VERSION,
        "mode":"SHADOW_RESEARCH",
        "broker_orders_enabled":False,
        "bridge":{"configured":bridge_configured(),"endpoint":"/api/live/candle","timeframes":["M3","M5","M15"]},
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
<title>AI Market Intelligence V0.8</title>
<style>
:root{--bg:#080d1b;--panel:#11192b;--panel2:#172137;--text:#eef3ff;--muted:#8fa0bd;--line:#26324c;--green:#49e59a;--amber:#ffcb66;--red:#ff7184;--blue:#68a7ff}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(180deg,#070b16,#0a1020);color:var(--text);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1260px;margin:auto;padding:24px}.top{display:flex;gap:16px;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;margin-bottom:18px}
h1{font-size:28px;margin:0 0 6px}.sub{color:var(--muted)}.badge{display:inline-flex;align-items:center;gap:7px;padding:7px 10px;border:1px solid var(--line);border-radius:999px;background:#0d1528}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);box-shadow:0 0 14px var(--green)}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}
.card{background:rgba(17,25,43,.94);border:1px solid var(--line);border-radius:16px;padding:18px;box-shadow:0 12px 30px rgba(0,0,0,.18)}
.span3{grid-column:span 3}.span4{grid-column:span 4}.span6{grid-column:span 6}.span8{grid-column:span 8}.span12{grid-column:span 12}
.k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.v{font-size:23px;font-weight:750;margin-top:6px}.small{font-size:12px;color:var(--muted)}
.ok{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}.blue{color:var(--blue)}
.tf{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:13px}.tf div{background:#0c1426;border:1px solid var(--line);border-radius:10px;padding:10px}
table{width:100%;border-collapse:collapse;margin-top:10px}th,td{text-align:left;padding:10px 8px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-weight:600;font-size:12px}
.scroll{overflow:auto}.footer{color:var(--muted);font-size:12px;margin:18px 2px}.pill{padding:4px 8px;border-radius:999px;background:#0d1528;border:1px solid var(--line);font-size:12px}
a{color:#8bb8ff;text-decoration:none}@media(max-width:900px){.span3,.span4,.span6,.span8{grid-column:span 12}.wrap{padding:14px}h1{font-size:23px}}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div><h1>AI Market Intelligence Agent V0.8</h1><div class="sub">Live research dashboard · MT5 M3/M5/M15 · SMC · ML inference · Shadow journal</div></div>
    <div class="badge"><span class="dot"></span><span id="apiState">Checking API…</span></div>
  </div>

  <div class="grid">
    <div class="card span3"><div class="k">Mode</div><div class="v warn">SHADOW RESEARCH</div><div class="small">Broker orders are disabled.</div></div>
    <div class="card span3"><div class="k">MT5 Bridge</div><div class="v" id="bridgeState">—</div><div class="small">Authenticated closed-candle feed.</div></div>
    <div class="card span3"><div class="k">Shadow Signals</div><div class="v" id="signalCount">0</div><div class="small">In-memory research journal.</div></div>
    <div class="card span3"><div class="k">Server UTC</div><div class="v" style="font-size:16px" id="serverTime">—</div><div class="small">Refresh every 5 seconds.</div></div>

    <div class="card span6">
      <div class="k">XAUUSD Live Feed</div><div class="v" id="xauPrice">WAITING FOR MT5</div>
      <div class="tf">
        <div><div class="k">M3</div><b id="xauM3">0 rows</b></div>
        <div><div class="k">M5</div><b id="xauM5">0 rows</b></div>
        <div><div class="k">M15</div><b id="xauM15">0 rows</b></div>
      </div>
      <p class="small" id="xauReady">Need 300 closed candles per timeframe before inference.</p>
    </div>

    <div class="card span6">
      <div class="k">BTC Live Feed</div><div class="v" id="btcPrice">WAITING FOR MT5</div>
      <div class="tf">
        <div><div class="k">M3</div><b id="btcM3">0 rows</b></div>
        <div><div class="k">M5</div><b id="btcM5">0 rows</b></div>
        <div><div class="k">M15</div><b id="btcM15">0 rows</b></div>
      </div>
      <p class="small" id="btcReady">Need 300 closed candles per timeframe before inference.</p>
    </div>

    <div class="card span4"><div class="k">XAUUSD ML Model</div><div class="v" id="xauModel">—</div><div class="small">LightGBM/XGBoost ensemble artifact.</div></div>
    <div class="card span4"><div class="k">BTC ML Model</div><div class="v" id="btcModel">—</div><div class="small">No fabricated probability if artifact is absent.</div></div>
    <div class="card span4"><div class="k">Performance</div><div class="v warn">N/A</div><div class="small">Win rate / PF / DD appear only after validated outcomes exist.</div></div>

    <div class="card span12">
      <div class="k">Latest Shadow Signals</div>
      <div class="scroll"><table><thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Probability</th><th>Expected R</th><th>Regime</th><th>Model</th><th>Qualified</th></tr></thead><tbody id="signalRows"><tr><td colspan="8" class="small">No model signals yet.</td></tr></tbody></table></div>
    </div>

    <div class="card span12">
      <div class="k">Pipeline Status</div>
      <p><span class="pill">MT5</span> → <span class="pill">M3/M5/M15</span> → <span class="pill">SMC BOS/CHOCH/MSS + OB/FVG</span> → <span class="pill">Regime</span> → <span class="pill">LightGBM/XGBoost</span> → <span class="pill">Calibration</span> → <span class="pill">Expected R</span> → <span class="pill">Shadow Journal</span></p>
      <div class="small">API docs: <a href="/docs">/docs</a> · Live status: <a href="/api/live/status">/api/live/status</a> · Signals: <a href="/api/live/signals">/api/live/signals</a></div>
    </div>
  </div>
  <div class="footer">V0.8 research mode. Live market values appear only after an authenticated MT5 bridge submits closed candles. Model metrics are never invented.</div>
</div>
<script>
const $=id=>document.getElementById(id);
function symbolData(data,name){
  return (data.buffer?.symbols||[]).find(x=>x.symbol===name);
}
function paintSymbol(prefix,s){
  if(!s)return;
  const m3=s.timeframes?.M3||{},m5=s.timeframes?.M5||{},m15=s.timeframes?.M15||{};
  $(prefix+"M3").textContent=(m3.rows||0)+" rows";
  $(prefix+"M5").textContent=(m5.rows||0)+" rows";
  $(prefix+"M15").textContent=(m15.rows||0)+" rows";
  const price=m3.last_close ?? m5.last_close ?? m15.last_close;
  $(prefix+"Price").textContent=price==null?"WAITING FOR MT5":Number(price).toLocaleString(undefined,{maximumFractionDigits:5});
  $(prefix+"Ready").textContent=s.ready?"Buffer ready for inference.":"Need 300 closed candles per timeframe before inference.";
  $(prefix+"Ready").className=s.ready?"small ok":"small";
}
function modelText(el,m){
  if(m?.available){el.textContent="READY "+(m.version||"");el.className="v ok";}
  else{el.textContent="MODEL NOT AVAILABLE";el.className="v warn";}
}
async function refresh(){
  try{
    const r=await fetch("/api/live/status",{cache:"no-store"}); const d=await r.json();
    $("apiState").textContent=d.status+" · V"+d.version;
    $("bridgeState").textContent=d.bridge?.configured?"CONFIGURED":"TOKEN NOT SET";
    $("bridgeState").className=d.bridge?.configured?"v ok":"v warn";
    $("signalCount").textContent=d.buffer?.shadow_signal_count||0;
    $("serverTime").textContent=(d.server_time||"").replace("T"," ").slice(0,19);
    paintSymbol("xau",symbolData(d,"XAUUSD"));
    paintSymbol("btc",symbolData(d,"BTCUSD")||symbolData(d,"BTCUSDT"));
    modelText($("xauModel"),d.models?.XAUUSD);
    modelText($("btcModel"),d.models?.BTCUSD?.available?d.models.BTCUSD:d.models?.BTCUSDT);
  }catch(e){$("apiState").textContent="API ERROR";}

  try{
    const r=await fetch("/api/live/signals?limit=20",{cache:"no-store"}); const d=await r.json(); const rows=$("signalRows"); rows.innerHTML="";
    const sig=(d.signals||[]).slice().reverse();
    if(!sig.length){rows.innerHTML='<tr><td colspan="8" class="small">No model signals yet.</td></tr>';return;}
    for(const s of sig){
      const tr=document.createElement("tr");
      const vals=[s.timestamp||s.recorded_at||"—",s.symbol||"—",s.side||"—",s.probability==null?"—":Number(s.probability).toFixed(3),s.expected_r==null?"—":Number(s.expected_r).toFixed(3),s.regime??"—",s.model_version||"—",s.qualified===true?"YES":s.qualified===false?"NO":"—"];
      for(const v of vals){const td=document.createElement("td");td.textContent=v;tr.appendChild(td)} rows.appendChild(tr);
    }
  }catch(e){}
}
refresh();setInterval(refresh,5000);
</script>
</body></html>"""

