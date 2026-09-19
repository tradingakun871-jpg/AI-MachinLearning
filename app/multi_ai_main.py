import asyncio
import hmac
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse

from multi_ai.consensus import run_consensus
from multi_ai.providers import provider_status, self_test_all
from multi_ai.risk import evaluate_gate
from multi_ai.specialists import build_evidence, specialist_health, warmup_specialists
from multi_ai.store import init_store, latest_decision, save_decision

VERSION = "1.1.1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await asyncio.to_thread(init_store)
    except Exception as exc:
        print("multi-ai store init warning:", exc)
    yield


app = FastAPI(
    title="Multi-AI Trading Brain",
    version=VERSION,
    description="Strict 4/4 reasoning consensus for ChatGPT, Claude, Gemini and DeepSeek.",
    lifespan=lifespan,
)


def _admin_token() -> str:
    return os.getenv("DASHBOARD_ADMIN_TOKEN", "").strip()


def require_admin(x_admin_token: str | None = Header(default=None)):
    expected = _admin_token()
    if not expected:
        raise HTTPException(503, "admin token is not configured")
    supplied = (x_admin_token or "").strip()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(401, "invalid admin token")
    return True


def provider_config():
    statuses = provider_status()
    return {name: bool(info.get("configured")) for name, info in statuses.items()}


@app.get("/health")
def health():
    p = provider_config()
    return {
        "status": "ok",
        "service": "multi-ai-trading-brain",
        "version": VERSION,
        "mode": os.getenv("TRADING_MODE", "SHADOW").upper(),
        "strict_consensus": "4/4",
        "max_rounds": 2,
        "providers_configured": sum(1 for x in p.values() if x),
        "providers": list(p.keys()),
        "admin_protection": bool(_admin_token()),
    }


@app.get("/api/v1/status")
def status():
    providers = provider_status()
    return {
        "status": "ONLINE",
        "version": VERSION,
        "mode": os.getenv("TRADING_MODE", "SHADOW").upper(),
        "reasoning": {
            "policy": "STRICT_UNANIMOUS",
            "required": "4/4",
            "max_rounds": 2,
            "on_conflict": "NO_TRADE_AFTER_ROUND_2",
            "providers": ["chatgpt", "claude", "gemini", "deepseek"],
        },
        "providers": providers,
        "specialists": {
            "machine_learning": os.getenv("MARKET_INTELLIGENCE_URL", ""),
            "nlp_service": "CONFIGURED" if os.getenv("NLP_SERVICE_URL") else "BASELINE_ONLY",
            "deep_forecast_service": "CONFIGURED" if os.getenv("DL_SERVICE_URL") else "NOT_CONFIGURED",
        },
        "execution": {
            "live_switch": os.getenv("ALLOW_MULTI_AI_EXECUTION", "false").lower() == "true",
            "minimum_reasoning_confidence": float(os.getenv("MIN_REASONING_CONFIDENCE", "70")),
        },
        "security": {"admin_protection": bool(_admin_token())},
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/system-check")
async def system_check():
    specialists = await specialist_health()
    providers = provider_status()
    return {
        "ok": True,
        "version": VERSION,
        "specialists": specialists,
        "providers": providers,
        "specialists_ready": all(x.get("reachable") for x in specialists.values()),
        "reasoning_4_of_4_configured": all(x.get("configured") for x in providers.values()),
        "execution_live": (
            os.getenv("ALLOW_MULTI_AI_EXECUTION", "false").lower() == "true"
            and os.getenv("TRADING_MODE", "SHADOW").upper() == "LIVE"
        ),
    }


@app.post("/api/v1/warmup-specialists")
async def warmup(x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    return await warmup_specialists()


@app.post("/api/v1/provider-test")
async def provider_test(x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    return await self_test_all()


@app.post("/api/v1/reason")
async def reason(payload: dict, x_admin_token: str | None = Header(default=None)):
    require_admin(x_admin_token)
    symbol = str(payload.get("symbol", "XAUUSD")).upper()
    if symbol not in ("XAUUSD", "BTCUSD"):
        raise HTTPException(422, "symbol must be XAUUSD or BTCUSD")

    evidence = await build_evidence(payload)
    consensus = await run_consensus(evidence)
    gate = evaluate_gate(evidence, consensus)

    try:
        decision_id = await asyncio.to_thread(save_decision, evidence, consensus, gate)
    except Exception as exc:
        decision_id = None
        gate = {**gate, "journal_warning": f"{type(exc).__name__}: {exc}"}

    return {
        "ok": True,
        "decision_id": decision_id,
        "symbol": evidence["symbol"],
        "timeframe": evidence["timeframe"],
        "final_decision": consensus["decision"],
        "single_voice": consensus["single_voice"],
        "confidence": consensus["confidence"],
        "agreement": consensus["agreement"],
        "rounds": consensus["rounds"],
        "research_signal_allowed": gate["research_signal_allowed"],
        "execution_allowed": gate["execution_allowed"],
        "blockers": gate["blockers"],
        "evidence": evidence,
        "audit": {
            "consensus_status": consensus["status"],
            "votes": consensus.get("votes", []),
            "round1": consensus.get("round1"),
            "gate": gate,
        },
    }


@app.get("/api/v1/latest")
async def latest():
    try:
        row = await asyncio.to_thread(latest_decision)
    except Exception as exc:
        raise HTTPException(503, f"journal unavailable: {exc}")
    return {"latest": row}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return r"""<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multi-AI Trading Brain</title>
<style>
:root{--bg:#07101c;--panel:#0d1928;--line:#22344c;--txt:#eef5ff;--muted:#8fa5bf;--ok:#48e39a;--warn:#ffc95f;--bad:#ff6f83}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% 0,#11213b 0,#07101c 40%);color:var(--txt);font:14px system-ui}.wrap{max-width:1220px;margin:auto;padding:26px}.top{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap}h1{margin:0;font-size:29px}.sub{color:var(--muted);margin-top:6px}.badge{border:1px solid var(--line);border-radius:999px;padding:7px 11px;background:#0b1727}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px;margin-top:16px}.card{grid-column:span 4;background:rgba(13,25,40,.95);border:1px solid var(--line);border-radius:16px;padding:18px}.wide{grid-column:span 12}.half{grid-column:span 6}.k{font-size:11px;color:var(--muted);letter-spacing:.09em;text-transform:uppercase}.v{font-size:24px;font-weight:800;margin-top:6px}.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.agents{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px}.agent{border:1px solid var(--line);background:#081422;border-radius:13px;padding:14px}.led{width:8px;height:8px;border-radius:50%;display:inline-block;background:var(--warn);margin-right:7px}.agent.ready .led{background:var(--ok)}button{background:#15365e;color:#fff;border:1px solid #2e5f97;border-radius:10px;padding:10px 14px;font-weight:750;cursor:pointer;margin-right:7px;margin-bottom:7px}input,select,textarea{width:100%;background:#081422;color:var(--txt);border:1px solid var(--line);border-radius:9px;padding:9px;margin-top:5px}.form{display:grid;grid-template-columns:1fr 1fr 2fr;gap:10px}.voice{font-size:19px;line-height:1.55;margin-top:10px;padding:15px;border-radius:12px;background:#081422;border:1px solid var(--line)}.pills{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.pill{border:1px solid var(--line);border-radius:999px;padding:6px 9px;color:var(--muted)}pre{white-space:pre-wrap;word-break:break-word;background:#07111e;border:1px solid var(--line);border-radius:12px;padding:13px;max-height:360px;overflow:auto}@media(max-width:800px){.card,.half{grid-column:span 12}.agents{grid-template-columns:1fr 1fr}.form{grid-template-columns:1fr}}
</style></head>
<body><div class="wrap">
<div class="top"><div><h1>Multi-AI Trading Brain V1.1.1</h1><div class="sub">ML + FinBERT + Chronos/PatchTST/TFT → ChatGPT + Claude + Gemini + DeepSeek → STRICT 4/4</div></div><div class="badge" id="mode">CHECKING...</div></div>
<div class="grid">
<div class="card wide"><div class="k">Reasoning Council</div><div class="agents" id="agents"></div><div class="pills"><span class="pill">4/4 required</span><span class="pill">Max 2 rounds</span><span class="pill">Conflict → NO TRADE</span><span class="pill">Live execution OFF</span></div></div>
<div class="card"><div class="k">Final Decision</div><div class="v" id="decision">—</div></div><div class="card"><div class="k">Consensus Confidence</div><div class="v" id="confidence">—</div></div><div class="card"><div class="k">Execution Gate</div><div class="v bad" id="gate">BLOCKED</div></div>
<div class="card wide"><div class="k">Single AI Voice</div><div class="voice" id="voice">Belum ada keputusan.</div></div>
<div class="card half"><div class="k">Admin / System</div><label>Admin token<input id="admin" type="password" placeholder="Masukkan DASHBOARD_ADMIN_TOKEN"></label><p><button onclick="saveToken()">Save token</button><button onclick="syscheck()">System Check</button><button onclick="warmup()">Warm-up Models</button><button onclick="providerTest()">Test 4 AI</button></p><pre id="system">—</pre></div>
<div class="card half"><div class="k">Run Shadow Analysis</div><div class="form"><div><label>Symbol<select id="symbol"><option>XAUUSD</option><option>BTCUSD</option></select></label></div><div><label>Timeframe<select id="tf"><option>M3</option><option>M5</option><option>M15</option></select></label></div><div><label>Context<input id="ctx" placeholder="mis. London session, spread normal"></label></div></div><label>News/headlines (satu per baris)<textarea id="news" rows="4" placeholder="Contoh: US CPI comes in above forecast"></textarea></label><p><button onclick="run()">Analyze with 4 AI</button></p><div class="sub" id="runstate">Execution request selalu false dari dashboard ini.</div></div>
<div class="card wide"><div class="k">Specialists</div><div class="pills" id="specialists"></div></div><div class="card wide"><div class="k">Gate / Blockers</div><pre id="blockers">—</pre></div>
</div></div>
<script>
const $=x=>document.getElementById(x);const token=()=>localStorage.getItem('mai_admin')||'';const hdr=()=>({'Content-Type':'application/json','X-Admin-Token':token()});
function saveToken(){localStorage.setItem('mai_admin',$('admin').value.trim());$('system').textContent='Token tersimpan hanya di browser ini.'}
function paintDecision(d){$('decision').textContent=d||'—';$('decision').className='v '+(d==='BUY'?'ok':d==='SELL'?'bad':'warn')}
async function refresh(){try{const s=await fetch('/api/v1/status',{cache:'no-store'}).then(r=>r.json());$('mode').textContent=s.mode+' · 4/4 STRICT';$('agents').innerHTML=Object.entries(s.providers).map(([k,v])=>'<div class="agent '+(v.configured?'ready':'')+'"><div><span class="led"></span><b>'+k.toUpperCase()+'</b></div><div class="sub">'+v.model+' · '+(v.configured?'CONFIGURED':'KEY NEEDED')+'</div></div>').join('');$('specialists').innerHTML='<span class="pill">ML</span><span class="pill">FinBERT: '+s.specialists.nlp_service+'</span><span class="pill">Deep: '+s.specialists.deep_forecast_service+'</span>'}catch(e){}try{const d=await fetch('/api/v1/latest',{cache:'no-store'}).then(r=>r.json()),x=d.latest;if(!x)return;paintDecision(x.decision);$('confidence').textContent=Number(x.confidence).toFixed(1)+'%';$('voice').textContent=x.single_voice||'—';$('gate').textContent=x.execution_allowed?'ALLOWED':'BLOCKED';$('gate').className='v '+(x.execution_allowed?'ok':'bad');$('blockers').textContent=JSON.stringify(x.gate?.blockers||[],null,2)}catch(e){}}
async function syscheck(){try{const d=await fetch('/api/v1/system-check',{cache:'no-store'}).then(r=>r.json());$('system').textContent=JSON.stringify(d,null,2)}catch(e){$('system').textContent='ERROR: '+e.message}}
async function warmup(){try{const r=await fetch('/api/v1/warmup-specialists',{method:'POST',headers:hdr(),body:'{}'}),d=await r.json();$('system').textContent=JSON.stringify(d,null,2)}catch(e){$('system').textContent='ERROR: '+e.message}}
async function providerTest(){try{const r=await fetch('/api/v1/provider-test',{method:'POST',headers:hdr(),body:'{}'}),d=await r.json();$('system').textContent=JSON.stringify(d,null,2)}catch(e){$('system').textContent='ERROR: '+e.message}}
async function run(){$('runstate').textContent='Evidence + 4 AI reasoning sedang diproses…';$('runstate').className='sub warn';const headlines=$('news').value.split('\n').map(x=>x.trim()).filter(Boolean);const body={symbol:$('symbol').value,timeframe:$('tf').value,news:headlines,context:{note:$('ctx').value},execution_requested:false};try{const r=await fetch('/api/v1/reason',{method:'POST',headers:hdr(),body:JSON.stringify(body)}),d=await r.json();if(!r.ok)throw Error(d.detail||'request failed');paintDecision(d.final_decision);$('confidence').textContent=Number(d.confidence).toFixed(1)+'%';$('voice').textContent=d.single_voice;$('gate').textContent=d.execution_allowed?'ALLOWED':'BLOCKED';$('gate').className='v '+(d.execution_allowed?'ok':'bad');$('blockers').textContent=JSON.stringify(d.blockers,null,2);$('runstate').textContent='Selesai · '+d.agreement+' · '+d.rounds+' round';$('runstate').className='sub ok'}catch(e){$('runstate').textContent='ERROR: '+e.message;$('runstate').className='sub bad'}}
$('admin').value=token();refresh();syscheck();setInterval(refresh,10000);
</script></body></html>"""
