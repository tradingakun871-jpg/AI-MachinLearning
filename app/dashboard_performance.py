PERFORMANCE_PANEL = r'''
<style>
#model-performance{max-width:1260px;margin:18px auto 0;padding:0 24px 24px}
.perf-shell{background:rgba(17,25,43,.96);border:1px solid #26324c;border-radius:16px;padding:18px;box-shadow:0 12px 30px rgba(0,0,0,.18)}
.perf-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.perf-title{font-size:19px;font-weight:800}.perf-sub{font-size:12px;color:#8fa0bd;margin-top:3px}.perf-btn{border:1px solid #26324c;background:#0d1528;color:#dce9ff;border-radius:10px;padding:8px 11px;cursor:pointer;font-weight:700}
.perf-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.perf-card{background:#0b1425;border:1px solid #26324c;border-radius:14px;padding:15px}.perf-symbol{font-size:18px;font-weight:850;display:flex;align-items:center;justify-content:space-between;gap:8px}.perf-pill{font-size:11px;padding:4px 8px;border:1px solid #26324c;border-radius:999px;color:#b9c7dd}
.perf-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:13px 0}.perf-kpi{background:#0d1729;border:1px solid #22304a;border-radius:10px;padding:10px;min-height:74px}.perf-k{font-size:10px;color:#8fa0bd;text-transform:uppercase;letter-spacing:.07em}.perf-v{font-size:19px;font-weight:800;margin-top:4px}.perf-note{font-size:11px;color:#8fa0bd}
.perf-table{width:100%;border-collapse:collapse;margin-top:8px}.perf-table th,.perf-table td{padding:7px 6px;border-bottom:1px solid #22304a;font-size:12px;text-align:left}.perf-table th{color:#8fa0bd}.perf-ok{color:#49e59a}.perf-bad{color:#ff7184}.perf-warn{color:#ffcb66}
@media(max-width:900px){.perf-grid{grid-template-columns:1fr}.perf-kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:520px){#model-performance{padding-left:12px;padding-right:12px}.perf-kpis{grid-template-columns:1fr 1fr}.perf-v{font-size:17px}}
</style>
<section id="model-performance">
  <div class="perf-shell">
    <div class="perf-head">
      <div><div class="perf-title">Winrate & Model Performance</div><div class="perf-sub">Held-out model test + walk-forward diagnostics. Bukan hasil live trading.</div></div>
      <button class="perf-btn" onclick="loadModelPerformance()">Refresh</button>
    </div>
    <div id="perf-grid" class="perf-grid"><div class="perf-note">Memuat data training...</div></div>
  </div>
</section>
<script>
(function(){
  const num=(v,d=2)=>v===null||v===undefined||Number.isNaN(Number(v))?'—':Number(v).toFixed(d);
  const pct=(v)=>v===null||v===undefined||Number.isNaN(Number(v))?'—':(Number(v)*100).toFixed(2)+'%';
  const rr=(v)=>v===null||v===undefined?'—':'1:'+num(v,0);
  const val=(v)=>v===null||v===undefined?'—':v;
  const cls=(ok,status)=>ok?'perf-ok':(status==='READY'?'perf-bad':'perf-warn');
  function sideRow(name,s){
    s=s||{}; const d=s.directional_stability||{};
    return `<tr><td>${name}</td><td>${d.usable_folds??'—'}</td><td>${pct(d.median_win_rate)}</td><td>${num(d.median_expectancy_r)}R</td><td>${num(d.median_profit_factor)}</td><td>${d.pooled_trades??'—'}</td><td class="${d.passed?'perf-ok':'perf-bad'}">${d.passed?'STABLE':'BLOCKED'}</td></tr>`;
  }
  function card(symbol,x){
    x=x||{}; const t=x.trading||{}; const ts=x.temporal_summary||{}; const hw=x.high_winrate_stability||{}; const sides=x.sides||{};
    const trades=t.trades??0; const wr=trades>0?pct(t.win_rate):'—';
    const failed=(x.failed_checks||[]).join(', ')||'None';
    return `<div class="perf-card">
      <div class="perf-symbol"><span>${symbol}</span><span class="perf-pill ${cls(x.quality_passed,x.status)}">${val(x.status)} / ${val(x.quality)}</span></div>
      <div class="perf-kpis">
        <div class="perf-kpi"><div class="perf-k">Win Rate</div><div class="perf-v">${wr}</div><div class="perf-note">${trades} holdout trades</div></div>
        <div class="perf-kpi"><div class="perf-k">Profit Factor</div><div class="perf-v">${num(t.profit_factor)}</div><div class="perf-note">final holdout</div></div>
        <div class="perf-kpi"><div class="perf-k">Expectancy</div><div class="perf-v">${num(t.expectancy_r)}R</div><div class="perf-note">per selected trade</div></div>
        <div class="perf-kpi"><div class="perf-k">Net R</div><div class="perf-v">${num(t.net_r)}R</div><div class="perf-note">max DD ${num(t.max_drawdown_r)}R</div></div>
        <div class="perf-kpi"><div class="perf-k">Coverage</div><div class="perf-v">${pct(x.coverage)}</div><div class="perf-note">selected / test rows</div></div>
        <div class="perf-kpi"><div class="perf-k">RR</div><div class="perf-v">${rr(x.rr)}</div><div class="perf-note">fixed target</div></div>
        <div class="perf-kpi"><div class="perf-k">Dataset</div><div class="perf-v">${val(x.dataset_rows)}</div><div class="perf-note">resolved candidates</div></div>
        <div class="perf-kpi"><div class="perf-k">WF Median WR</div><div class="perf-v">${pct(hw.median_win_rate)}</div><div class="perf-note">${ts.fold_count??'—'} folds / PF ${num(ts.median_profit_factor)}</div></div>
      </div>
      <table class="perf-table"><thead><tr><th>Side</th><th>Usable Folds</th><th>Median WR</th><th>Expectancy</th><th>PF</th><th>Trades</th><th>Gate</th></tr></thead><tbody>${sideRow('BUY',sides.BUY)}${sideRow('SELL',sides.SELL)}</tbody></table>
      <div class="perf-note" style="margin-top:9px">Temporal: <b>${val(x.temporal_stability)}</b> · Stable sides: <b>${(x.stable_sides||[]).join(', ')||'None'}</b></div>
      <div class="perf-note" style="margin-top:4px">Failed checks: ${failed}</div>
    </div>`;
  }
  function updateLegacyPerformance(d){
    const legacy=[...document.querySelectorAll('.card.span4')].find(el=>{
      const k=el.querySelector('.k'); return k&&k.textContent.trim()==='Performance';
    });
    if(!legacy)return;
    const x=d.XAUUSD||{}, b=d.BTCUSD||{};
    const xt=x.trading||{}, bt=b.trading||{};
    const xHold=(xt.trades??0)>0?pct(xt.win_rate):null;
    const bHold=(bt.trades??0)>0?pct(bt.win_rate):null;
    const xWf=pct((x.high_winrate_stability||{}).median_win_rate);
    const bWf=pct((b.high_winrate_stability||{}).median_win_rate);
    const main=legacy.querySelector('.v');
    const note=legacy.querySelector('.small');
    if(main){
      main.textContent=(xHold||bHold)?`XAU ${xHold||'—'} · BTC ${bHold||'—'}`:`WF XAU ${xWf} · BTC ${bWf}`;
      main.className='v '+((x.quality_passed||b.quality_passed)?'ok':'warn');
    }
    if(note)note.textContent=(xHold||bHold)?'Final holdout win rate. Detail PF / expectancy / DD ada di panel bawah.':'Final gate belum menghasilkan trade; angka di atas adalah walk-forward median WR. Detail lengkap ada di panel bawah.';
  }
  window.loadModelPerformance=async function(){
    const target=document.getElementById('perf-grid'); if(!target)return;
    try{
      const r=await fetch('/api/training/summary',{cache:'no-store'}); if(!r.ok)throw new Error('HTTP '+r.status);
      const d=await r.json();
      target.innerHTML=card('XAUUSD',d.XAUUSD)+card('BTCUSD',d.BTCUSD);
      updateLegacyPerformance(d);
    }catch(e){
      target.innerHTML=`<div class="perf-note perf-bad">Performance data gagal dimuat: ${e.message}</div>`;
    }
  };
  loadModelPerformance(); setInterval(loadModelPerformance,10000);
})();
</script>
'''
