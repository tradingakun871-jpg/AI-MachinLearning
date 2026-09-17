import pandas as pd
from data.feature_builder import add_market_features
from data.multitimeframe import attach_htf_context
from smc.engine import add_smc_features
from live.model_loader import LiveModelLoader
from live.shadow import ShadowJournal

class LiveInference:
    def __init__(self,threshold=.60,rr=3.0): self.threshold=threshold; self.rr=rr; self.loader=LiveModelLoader(); self.journal=ShadowJournal()
    def analyze(self,symbol,m3,m5,m15,version="v0.5"):
        bundle=self.loader.load(symbol,version)
        if bundle is None: return {"status":"MODEL_NOT_AVAILABLE","symbol":symbol,"mode":"RESEARCH"}
        d=add_smc_features(add_market_features(m3)); d=attach_htf_context(d,m5,m15); d["trend_m3"]=d.structure_bias
        row=d.tail(1).copy(); row["regime"]=bundle["regime"].predict(row); features=bundle["features"]
        missing=[c for c in features if c not in row]
        if missing: return {"status":"FEATURE_MISMATCH","missing":missing,"symbol":symbol,"mode":"RESEARCH"}
        p=float(bundle["model"].predict_proba(row[features])[0]); expected_r=p*self.rr-(1-p); direction=int(row.mss.iloc[0]); side="BUY" if direction>0 else "SELL" if direction<0 else "WAIT"
        qualified=side!="WAIT" and p>=self.threshold and expected_r>0
        event={"status":"OK","symbol":symbol,"timestamp":str(row.timestamp.iloc[0]),"side":side,"probability":round(p,4),"expected_r":round(expected_r,4),"regime":int(row.regime.iloc[0]),"qualified":bool(qualified),"mode":"SHADOW_RESEARCH","model_version":version}
        self.journal.record(event); return event
