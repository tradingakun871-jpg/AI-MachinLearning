import numpy as np
import pandas as pd

from data.feature_builder import add_market_features
from data.multitimeframe import attach_htf_context
from smc.engine import add_smc_features
from live.model_loader import LiveModelLoader
from live.shadow import ShadowJournal
from ml.setup_features import add_session_name, add_setup_features, structural_stop_distance


class LiveInference:
    def __init__(self,threshold=.60,rr=3.0):
        self.threshold=threshold
        self.rr=rr
        self.loader=LiveModelLoader()
        self.journal=ShadowJournal()

    def analyze(self,symbol,m3,m5,m15,version="v0.11"):
        bundle=self.loader.load(symbol,version)
        if bundle is None:
            return {
                "status":"MODEL_NOT_AVAILABLE","symbol":symbol,
                "mode":"RESEARCH","model_version":version,
            }

        quality=bundle.get("quality_gate") or {}
        if not quality.get("passed",False):
            return {
                "status":"MODEL_QUALITY_BLOCKED",
                "symbol":symbol,
                "mode":"SHADOW_RESEARCH",
                "model_version":version,
                "quality_gate":quality.get("status","FAILED"),
            }

        d=add_smc_features(add_market_features(m3))
        d=attach_htf_context(d,m5,m15)
        d["trend_m3"]=d.structure_bias
        d=add_setup_features(d)
        d=add_session_name(d)
        row=d.tail(1).copy()

        if row.empty or int(row.candidate.iloc[0])!=1:
            return {
                "status":"NO_SMC_SETUP","symbol":symbol,
                "mode":"SHADOW_RESEARCH","model_version":version,
            }

        atr=float(row.atr.iloc[0]) if pd.notna(row.atr.iloc[0]) else 0.0
        planned=structural_stop_distance(row.iloc[0],float(row.close.iloc[0]))
        row["stop_distance_atr"]=(
            float(planned/atr) if planned is not None and atr>0 else np.nan
        )

        row["regime"]=bundle["regime"].predict(row)
        filters=bundle.get("learned_filters") or {}
        regime_name=str(int(row.regime.iloc[0]))
        session=str(row.session_name.iloc[0])
        allowed_regimes=filters.get("allowed_regimes") or []
        allowed_sessions=filters.get("allowed_sessions") or []

        if allowed_regimes and regime_name not in allowed_regimes:
            return {
                "status":"FILTERED_OUT","reason":"REGIME",
                "symbol":symbol,"regime":int(row.regime.iloc[0]),
                "session":session,"mode":"SHADOW_RESEARCH",
                "model_version":version,
            }
        if allowed_sessions and session not in allowed_sessions:
            return {
                "status":"FILTERED_OUT","reason":"SESSION",
                "symbol":symbol,"regime":int(row.regime.iloc[0]),
                "session":session,"mode":"SHADOW_RESEARCH",
                "model_version":version,
            }

        features=bundle["features"]
        missing=[c for c in features if c not in row]
        if missing:
            return {
                "status":"FEATURE_MISMATCH","missing":missing,
                "symbol":symbol,"mode":"RESEARCH","model_version":version,
            }

        if row[features].isna().any(axis=None):
            return {
                "status":"FEATURE_NOT_READY","symbol":symbol,
                "mode":"SHADOW_RESEARCH","model_version":version,
            }

        p=float(bundle["model"].predict_proba(row[features])[0])
        rr=float(bundle.get("rr",self.rr))
        threshold=float(bundle.get("threshold",self.threshold))
        expected_r=p*rr-(1-p)
        direction=int(row.signal_direction.iloc[0])
        side="BUY" if direction>0 else "SELL"
        qualified=bool(p>=threshold and expected_r>0)

        event_row={
            "status":"OK",
            "symbol":symbol,
            "timestamp":str(row.timestamp.iloc[0]),
            "side":side,
            "probability":round(p,4),
            "expected_r":round(expected_r,4),
            "regime":int(row.regime.iloc[0]),
            "session":session,
            "smc_confluence":int(row.smc_confluence.iloc[0]),
            "htf_alignment":int(row.htf_alignment.iloc[0]),
            "threshold":round(threshold,4),
            "qualified":qualified,
            "quality_gate":"PASSED",
            "mode":"SHADOW_RESEARCH",
            "model_version":version,
        }

        # The shadow journal contains only qualified signals from models that
        # passed the independent quality gate.
        if qualified:
            self.journal.record(event_row)
        return event_row
