import os
from math import exp
from app.schemas import AnalysisRequest, AnalysisResponse

class MarketAgent:
    def __init__(self):
        self.mode = "RESEARCH"
        self.rr = float(os.getenv("DEFAULT_RR", "3.0"))
        self.signals = []
        self.candles = []

    def status(self):
        return {"mode": self.mode, "symbols": ["XAUUSD", "BTCUSD", "BTCUSDT"], "signals": len(self.signals), "candles": len(self.candles), "model": "baseline-research-v0"}

    def ingest(self, payload: dict):
        self.candles.append(payload)
        self.candles = self.candles[-10000:]
        return {"accepted": True, "stored": len(self.candles)}

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + exp(-max(-20.0, min(20.0, x))))

    def analyze(self, r: AnalysisRequest) -> AnalysisResponse:
        score = 0.9*r.trend_m15 + 0.7*r.trend_m5 + 0.5*r.trend_m3 + r.mss
        side = "BUY" if score > 0 else "SELL" if score < 0 else "WAIT"
        sign = 1 if side == "BUY" else -1 if side == "SELL" else 0
        confluence = 0.45*r.liquidity_sweep + 0.35*r.order_block + 0.30*r.fvg
        probability = self.sigmoid(-1.35 + 0.58*abs(score) + confluence) if sign else 0.0
        expectancy = probability*self.rr - (1-probability) if sign else -1.0
        reasons = [n for ok,n in [(r.liquidity_sweep,"liquidity sweep"),(r.order_block,"order block"),(r.fvg,"FVG")] if ok]
        result = AnalysisResponse(symbol=r.symbol, side=side, probability=round(probability,4), expectancy_r=round(expectancy,4), action="RESEARCH_ONLY", entry=r.close, stop=None, target=None, rr=self.rr, reasons=reasons, mode=self.mode)
        self.signals.append(result.model_dump())
        return result
