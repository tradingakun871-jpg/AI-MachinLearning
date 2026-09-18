from live.buffer import CandleBuffer
from live.inference import LiveInference

class ShadowService:
    def __init__(self):
        self.buffer=CandleBuffer()
        self.inference=LiveInference()

    def ingest(self,symbol,timeframe,candle):
        rows=self.buffer.ingest(symbol,timeframe,candle)
        result={"accepted":True,"rows":rows,"ready":self.buffer.ready(symbol)}
        if timeframe.upper()=="M3" and result["ready"]:
            result["analysis"]=self.inference.analyze(
                symbol,
                self.buffer.frame(symbol,"M3"),
                self.buffer.frame(symbol,"M5"),
                self.buffer.frame(symbol,"M15"),
            )
        return result

    def recent(self,limit=100):
        return self.inference.journal.recent(limit)

    def status(self,symbols=("XAUUSD","BTCUSD","BTCUSDT")):
        return {
            "symbols":[self.buffer.summary(symbol) for symbol in symbols],
            "shadow_signal_count":len(self.inference.journal.rows),
        }
