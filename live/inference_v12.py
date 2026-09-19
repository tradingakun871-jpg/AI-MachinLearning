from live.inference import LiveInference as BaseLiveInference


class LiveInference(BaseLiveInference):
    """V0.12.1 live shadow inference with RR 1:2 and meta-precision veto."""

    def __init__(self, threshold=.35, rr=2.0):
        super().__init__(threshold=threshold, rr=rr)

    def analyze(self, symbol, m3, m5, m15, version="v0.12.1"):
        return super().analyze(symbol,m3,m5,m15,version=version)
