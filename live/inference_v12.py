from live.inference import LiveInference as BaseLiveInference


class LiveInference(BaseLiveInference):
    """V0.12 live shadow inference.

    Uses the v0.12.0 model artifacts and RR 1:2 bundle. Broker execution remains
    disabled; only models that pass the independent quality gate can be journaled.
    """

    def __init__(self, threshold=.35, rr=2.0):
        super().__init__(threshold=threshold, rr=rr)

    def analyze(self, symbol, m3, m5, m15, version="v0.12.0"):
        return super().analyze(symbol,m3,m5,m15,version=version)
