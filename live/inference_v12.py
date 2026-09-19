from live.inference import LiveInference as BaseLiveInference


class LiveInference(BaseLiveInference):
    """V0.12.5 live shadow inference for RR 1:2.

    Uses SMC + Price Action + conditional Order Flow + Regime with deep-history,
    stable-context and development-only directional walk-forward gating. Broker
    execution remains disabled. Only artifacts that pass the independent final
    quality gate may be journaled as shadow signals.
    """

    def __init__(self, threshold=.34, rr=2.0):
        super().__init__(threshold=threshold, rr=rr)

    def analyze(self, symbol, m3, m5, m15, version="v0.12.5"):
        return super().analyze(symbol,m3,m5,m15,version=version)
