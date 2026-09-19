import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


REGIME_COLUMNS = [
    "volatility_z",
    "atr_norm",
    "return_3",
    "range_atr",
    "pa_trend_efficiency",
    "pa_compression",
    "pa_strength",
    "of_volume_z",
    "orderflow_strength",
    "orderflow_source_quality",
]


class RegimeClassifier:
    """Unsupervised market-regime classifier fitted only on training data.

    V0.12.2 extends the regime state with causal price-action and optional
    order-flow context. Missing order-flow inputs are represented as zeros so
    XAU tick-volume and BTC exchange-volume histories remain compatible.
    """

    def __init__(self, n_regimes=4, random_state=42):
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=20)
        self.columns = list(REGIME_COLUMNS)

    def _matrix(self, frame):
        cols = []
        for name in self.columns:
            if name in frame:
                values = pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
            else:
                values = pd.Series(0.0, index=frame.index)
            cols.append(values.to_numpy(dtype=float))
        return np.column_stack(cols)

    def fit(self, frame):
        x = self._matrix(frame)
        self.model.fit(self.scaler.fit_transform(x))
        return self

    def predict(self, frame):
        x = self._matrix(frame)
        return self.model.predict(self.scaler.transform(x))
