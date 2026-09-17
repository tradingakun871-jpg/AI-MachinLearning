import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class RegimeClassifier:
    """Unsupervised market-regime classifier fitted only on training data."""
    def __init__(self, n_regimes=4, random_state=42):
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=20)

    def fit(self, frame):
        cols = ["volatility_z", "atr_norm", "return_3", "range_atr"]
        x = frame[cols].fillna(0.0).to_numpy(dtype=float)
        self.model.fit(self.scaler.fit_transform(x))
        return self

    def predict(self, frame):
        cols = ["volatility_z", "atr_norm", "return_3", "range_atr"]
        x = frame[cols].fillna(0.0).to_numpy(dtype=float)
        return self.model.predict(self.scaler.transform(x))
