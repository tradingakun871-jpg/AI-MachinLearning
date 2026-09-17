FEATURE_COLUMNS = [
    "trend_m15", "trend_m5", "trend_m3", "mss",
    "liquidity_sweep", "order_block", "fvg",
    "atr_norm", "volatility_z", "spread_atr",
    "return_1", "return_3", "range_atr", "hour_sin", "hour_cos"
]

def build_features(row):
    return {name: float(row.get(name, 0.0)) for name in FEATURE_COLUMNS}
