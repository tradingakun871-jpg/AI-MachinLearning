FEATURE_COLUMNS = [
    "trend_m15","trend_m5","trend_m3","mss","bos","choch","structure_bias",
    "liquidity_sweep","liquidity_sweep_direction","order_block","ob_direction",
    "fvg","fvg_direction","fvg_size","dealing_range_position","discount","premium",
    "session_asia","session_london","session_newyork","session_overlap",
    "atr_norm","volatility_z","spread_atr","return_1","return_3","range_atr","hour_sin","hour_cos"
]

def build_features(row):
    return {name: float(row.get(name,0.0)) for name in FEATURE_COLUMNS}
