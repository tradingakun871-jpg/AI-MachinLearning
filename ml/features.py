FEATURE_COLUMNS = [
    "trend_m15","trend_m5","trend_m3","mss","bos","choch","structure_bias",
    "liquidity_sweep","liquidity_sweep_direction","order_block","ob_direction",
    "fvg","fvg_direction","fvg_size","dealing_range_position","discount","premium",
    "session_asia","session_london","session_newyork","session_overlap",
    "atr_norm","volatility_z","spread_atr","return_1","return_3","range_atr","hour_sin","hour_cos",

    # Price action: causal candle anatomy, rejection, breakout, momentum and compression.
    "pa_body_ratio","pa_upper_wick_ratio","pa_lower_wick_ratio","pa_close_location",
    "pa_candle_direction","pa_engulfing_direction","pa_inside_bar","pa_outside_bar",
    "pa_rejection_direction","pa_breakout_direction","pa_displacement_direction",
    "pa_momentum_3_atr","pa_trend_efficiency","pa_compression","pa_direction","pa_strength",

    # Order flow: true delta when supplied, otherwise an explicitly-marked proxy.
    "orderflow_available","orderflow_is_proxy","orderflow_source_quality",
    "of_volume_z","of_delta_ratio","of_cvd_20","of_absorption_direction",
    "orderflow_direction","orderflow_strength",

    "signal_direction","signal_mss","signal_bos","signal_sweep","signal_ob","signal_fvg",
    "signal_pa","signal_pa_breakout","signal_pa_rejection","signal_orderflow",
    "primary_structure_count","imbalance_count","smc_confluence","price_action_confluence",
    "orderflow_confluence","context_confluence","htf_alignment",
    "distance_to_swing_atr","stop_distance_atr"
]


def build_features(row):
    return {name: float(row.get(name, 0.0)) for name in FEATURE_COLUMNS}
