from typing import Literal
from pydantic import BaseModel, Field

class AnalysisRequest(BaseModel):
    symbol: Literal["XAUUSD", "BTCUSD", "BTCUSDT"] = "XAUUSD"
    timeframe: str = "M3"
    close: float
    atr: float = Field(gt=0)
    trend_m15: Literal[-1, 0, 1] = 0
    trend_m5: Literal[-1, 0, 1] = 0
    trend_m3: Literal[-1, 0, 1] = 0
    mss: Literal[-1, 0, 1] = 0
    liquidity_sweep: bool = False
    order_block: bool = False
    fvg: bool = False
    volatility_z: float = 0.0
    spread_atr: float = 0.0

class AnalysisResponse(BaseModel):
    symbol: str
    side: str
    probability: float
    expectancy_r: float
    action: str
    entry: float
    stop: float | None = None
    target: float | None = None
    rr: float
    reasons: list[str]
    mode: str
