"""Run locally on a Windows machine with MetaTrader 5 installed.
Exports chronological OHLCV/spread CSVs for the V0.3 dataset builder.
"""
import argparse
from pathlib import Path
import pandas as pd

TF_NAMES={"M3":3,"M5":5,"M15":15}

def export(symbol, timeframe, start, end, output):
    try:
        import MetaTrader5 as mt5
    except ImportError as e:
        raise SystemExit("Install MetaTrader5 package on the MT5 Windows host") from e
    if not mt5.initialize(): raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    tf={"M3":mt5.TIMEFRAME_M3,"M5":mt5.TIMEFRAME_M5,"M15":mt5.TIMEFRAME_M15}[timeframe]
    rates=mt5.copy_rates_range(symbol,tf,pd.Timestamp(start,tz="UTC").to_pydatetime(),pd.Timestamp(end,tz="UTC").to_pydatetime())
    mt5.shutdown()
    if rates is None or len(rates)==0: raise SystemExit("No rates returned; check symbol name/history")
    d=pd.DataFrame(rates).rename(columns={"time":"timestamp","tick_volume":"volume"})
    d["timestamp"]=pd.to_datetime(d.timestamp,unit="s",utc=True)
    Path(output).parent.mkdir(parents=True,exist_ok=True); d.to_csv(output,index=False); print({"symbol":symbol,"timeframe":timeframe,"rows":len(d),"output":output})

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--symbol",required=True); p.add_argument("--timeframe",choices=TF_NAMES,required=True); p.add_argument("--start",required=True); p.add_argument("--end",required=True); p.add_argument("--output",required=True); a=p.parse_args(); export(a.symbol,a.timeframe,a.start,a.end,a.output)
