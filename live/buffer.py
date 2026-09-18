import pandas as pd

class CandleBuffer:
    def __init__(self,max_rows=3000):
        self.max_rows=max_rows
        self.data={}

    def ingest(self,symbol,timeframe,candle):
        key=(symbol.upper(),timeframe.upper())
        df=self.data.get(key,pd.DataFrame())
        row=pd.DataFrame([candle])
        row["timestamp"]=pd.to_datetime(row.timestamp,utc=True)
        df=(pd.concat([df,row],ignore_index=True)
              .drop_duplicates("timestamp",keep="last")
              .sort_values("timestamp")
              .tail(self.max_rows)
              .reset_index(drop=True))
        self.data[key]=df
        return len(df)

    def frame(self,symbol,timeframe):
        return self.data.get((symbol.upper(),timeframe.upper()),pd.DataFrame()).copy()

    def ready(self,symbol,min_rows=300):
        return all(len(self.frame(symbol,tf))>=min_rows for tf in ("M3","M5","M15"))

    def symbols(self):
        return sorted({symbol for symbol,_ in self.data.keys()})

    def summary(self,symbol):
        result={"symbol":symbol.upper(),"ready":self.ready(symbol),"timeframes":{}}
        for tf in ("M3","M5","M15"):
            df=self.frame(symbol,tf)
            info={"rows":int(len(df)),"last_timestamp":None,"last_close":None}
            if not df.empty:
                info["last_timestamp"]=str(df.iloc[-1]["timestamp"])
                if "close" in df.columns:
                    try: info["last_close"]=float(df.iloc[-1]["close"])
                    except Exception: pass
            result["timeframes"][tf]=info
        return result
