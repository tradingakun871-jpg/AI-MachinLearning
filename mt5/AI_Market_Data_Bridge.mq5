#property strict
#property version   "0.81"
#property description "XAUUSD MT5 -> AI Market Intelligence bridge. Sends CLOSED M3/M5/M15 candles only; never places orders."

input string ApiBaseUrl="https://ai-machine-learning-production.up.railway.app";
input string BridgeToken="change-me";
input string MarketSymbolOverride="";
input string ApiSymbol="XAUUSD";
input int TimerSeconds=2;

datetime lastM3=0,lastM5=0,lastM15=0;

string JsonEscape(string s){StringReplace(s,"\\","\\\\");StringReplace(s,"\"","\\\"");return s;}

datetime BrokerTimeToUtc(datetime brokerTime)
{
   long offset=(long)TimeTradeServer()-(long)TimeGMT();
   return (datetime)((long)brokerTime-offset);
}

string IsoUtc(datetime brokerTime)
{
   datetime utc=BrokerTimeToUtc(brokerTime);
   MqlDateTime x; TimeToStruct(utc,x);
   return StringFormat("%04d-%02d-%02dT%02d:%02d:%02dZ",x.year,x.mon,x.day,x.hour,x.min,x.sec);
}

bool SendClosedBar(ENUM_TIMEFRAMES tf,string tfName,datetime &lastSent)
{
   string marketSymbol=(MarketSymbolOverride==""?_Symbol:MarketSymbolOverride);
   MqlRates r[]; ArraySetAsSeries(r,true);
   ResetLastError();
   if(CopyRates(marketSymbol,tf,1,1,r)!=1){Print("Bridge CopyRates failed tf=",tfName," symbol=",marketSymbol," error=",GetLastError());return false;}
   if(r[0].time==lastSent) return true;

   string body=StringFormat(
      "{\"symbol\":\"%s\",\"timeframe\":\"%s\",\"timestamp\":\"%s\",\"open\":%.10f,\"high\":%.10f,\"low\":%.10f,\"close\":%.10f,\"volume\":%I64d,\"spread\":%d}",
      JsonEscape(ApiSymbol),tfName,IsoUtc(r[0].time),r[0].open,r[0].high,r[0].low,r[0].close,r[0].tick_volume,r[0].spread
   );

   char data[],result[];
   string headers="Content-Type: application/json\r\nX-Bridge-Token: "+BridgeToken+"\r\n",responseHeaders;
   StringToCharArray(body,data,0,WHOLE_ARRAY,CP_UTF8);
   if(ArraySize(data)>0) ArrayResize(data,ArraySize(data)-1);

   ResetLastError();
   int code=WebRequest("POST",ApiBaseUrl+"/api/live/candle",headers,10000,data,result,responseHeaders);
   if(code>=200 && code<300){lastSent=r[0].time;Print("Bridge OK ",tfName," ",marketSymbol," -> ",ApiSymbol," HTTP ",code);return true;}
   Print("Bridge HTTP error code=",code," mt5_error=",GetLastError()," tf=",tfName," response=",CharArrayToString(result,0,-1,CP_UTF8));
   return false;
}

int OnInit(){int seconds=(TimerSeconds<1?1:TimerSeconds);EventSetTimer(seconds);return INIT_SUCCEEDED;}
void OnDeinit(const int reason){EventKillTimer();}
void OnTimer(){SendClosedBar(PERIOD_M15,"M15",lastM15);SendClosedBar(PERIOD_M5,"M5",lastM5);SendClosedBar(PERIOD_M3,"M3",lastM3);}
