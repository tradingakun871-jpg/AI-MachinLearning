#property strict
#property version   "0.70"
#property description "Research-only MT5 candle bridge. Sends closed OHLC data; never places orders."

input string ApiBaseUrl="http://127.0.0.1:8000";
input string BridgeToken="change-me";
input string SymbolOverride="";
input int TimerSeconds=2;

datetime lastM3=0,lastM5=0,lastM15=0;

string JsonEscape(string s){StringReplace(s,"\\","\\\\");StringReplace(s,"\"","\\\"");return s;}
string IsoTime(datetime t){MqlDateTime x;TimeToStruct(t,x);return StringFormat("%04d-%02d-%02dT%02d:%02d:%02dZ",x.year,x.mon,x.day,x.hour,x.min,x.sec);}

bool SendClosedBar(ENUM_TIMEFRAMES tf,string tfName,datetime &lastSent){
 string sym=SymbolOverride==""?_Symbol:SymbolOverride; MqlRates r[]; ArraySetAsSeries(r,true);
 if(CopyRates(sym,tf,1,1,r)!=1) return false;
 if(r[0].time==lastSent) return true;
 string body=StringFormat("{\"symbol\":\"%s\",\"timeframe\":\"%s\",\"timestamp\":\"%s\",\"open\":%.10f,\"high\":%.10f,\"low\":%.10f,\"close\":%.10f,\"volume\":%I64d,\"spread\":%d}",JsonEscape(sym),tfName,IsoTime(r[0].time),r[0].open,r[0].high,r[0].low,r[0].close,r[0].tick_volume,r[0].spread);
 char data[],result[]; string headers="Content-Type: application/json\r\nX-Bridge-Token: "+BridgeToken+"\r\n",responseHeaders; StringToCharArray(body,data,0,WHOLE_ARRAY,CP_UTF8); if(ArraySize(data)>0) ArrayResize(data,ArraySize(data)-1);
 int code=WebRequest("POST",ApiBaseUrl+"/api/live/candle",headers,5000,data,result,responseHeaders);
 if(code>=200 && code<300){lastSent=r[0].time; return true;} Print("Bridge HTTP error ",code," MT5=",GetLastError()," tf=",tfName); return false;
}

int OnInit(){EventSetTimer(MathMax(1,TimerSeconds));return INIT_SUCCEEDED;}
void OnDeinit(const int reason){EventKillTimer();}
void OnTimer(){SendClosedBar(PERIOD_M15,"M15",lastM15);SendClosedBar(PERIOD_M5,"M5",lastM5);SendClosedBar(PERIOD_M3,"M3",lastM3);}
