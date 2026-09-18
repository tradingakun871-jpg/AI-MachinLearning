#property strict
#property version   "1.010"
#property script_show_inputs
#property description "One-time XAUUSD historical backfill to AI Market Intelligence. Sends closed M3/M5/M15 candles only."

input string ApiBaseUrl="https://ai-machine-learning-production.up.railway.app";
input string BridgeToken="change-me";
input string MarketSymbolOverride="";
input string ApiSymbol="XAUUSD";
input int M3Bars=15000;
input int M5Bars=9000;
input int M15Bars=3000;
input int BatchSize=150;

string JsonEscape(string s)
{
   StringReplace(s,"\\","\\\\");
   StringReplace(s,"\"","\\\"");
   return s;
}

datetime BrokerTimeToUtc(datetime brokerTime)
{
   long offset=(long)TimeTradeServer()-(long)TimeGMT();
   return (datetime)((long)brokerTime-offset);
}

string IsoUtc(datetime brokerTime)
{
   datetime utc=BrokerTimeToUtc(brokerTime);
   MqlDateTime x;
   TimeToStruct(utc,x);
   return StringFormat("%04d-%02d-%02dT%02d:%02d:%02dZ",x.year,x.mon,x.day,x.hour,x.min,x.sec);
}

bool PostBatch(string tfName,MqlRates &rates[],int fromIndex,int toIndex)
{
   string body="{\"symbol\":\""+JsonEscape(ApiSymbol)+"\",\"timeframe\":\""+tfName+"\",\"candles\":[";
   bool first=true;

   for(int i=fromIndex;i<=toIndex;i++)
   {
      if(!first) body+=",";
      first=false;
      body+=StringFormat(
         "{\"timestamp\":\"%s\",\"open\":%.10f,\"high\":%.10f,\"low\":%.10f,\"close\":%.10f,\"volume\":%I64d,\"spread\":%d}",
         IsoUtc(rates[i].time),
         rates[i].open,rates[i].high,rates[i].low,rates[i].close,
         rates[i].tick_volume,rates[i].spread
      );
   }
   body+="]}";

   char data[],result[];
   string responseHeaders;
   string headers="Content-Type: application/json\r\nX-Bridge-Token: "+BridgeToken+"\r\n";
   StringToCharArray(body,data,0,WHOLE_ARRAY,CP_UTF8);
   if(ArraySize(data)>0) ArrayResize(data,ArraySize(data)-1);

   ResetLastError();
   int code=WebRequest("POST",ApiBaseUrl+"/api/history/batch",headers,30000,data,result,responseHeaders);
   if(code>=200 && code<300)
   {
      Print("History backfill OK ",tfName," rows=",toIndex-fromIndex+1," HTTP ",code);
      return true;
   }

   Print("History backfill FAILED ",tfName," HTTP=",code," MT5 error=",GetLastError()," response=",CharArrayToString(result,0,-1,CP_UTF8));
   return false;
}

bool SendTimeframe(ENUM_TIMEFRAMES tf,string tfName,int requestedBars)
{
   string marketSymbol=(MarketSymbolOverride==""?_Symbol:MarketSymbolOverride);
   MqlRates rates[];
   ArraySetAsSeries(rates,false);

   ResetLastError();
   int copied=CopyRates(marketSymbol,tf,1,requestedBars,rates);
   if(copied<=0)
   {
      Print("CopyRates failed ",tfName," symbol=",marketSymbol," error=",GetLastError());
      return false;
   }

   int batch=(BatchSize<20?20:BatchSize);
   Print("Starting ",tfName," backfill. copied=",copied," batch=",batch);

   for(int start=0;start<copied;start+=batch)
   {
      int finish=MathMin(start+batch-1,copied-1);
      if(!PostBatch(tfName,rates,start,finish))
         return false;
      Sleep(150);
   }

   Print("Completed ",tfName," backfill rows=",copied);
   return true;
}

void OnStart()
{
   if(BridgeToken=="" || BridgeToken=="change-me")
   {
      Print("ERROR: set BridgeToken to the same token configured on Railway.");
      return;
   }

   bool ok15=SendTimeframe(PERIOD_M15,"M15",M15Bars);
   bool ok5 =SendTimeframe(PERIOD_M5 ,"M5" ,M5Bars);
   bool ok3 =SendTimeframe(PERIOD_M3 ,"M3" ,M3Bars);

   if(ok15 && ok5 && ok3)
      Print("XAUUSD historical backfill COMPLETE. Server can now start XAU model training.");
   else
      Print("XAUUSD historical backfill stopped with error. Check Experts/Journal and WebRequest whitelist.");
}
