#property strict
#property version   "1.030"
#property script_show_inputs
#property description "Deep XAUUSD historical backfill for AI Market Intelligence V0.12.4. Sends closed M3/M5/M15 candles only."

input string ApiBaseUrl="https://ai-machine-learning-production.up.railway.app";
input string BridgeToken="change-me";
input string MarketSymbolOverride="";
input string ApiSymbol="XAUUSD";
input int M3Bars=50000;
input int M5Bars=30000;
input int M15Bars=10000;
input int BatchSize=300;
input int RetryCount=3;
input int HistoryLoadAttempts=4;

const int REQUIRED_M3_BARS=45000;
const int REQUIRED_M5_BARS=27000;
const int REQUIRED_M15_BARS=9000;

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

bool PostJson(string endpoint,string body,string label)
{
   char data[],result[];
   string responseHeaders;
   string headers="Content-Type: application/json\r\nX-Bridge-Token: "+BridgeToken+"\r\n";
   StringToCharArray(body,data,0,WHOLE_ARRAY,CP_UTF8);
   if(ArraySize(data)>0) ArrayResize(data,ArraySize(data)-1);

   int retries=(RetryCount<1?1:RetryCount);
   for(int attempt=1;attempt<=retries;attempt++)
   {
      ArrayResize(result,0);
      responseHeaders="";
      ResetLastError();
      int code=WebRequest("POST",ApiBaseUrl+endpoint,headers,45000,data,result,responseHeaders);
      if(code>=200 && code<300)
      {
         Print(label," OK HTTP ",code);
         return true;
      }

      Print(label," FAILED attempt=",attempt,"/",retries," HTTP=",code,
            " MT5 error=",GetLastError()," response=",CharArrayToString(result,0,-1,CP_UTF8));
      if(attempt<retries) Sleep(1000*attempt);
   }
   return false;
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

   string label="History backfill "+tfName+" rows="+IntegerToString(toIndex-fromIndex+1);
   return PostJson("/api/history/batch",body,label);
}

bool SendTimeframe(ENUM_TIMEFRAMES tf,string tfName,int requestedBars,int minimumBars)
{
   string marketSymbol=(MarketSymbolOverride==""?_Symbol:MarketSymbolOverride);
   if(!SymbolSelect(marketSymbol,true))
   {
      Print("SymbolSelect failed symbol=",marketSymbol," error=",GetLastError());
      return false;
   }

   if(requestedBars<minimumBars)
   {
      Print("ERROR ",tfName,": requestedBars=",requestedBars,
            " is below V0.12.4 minimum=",minimumBars);
      return false;
   }

   MqlRates rates[];
   ArraySetAsSeries(rates,false);

   int attempts=(HistoryLoadAttempts<1?1:HistoryLoadAttempts);
   int copied=0;
   for(int attempt=1;attempt<=attempts;attempt++)
   {
      ArrayResize(rates,0);
      ResetLastError();
      copied=CopyRates(marketSymbol,tf,1,requestedBars,rates);
      if(copied>=minimumBars)
         break;

      Print(tfName," history not ready attempt=",attempt,"/",attempts,
            " requested=",requestedBars," copied=",copied,
            " minimum=",minimumBars," error=",GetLastError());
      if(attempt<attempts) Sleep(1500*attempt);
   }

   if(copied<minimumBars)
   {
      Print("ERROR ",tfName,": broker/terminal supplied only ",copied,
            " bars; V0.12.4 requires at least ",minimumBars,
            ". Open the timeframe chart and load more history, then rerun.");
      return false;
   }

   int batch=(BatchSize<50?50:MathMin(BatchSize,500));
   int totalBatches=(copied+batch-1)/batch;
   Print("Starting ",tfName," V0.12.4 deep backfill. requested=",requestedBars,
         " copied=",copied," minimum=",minimumBars,
         " batch=",batch," total_batches=",totalBatches);

   int batchNo=0;
   for(int start=0;start<copied;start+=batch)
   {
      int finish=MathMin(start+batch-1,copied-1);
      batchNo++;
      if(!PostBatch(tfName,rates,start,finish))
         return false;

      double pct=100.0*(double)(finish+1)/(double)copied;
      if(batchNo==1 || batchNo==totalBatches || batchNo%10==0)
         Print(tfName," progress ",DoubleToString(pct,1),"% (",finish+1,"/",copied,")");
      Sleep(120);
   }

   Print("Completed ",tfName," deep backfill rows=",copied,
         " minimum requirement=",minimumBars," PASSED");
   return true;
}

bool NotifyComplete()
{
   string body="{\"symbol\":\""+JsonEscape(ApiSymbol)+"\"}";
   return PostJson("/api/history/complete",body,"History completion notification");
}

void OnStart()
{
   if(BridgeToken=="" || BridgeToken=="change-me")
   {
      Print("ERROR: set BridgeToken to the same token configured on Railway.");
      return;
   }

   Print("V0.12.4 DEEP HISTORY upload target: M3=",M3Bars,
         " M5=",M5Bars," M15=",M15Bars);
   Print("V0.12.4 MINIMUM required: M3=",REQUIRED_M3_BARS,
         " M5=",REQUIRED_M5_BARS," M15=",REQUIRED_M15_BARS);

   bool ok15=SendTimeframe(PERIOD_M15,"M15",M15Bars,REQUIRED_M15_BARS);
   bool ok5 =SendTimeframe(PERIOD_M5 ,"M5" ,M5Bars,REQUIRED_M5_BARS);
   bool ok3 =SendTimeframe(PERIOD_M3 ,"M3" ,M3Bars,REQUIRED_M3_BARS);

   if(ok15 && ok5 && ok3)
   {
      if(NotifyComplete())
         Print("XAUUSD V0.12.4 deep historical backfill COMPLETE. Server was notified to validate/start training.");
      else
         Print("Candles uploaded, but completion notification failed. Re-run script or call completion after connection recovers.");
   }
   else
      Print("XAUUSD historical backfill stopped before completion. Training was NOT notified. Check Experts/Journal, broker history availability, and WebRequest whitelist.");
}
