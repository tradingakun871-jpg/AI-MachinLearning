# V0.7 MT5 Live Data Bridge

This bridge is data-only. It does not call MT5 trading functions and cannot open, modify, or close positions.

1. Copy `AI_Market_Data_Bridge.mq5` into `MQL5/Experts/` and compile it in MetaEditor.
2. In MT5 open Tools → Options → Expert Advisors and allow WebRequest for the API base URL.
3. Set the server environment variable `BRIDGE_TOKEN` to a strong random secret. Set the exact same value in the EA input `BridgeToken`. Do not commit the real token.
4. Set `ApiBaseUrl` to the deployed API URL. Attach the EA to the XAUUSD chart. The EA reads closed M3, M5 and M15 bars directly, independent of the chart timeframe.
5. Check `/health`, then `/api/live/signals`. The buffer requires historical/live rows before inference becomes ready.

The endpoint rejects requests when `BRIDGE_TOKEN` is unset/default or does not match. Model inference also refuses to fabricate predictions when a trained model artifact is absent.
