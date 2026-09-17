# V0.3 Data Engine

Required OHLC CSV columns: `timestamp,open,high,low,close` with optional `volume,spread`.

For XAUUSD export M3/M5/M15 from the same broker/MT5 account to reduce feed mismatch. Example:

```bash
python data/mt5_export.py --symbol XAUUSD --timeframe M3 --start 2024-01-01 --end 2026-09-01 --output datasets/xau_m3.csv
python data/mt5_export.py --symbol XAUUSD --timeframe M5 --start 2024-01-01 --end 2026-09-01 --output datasets/xau_m5.csv
python data/mt5_export.py --symbol XAUUSD --timeframe M15 --start 2024-01-01 --end 2026-09-01 --output datasets/xau_m15.csv
python data/dataset.py --m3 datasets/xau_m3.csv --m5 datasets/xau_m5.csv --m15 datasets/xau_m15.csv --rr 3 --output datasets/xau_candidates.csv
python -m ml.train datasets/xau_candidates.csv --output artifacts/xau-v0.3
```

The higher-timeframe joins are backward-only. Labels use future bars only after the candidate timestamp. Same-bar TP+SL collisions are excluded rather than resolved optimistically.

BTC should use a separate dataset/model artifact. Do not mix XAUUSD and BTC training rows into one model in V0.3.
