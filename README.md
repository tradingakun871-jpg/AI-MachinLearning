# AI-MachinLearning — Market Intelligence Agent

AI/ML trading research platform for XAUUSD and BTC. The system is designed around market-regime detection, multi-timeframe structure, feature engineering, probabilistic ML scoring, expectancy/risk controls, journaling, and MT5 integration.

> Default mode is **SHADOW**. Live order execution must be explicitly enabled only after out-of-sample and paper-trading validation. No model guarantees profit.

## Architecture

`Market Data -> Feature Engine -> Structure/SMC Engine -> ML Probability -> Decision Engine -> Risk Manager -> Shadow/Execution -> Journal`

Primary XAUUSD execution timeframe: M3, with M5/M15 context. BTC is modeled separately.

## Quick start

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `/docs` for the API and `/` for the dashboard.

## Initial API

- `GET /health`
- `GET /api/status`
- `POST /api/analyze`
- `POST /api/market/candle`
- `GET /api/signals`

## Roadmap

1. Data ingestion and feature store
2. M15/M5/M3 market structure and SMC features
3. Probability model + calibration
4. Walk-forward/backtest engine
5. Shadow/paper trading
6. MT5 Bridge + execution guardrails
7. Telegram alerts and production dashboard
