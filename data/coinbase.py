import asyncio
import math
from datetime import datetime, timezone, timedelta

import httpx
import pandas as pd


class CoinbaseBTCFeed:
    """Public Coinbase BTC-USD market-data feed for research/shadow use."""

    BASE_URL = "https://api.exchange.coinbase.com"
    PRODUCT_ID = "BTC-USD"
    SYMBOL = "BTCUSD"

    def __init__(self, shadow_service, poll_seconds=60):
        self.shadow = shadow_service
        self.poll_seconds = max(30, int(poll_seconds))
        self.task = None
        self.last_success = None
        self.last_error = None
        self.last_sync = None
        self.last_analyzed_m3 = None
        self.running = False

    def status(self):
        return {
            "source": "Coinbase Exchange",
            "product_id": self.PRODUCT_ID,
            "symbol": self.SYMBOL,
            "running": self.running,
            "poll_seconds": self.poll_seconds,
            "last_sync": self.last_sync,
            "last_success": self.last_success,
            "last_error": self.last_error,
        }

    async def start(self):
        if self.task and not self.task.done():
            return
        self.running = True
        self.task = asyncio.create_task(self._run(), name="coinbase-btc-feed")

    async def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def _run(self):
        while self.running:
            try:
                await self.sync()
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(self.poll_seconds)

    async def sync(self):
        self.last_sync = datetime.now(timezone.utc).isoformat()

        need_bootstrap = not self.shadow.buffer.ready(self.SYMBOL)
        one_minute_points = 960 if need_bootstrap else 360

        async with httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=20.0,
            headers={
                "Accept": "application/json",
                "User-Agent": "AI-Market-Intelligence/0.11.1",
            },
        ) as client:
            m1 = await self._fetch_candles(client, 60, one_minute_points)
            m5 = await self._fetch_candles(client, 300, 320)
            m15 = await self._fetch_candles(client, 900, 320)

        m3 = self._resample_m3(m1)

        # Load HTF first so M3 inference always sees M5/M15 context.
        self._load_frame("M5", m5.tail(320))
        self._load_frame("M15", m15.tail(320))
        self._load_frame("M3", m3.tail(340))

        analysis = None
        m3_frame = self.shadow.buffer.frame(self.SYMBOL, "M3")
        if self.shadow.buffer.ready(self.SYMBOL) and not m3_frame.empty:
            latest_ts = str(m3_frame.iloc[-1]["timestamp"])
            if latest_ts != self.last_analyzed_m3:
                analysis = self.shadow.inference.analyze(
                    self.SYMBOL,
                    self.shadow.buffer.frame(self.SYMBOL, "M3"),
                    self.shadow.buffer.frame(self.SYMBOL, "M5"),
                    self.shadow.buffer.frame(self.SYMBOL, "M15"),
                )
                self.last_analyzed_m3 = latest_ts

        self.last_success = datetime.now(timezone.utc).isoformat()
        return {
            "ok": True,
            "source": "Coinbase Exchange",
            "product_id": self.PRODUCT_ID,
            "rows": {
                "M3": int(len(m3)),
                "M5": int(len(m5)),
                "M15": int(len(m15)),
            },
            "analysis": analysis,
        }

    async def _fetch_candles(self, client, granularity, desired_points):
        max_per_request = 300
        pages = max(1, math.ceil(desired_points / max_per_request))
        now = datetime.now(timezone.utc)
        end = now.replace(second=0, microsecond=0)
        frames = []

        for _ in range(pages):
            start = end - timedelta(seconds=granularity * max_per_request)
            response = await client.get(
                f"/products/{self.PRODUCT_ID}/candles",
                params={
                    "granularity": granularity,
                    "start": start.isoformat().replace("+00:00", "Z"),
                    "end": end.isoformat().replace("+00:00", "Z"),
                },
            )
            response.raise_for_status()
            rows = response.json()
            if not isinstance(rows, list):
                raise RuntimeError("unexpected Coinbase candle response")

            parsed = []
            for row in rows:
                if not isinstance(row, list) or len(row) < 6:
                    continue
                parsed.append(
                    {
                        "timestamp": pd.to_datetime(int(row[0]), unit="s", utc=True),
                        "low": float(row[1]),
                        "high": float(row[2]),
                        "open": float(row[3]),
                        "close": float(row[4]),
                        "volume": float(row[5]),
                        "spread": 0.0,
                    }
                )

            if parsed:
                frames.append(pd.DataFrame(parsed))
            end = start

        if not frames:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "spread"])

        df = (
            pd.concat(frames, ignore_index=True)
            .drop_duplicates("timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # Coinbase may include the currently forming candle. Research pipeline uses closed candles only.
        cutoff = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(seconds=granularity)
        df = df[df["timestamp"] <= cutoff].copy()
        return df.tail(desired_points).reset_index(drop=True)

    def _resample_m3(self, m1):
        if m1.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "spread"])

        d = m1.copy().set_index("timestamp").sort_index()
        agg = d.resample("3min", label="left", closed="left").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "spread": "mean",
            }
        )
        counts = d["close"].resample("3min", label="left", closed="left").count()
        agg["count"] = counts
        agg = agg[agg["count"] == 3].drop(columns=["count"]).dropna().reset_index()

        cutoff = pd.Timestamp(datetime.now(timezone.utc)) - pd.Timedelta(minutes=3)
        agg = agg[agg["timestamp"] <= cutoff].copy()
        return agg.reset_index(drop=True)

    def _load_frame(self, timeframe, frame):
        if frame.empty:
            return
        for row in frame.to_dict("records"):
            timestamp = row["timestamp"]
            if hasattr(timestamp, "isoformat"):
                timestamp = timestamp.isoformat()
            self.shadow.buffer.ingest(
                self.SYMBOL,
                timeframe,
                {
                    "timestamp": timestamp,
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row.get("volume", 0.0)),
                    "spread": float(row.get("spread", 0.0)),
                },
            )
