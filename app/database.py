import os
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, JSON, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

_engine = None
_Session = None


def _database_url():
    url = os.getenv("DATABASE_URL", "").strip()
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    elif url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    return url


class Base(DeclarativeBase):
    pass


class ShadowSignal(Base):
    __tablename__ = "shadow_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    event_time: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    side: Mapped[str | None] = mapped_column(String(12), nullable=True)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_r: Mapped[float | None] = mapped_column(Float, nullable=True)
    regime: Mapped[int | None] = mapped_column(Integer, nullable=True)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    qualified: Mapped[bool | None] = mapped_column(Boolean, nullable=True, index=True)
    mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON)


class TrainingState(Base):
    __tablename__ = "training_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    symbol: Mapped[str] = mapped_column(String(24), index=True)
    version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(64), index=True)
    started_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    finished_at: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dataset_rows: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)


def init_database():
    global _engine, _Session
    url = _database_url()
    if not url:
        return False
    if _engine is None:
        _engine = create_engine(url, pool_pre_ping=True, pool_recycle=300)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False)
        Base.metadata.create_all(_engine)
    return True


def database_health():
    try:
        if not init_database():
            return {"configured": False, "connected": False}
        with _Session() as s:
            s.execute(select(1))
        return {"configured": True, "connected": True}
    except Exception as exc:
        return {"configured": True, "connected": False, "error": f"{type(exc).__name__}: {exc}"}


def _to_float(value):
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _to_int(value):
    try:
        return None if value is None else int(value)
    except Exception:
        return None


def record_shadow_signal(row: dict):
    if not init_database():
        return False
    item = ShadowSignal(
        recorded_at=datetime.now(timezone.utc),
        event_time=str(row.get("timestamp") or row.get("event_time") or "") or None,
        symbol=str(row.get("symbol") or "UNKNOWN").upper(),
        side=row.get("side"),
        probability=_to_float(row.get("probability")),
        expected_r=_to_float(row.get("expected_r")),
        regime=_to_int(row.get("regime")),
        threshold=_to_float(row.get("threshold")),
        qualified=row.get("qualified") if isinstance(row.get("qualified"), bool) else None,
        mode=row.get("mode"),
        model_version=row.get("model_version"),
        payload=row,
    )
    with _Session() as s:
        s.add(item)
        s.commit()
    return True


def recent_shadow_signals(limit=100):
    if not init_database():
        return []
    limit=max(1,min(int(limit),5000))
    with _Session() as s:
        rows=s.execute(select(ShadowSignal).order_by(ShadowSignal.id.desc()).limit(limit)).scalars().all()
    return [dict(x.payload) for x in reversed(rows)]


def shadow_signal_count():
    if not init_database():
        return 0
    with _Session() as s:
        return int(s.execute(select(func.count(ShadowSignal.id))).scalar_one())


def save_training_state(symbol: str, state: dict):
    if not init_database():
        return False
    item=TrainingState(
        recorded_at=datetime.now(timezone.utc),
        symbol=symbol.upper(),
        version=state.get("version"),
        status=str(state.get("status") or "UNKNOWN"),
        started_at=state.get("started_at"),
        finished_at=state.get("finished_at"),
        dataset_rows=int(state.get("dataset_rows") or 0),
        error=state.get("error"),
        metrics=state.get("metrics"),
    )
    with _Session() as s:
        s.add(item)
        s.commit()
    return True


def latest_training_state(symbol: str):
    if not init_database():
        return None
    with _Session() as s:
        row=s.execute(
            select(TrainingState)
            .where(TrainingState.symbol == symbol.upper())
            .order_by(TrainingState.id.desc())
            .limit(1)
        ).scalar_one_or_none()
    if row is None:
        return None
    return {
        "status":row.status,
        "version":row.version,
        "started_at":row.started_at,
        "finished_at":row.finished_at,
        "error":row.error,
        "dataset_rows":row.dataset_rows,
        "metrics":row.metrics,
    }
