from datetime import datetime,timezone

from app.database import record_shadow_signal, recent_shadow_signals, shadow_signal_count


class ShadowJournal:
    """Research-only journal. It never submits broker orders."""

    def __init__(self,max_rows=5000):
        self.rows=[]
        self.max_rows=max_rows

    def record(self,event):
        row={"recorded_at":datetime.now(timezone.utc).isoformat(),**event}
        self.rows.append(row)
        self.rows=self.rows[-self.max_rows:]
        try:
            record_shadow_signal(row)
        except Exception:
            pass
        return row

    def recent(self,limit=100):
        limit=max(1,min(int(limit),self.max_rows))
        try:
            persisted=recent_shadow_signals(limit)
            if persisted:
                return persisted
        except Exception:
            pass
        return self.rows[-limit:]

    def count(self):
        try:
            persisted=shadow_signal_count()
            if persisted:
                return persisted
        except Exception:
            pass
        return len(self.rows)
