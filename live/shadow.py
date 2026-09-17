from datetime import datetime,timezone

class ShadowJournal:
    """Research-only journal. It never submits broker orders."""
    def __init__(self,max_rows=5000): self.rows=[]; self.max_rows=max_rows
    def record(self,event):
        row={"recorded_at":datetime.now(timezone.utc).isoformat(),**event}; self.rows.append(row); self.rows=self.rows[-self.max_rows:]; return row
    def recent(self,limit=100): return self.rows[-max(1,min(limit,self.max_rows)):]
