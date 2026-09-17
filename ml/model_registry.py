from pathlib import Path
import json,joblib

class ModelRegistry:
    """Keep XAUUSD and BTC artifacts isolated by symbol and version."""
    def __init__(self,root="artifacts/models"): self.root=Path(root)
    def path(self,symbol,version): return self.root/symbol.upper()/version
    def save(self,symbol,version,bundle,metadata):
        p=self.path(symbol,version); p.mkdir(parents=True,exist_ok=True); joblib.dump(bundle,p/"model.joblib"); (p/"metadata.json").write_text(json.dumps(metadata,indent=2)); return p
    def load(self,symbol,version): return joblib.load(self.path(symbol,version)/"model.joblib")
    def metadata(self,symbol,version): return json.loads((self.path(symbol,version)/"metadata.json").read_text())
