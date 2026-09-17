from pathlib import Path
import joblib

class LiveModelLoader:
    def __init__(self,root="artifacts/models"): self.root=Path(root); self.cache={}
    def load(self,symbol,version="v0.5"):
        key=(symbol.upper(),version)
        if key not in self.cache:
            path=self.root/key[0]/version/"model.joblib"
            if not path.exists(): return None
            self.cache[key]=joblib.load(path)
        return self.cache[key]
