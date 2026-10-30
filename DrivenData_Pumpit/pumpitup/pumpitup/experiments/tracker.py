import json
import time
import math
from pathlib import Path


class ExperimentTracker:
    """
    Simple JSON-based experiment tracker.
    """

    def __init__(self, path=None):
        if path is None:
            # FORCE the correct absolute path
            self.path = Path(__file__).resolve().parent / "experiments.json"
        else:
            self.path = Path(path)

        if not self.path.exists():
            self.path.write_text("[]")
        print("DEBUG __file__:", __file__)
        print("DEBUG resolved path:", Path(__file__).resolve())
        print("DEBUG parent:", Path(__file__).resolve().parent)
        print("DEBUG final experiments.json:", self.path.resolve())


    def log(self, name, params, cv_score, notes=""):
        data = json.loads(self.path.read_text())

        if cv_score is None or (isinstance(cv_score, float) and math.isnan(cv_score)):
            cv_score_json = None
        else:
            cv_score_json = float(cv_score)

        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "cv_score": cv_score_json,
            "params": params,
            "notes": notes,
        }

        data.append(entry)
        self.path.write_text(json.dumps(data, indent=2))

    def list(self):
        data = json.loads(self.path.read_text())

        for exp in data:
            if isinstance(exp["cv_score"], str):
                try:
                    exp["cv_score"] = float(exp["cv_score"])
                except:
                    exp["cv_score"] = None

        return data

    def best(self):
        data = self.list()
        if not data:
            print("No experiments logged yet.")
            return None

        valid = [exp for exp in data if exp["cv_score"] is not None]

        if not valid:
            print("No experiments with valid CV scores.")
            return None

        return max(valid, key=lambda x: x["cv_score"])

    def filter(self, name):
        data = self.list()
        return [exp for exp in data if exp["name"] == name]
