import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"

for d in (DATA_DIR, CKPT_DIR, RESULTS_DIR, METRICS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def save_metrics(name, metrics):
    with open(METRICS_DIR / f"{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(name):
    path = METRICS_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
