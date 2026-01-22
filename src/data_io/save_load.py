import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def load_tracking_json(game_id):
    path = DATA_DIR / "tracking" / f"{game_id}.json"
    with open(path, "r") as f:
        return json.load(f)

def save_dataframe(df, name):
    path = DATA_DIR / "processed" / f"{name}.parquet"
    df.to_parquet(path)

def load_dataframe(name):
    path = DATA_DIR / "processed" / f"{name}.parquet"
    return pd.read_parquet(path)

def load_json(path):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, obj, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)