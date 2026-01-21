import pandas as pd
from pathlib import Path

from src.processing.sportvu_to_events import sportvu_game_to_processed_events
from src.data_io.save_load import load_json, save_json

def main():
    file_name = "0021500622"

    # ✅ f-strings so {file_name} actually interpolates
    GAME_JSON = Path(f"data/raw/json/{file_name}.json")
    PBP_CSV = Path("data/raw/2015-16_pbp.csv")
    OUTPUT_JSON = Path(f"data/processed/{file_name}_processed.json")

    game = load_json(GAME_JSON)
    pbp = pd.read_csv(PBP_CSV)

    events_out = sportvu_game_to_processed_events(game, pbp)
    save_json(events_out, OUTPUT_JSON)

    # optional: compute frame count for the print
    frame_count = sum(len(e["frames"]) for e in events_out)
    print(f"✅ Saved {len(events_out)} events with {frame_count} frames to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
