import json
from pathlib import Path
import pandas as pd

from src.utils.casting import safe_int, safe_float
from src.tracking.possession import identify_possession

def build_pbp_index(pbp: pd.DataFrame) -> dict[tuple[int, int], pd.Series]:
    """
    Build a fast lookup: (GAME_ID, EVENTNUM) -> row
    Assumes pbp has GAME_ID and EVENTNUM columns.
    """
    pbp = pbp.copy()
    pbp["GAME_ID"] = pbp["GAME_ID"].astype(int)
    pbp["EVENTNUM"] = pbp["EVENTNUM"].astype(int)

    # keep last occurrence if duplicates exist
    return {(int(r.GAME_ID), int(r.EVENTNUM)): r for _, r in pbp.iterrows()}

def sportvu_game_to_processed_events(game: dict, pbp: pd.DataFrame) -> list[dict]:
    """
    Convert SportVU moments into per-event frame objects, joined with PBP possession_team_id.
    Returns a list of event objects (your events_out).
    """
    game_id = int(game["gameid"])
    pbp_idx = build_pbp_index(pbp)

    frame_counter = 0
    events_out: list[dict] = []

    for event in game.get("events", []):
        moments = event.get("moments")
        if not moments:
            continue

        event_id = int(event.get("eventId"))
        row = pbp_idx.get((game_id, event_id))
        if row is None:
            continue

        possession_team_id = safe_int(identify_possession(row))
        quarter = int(moments[0][0])

        event_obj = {
            "gameid": game_id,
            "event_id": event_id,
            "possession_team_id": possession_team_id,
            "quarter": quarter,
            "frames": []
        }

        for moment in moments:
            if moment is None or len(moment) < 6:
                continue

            frame_counter += 1

            frame = {
                "frame_id": frame_counter,
                "game_clock": safe_float(moment[2]),
                "shot_clock": safe_float(moment[3]),
                "ball": {
                    "x": safe_float(moment[5][0][2]),
                    "y": safe_float(moment[5][0][3]),
                    "z": safe_float(moment[5][0][4]),
                },
                "players": [
                    {
                        "teamid": safe_int(p[0]),
                        "playerid": safe_int(p[1]),
                        "x": safe_float(p[2]),
                        "y": safe_float(p[3]),
                        "z": safe_float(p[4]),
                    }
                    for p in moment[5][1:]
                ]
            }

            event_obj["frames"].append(frame)

        if event_obj["frames"]:
            events_out.append(event_obj)

    return events_out

