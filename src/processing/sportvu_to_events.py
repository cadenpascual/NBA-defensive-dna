import json
from pathlib import Path
import pandas as pd


from src.utils.casting import safe_int, safe_float
from src.tracking.possession import identify_possession
from src.processing.pbp.context import pbp_context
from src.processing.pbp.indexing import build_pbp_index
from src.tracking.event_summaries import event_clock_span, first_ball_xy

def sportvu_game_to_processed_events(game: dict, pbp: pd.DataFrame) -> list[dict]:
    game_id = int(game["gameid"])
    pbp_idx = build_pbp_index(pbp)

    frame_counter = 0
    events_out: list[dict] = []

    for event in game.get("events", []):
        moments = event.get("moments")
        if not moments:
            continue

        event_id = int(event.get("eventId"))

        # lookup pbp row (fast)
        try:
            row = pbp_idx.loc[(game_id, event_id)]
        except KeyError:
            continue

        possession_team_id = safe_int(identify_possession(row))
        quarter = int(moments[0][0])

        event_obj = {
            "gameid": game_id,
            "event_id": event_id,
            "quarter": quarter,
            "possession_team_id": possession_team_id,

            # ✅ add pbp metadata for play-type classification
            **pbp_context(row),

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
                ],
            }

            event_obj["frames"].append(frame)

        if not event_obj["frames"]:
            continue

        # ✅ event-level summaries for indexing/matching without rescanning frames
        gc_start, gc_end = event_clock_span(event_obj)
        bx, by = first_ball_xy(event_obj)
        event_obj["gc_start"] = gc_start
        event_obj["gc_end"] = gc_end
        event_obj["ball_x0"] = bx
        event_obj["ball_y0"] = by

        events_out.append(event_obj)

    return events_out


# raw SportVU JSON to processed tracking events
def raw_sportvu_to_tracking_events(game: dict) -> list[dict]:
    """
    Convert raw SportVU 'events' with 'moments' into your standard tracking_events format.
    No PBP join, no possession assignment — just frames with clocks + positions.
    """
    tracking_events = []
    frame_counter = 0
    gameid = int(game["gameid"])

    for ev in game.get("events", []):
        moments = ev.get("moments")
        if not moments:
            continue

        # quarter is in moment[0] (based on your raw format)
        quarter = int(moments[0][0])

        event_obj = {
            "gameid": gameid,
            "event_id_raw": ev.get("eventId"),  # keep raw id if you want
            "quarter": quarter,
            "frames": []
        }

        for moment in moments:
            if moment is None or len(moment) < 6:
                continue

            frame_counter += 1

            ball_row = moment[5][0]   # [-1, -1, x, y, z]
            players_rows = moment[5][1:]  # [teamid, playerid, x, y, z] x10

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
                ],
            }

            event_obj["frames"].append(frame)

        if event_obj["frames"]:
            tracking_events.append(event_obj)

    return tracking_events
