import numpy as np
import pandas as pd
from src.features.defense_single import compute_pre_shot_defense_features
from src.tracking.release import find_release_frame_idx
from src.processing.indexing import find_event_for_shot_by_clock

def compute_defense_features_for_shots(
    shots_g,
    tracking_events,
    event_index,
    *,
    fps=25,
    window_seconds=1.0,
    smooth_window=5,
    span_pad=4.0,
    max_center_diff=20.0,
    max_time_diff=1.5
):
    """
    Compute defense features for all shots in shots_g.
    Returns a DataFrame indexed like shots_g, containing ONLY valid shots.
    """

    rows = []

    for idx, shot in shots_g.iterrows():

        try:
            # --- shot time ---
            shot_gc = float(shot["game_clock"])
            quarter = int(shot["PERIOD"])

            # normalize GAME_ID to int if needed
            gameid = int(shot["GAME_ID"])

            # --- find tracking event ---
            ev_idx, _ = find_event_for_shot_by_clock(
                event_index,
                gameid,
                quarter,
                shot_gc,
                span_pad=span_pad,
                max_center_diff=max_center_diff
            )

            if ev_idx is None:
                continue

            event = tracking_events[int(ev_idx)]
            frames = event["frames"]

            # --- find release frame ---
            release_idx, _ = find_release_frame_idx(
                event_frames=frames,
                shot_game_clock=shot_gc,
                match="prev",
                max_time_diff=max_time_diff
            )

            if release_idx is None:
                continue

            # --- compute defense features ---
            feats = compute_pre_shot_defense_features(
                event_frames=frames,
                release_frame_idx=release_idx,
                shooter_id=int(shot["PLAYER_ID"]),
                offense_team_id=int(shot["TEAM_ID"]),
                fps=fps,
                window_seconds=window_seconds,
                smooth_window=smooth_window
            )

            if "error" in feats:
                continue

            # --- store row ---
            feats["shot_index"] = idx
            feats["release_idx"] = release_idx
            rows.append(feats)

        except Exception:
            # hard fail-safe: skip shot silently
            continue

    if len(rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("shot_index")
    return df
