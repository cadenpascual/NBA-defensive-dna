from __future__ import annotations

from typing import Tuple
import pandas as pd

from src.processing.sportvu_to_events import raw_sportvu_to_tracking_events
from src.processing.tracking_cleaning import dedupe_tracking_events
from src.processing.indexing import build_tracking_time_index
from src.processing.pbp.restart_detection import detect_restart_triggers
from src.processing.pbp.alignment import align_pbp_to_tracking_by_clock
from src.processing.play_start_classifier import classify_play_start
from src.utils.casting import timestring_to_seconds


def build_labeled_tracking_events(
    game: dict,
    pbp: pd.DataFrame,
    *,
    span_pad: float = 2.0,
    max_center_diff: float = 10.0,
) -> Tuple[list[dict], pd.DataFrame]:
    """
    End-to-end:
      raw SportVU + raw PBP -> tracking_events with 'start_type' and aligned pbp table.
    """

    game_id = int(game["gameid"])

    # ---- PBP for this game (keep as DataFrame!) ----
    pbp_g = pbp.loc[pbp["GAME_ID"].astype(int) == game_id].copy()
    pbp_g = pbp_g.reset_index(drop=True)  # avoid EVENTNUM index/column ambiguity
    pbp_g["game_clock"] = pbp_g["PCTIMESTRING"].apply(timestring_to_seconds)

    # drop admin/junk if you want
    pbp_g = pbp_g[pbp_g["EVENTMSGTYPE"] != 18].copy()

    # ---- Tracking events from raw ----
    tracking_events = raw_sportvu_to_tracking_events(game)
    tracking_events = dedupe_tracking_events(tracking_events)

    tracking_time_index = build_tracking_time_index(tracking_events)

    # ---- Restart triggers + alignment ----
    pbp_g = detect_restart_triggers(pbp_g)

    pbp_aligned = align_pbp_to_tracking_by_clock(
        pbp_g,
        tracking_time_index,
        span_pad=span_pad,
        max_center_diff=max_center_diff,
        keep_debug=True,
    )

    # ---- Assign start_type onto tracking events ----
    # Choose ONE representative pbp row per event_list_idx: smallest align_center_diff
    pbp_ev = pbp_aligned.dropna(subset=["event_list_idx"]).copy()
    pbp_ev["event_list_idx"] = pbp_ev["event_list_idx"].astype(int)
    pbp_ev["align_center_diff"] = (
    pd.to_numeric(pbp_ev["align_center_diff"], errors="coerce")
      .fillna(1e9))


    rep = (
        pbp_ev.sort_values(["event_list_idx", "align_center_diff"])
              .groupby("event_list_idx", as_index=False)
              .head(1)
    )

    for _, row in rep.iterrows():
        ev_idx = int(row["event_list_idx"])
        st = classify_play_start(
            tracking_events[ev_idx],
            restart_trigger=row.get("restart_trigger"),
        )
        tracking_events[ev_idx]["start_type"] = st

    return tracking_events, pbp_aligned
