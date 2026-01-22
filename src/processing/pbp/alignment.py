# src/processing/pbp_alignment.py
from __future__ import annotations
import pandas as pd

from src.processing.indexing import find_event_for_shot_by_clock

def align_pbp_to_tracking_by_clock(
    pbp_g: pd.DataFrame,
    tracking_time_index: pd.DataFrame,
    *,
    span_pad: float = 2.0,
    max_center_diff: float = 10.0,
    keep_debug: bool = False,
) -> pd.DataFrame:
    """
    Align PBP rows to tracking events by (GAME_ID, PERIOD, game_clock).

    Expects pbp_g to have columns: GAME_ID, PERIOD, game_clock.
    Returns a copy of pbp_g with an added column 'event_list_idx'.
    If keep_debug=True, also adds 'align_reason', 'align_center_diff', etc.
    """
    out = pbp_g.copy()
    out["event_list_idx"] = pd.NA

    if keep_debug:
        out["align_reason"] = pd.NA
        out["align_center_diff"] = pd.NA

    for i, row in out.iterrows():
        if pd.isna(row.get("game_clock")) or pd.isna(row.get("PERIOD")) or pd.isna(row.get("GAME_ID")):
            continue

        ev_idx, info = find_event_for_shot_by_clock(
            tracking_time_index,
            gameid=int(row["GAME_ID"]),
            quarter=int(row["PERIOD"]),
            shot_game_clock=float(row["game_clock"]),
            span_pad=span_pad,
            max_center_diff=max_center_diff,
        )

        if ev_idx is None:
            if keep_debug:
                out.at[i, "align_reason"] = info.get("reason")
            continue

        out.at[i, "event_list_idx"] = int(ev_idx)

        if keep_debug:
            out.at[i, "align_reason"] = info.get("reason")
            out.at[i, "align_center_diff"] = info.get("center_diff")

    return out
