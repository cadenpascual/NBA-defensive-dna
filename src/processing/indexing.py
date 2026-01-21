# src/processing/indexing.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple

def build_tracking_event_index(tracking_events: list[dict]) -> pd.DataFrame:
    rows = []
    for k, ev in enumerate(tracking_events):
        frames = ev.get("frames", [])
        gcs = [fr.get("game_clock") for fr in frames if fr.get("game_clock") is not None]
        if not gcs:
            continue
        rows.append({
            "gameid": ev.get("gameid"),
            "quarter": ev.get("quarter"),
            "event_list_idx": k,
            "gc_start": float(np.max(gcs)),
            "gc_end": float(np.min(gcs)),
            "n_frames": len(frames),
        })
    return pd.DataFrame(rows)

def find_event_for_shot_by_clock(
    event_index: pd.DataFrame,
    gameid: int,
    quarter: int,
    shot_game_clock: float,
    span_pad: float = 1.0,
    max_center_diff: float = 2.0,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Returns (event_list_idx, debug_info) where event_list_idx is an index into tracking_events, or None.
    """
    df = event_index[
        (event_index["gameid"] == gameid) &
        (event_index["quarter"] == quarter)
    ]
    if df.empty:
        return None, {"reason": "no_events_for_game_quarter"}

    # Event span is [gc_end, gc_start] because clock counts down
    in_span = df[
        (shot_game_clock <= (df["gc_start"] + span_pad)) &
        (shot_game_clock >= (df["gc_end"]   - span_pad))
    ].copy()

    if in_span.empty:
        # fallback: choose closest boundary distance
        df2 = df.copy()
        df2["dist_to_span"] = np.minimum(
            np.abs(df2["gc_start"] - shot_game_clock),
            np.abs(df2["gc_end"] - shot_game_clock)
        )
        best = df2.loc[df2["dist_to_span"].idxmin()]
        return int(best["event_list_idx"]), {
            "reason": "fallback_closest_span",
            "dist_to_span": float(best["dist_to_span"]),
            "gc_start": float(best["gc_start"]),
            "gc_end": float(best["gc_end"]),
        }

    in_span["gc_center"] = (in_span["gc_start"] + in_span["gc_end"]) / 2.0
    in_span["center_diff"] = np.abs(in_span["gc_center"] - shot_game_clock)
    best = in_span.loc[in_span["center_diff"].idxmin()]

    if float(best["center_diff"]) > max_center_diff:
        return None, {
            "reason": "center_diff_too_large",
            "center_diff": float(best["center_diff"]),
            "gc_start": float(best["gc_start"]),
            "gc_end": float(best["gc_end"]),
        }

    return int(best["event_list_idx"]), {
        "reason": "ok",
        "center_diff": float(best["center_diff"]),
        "gc_start": float(best["gc_start"]),
        "gc_end": float(best["gc_end"]),
    }
