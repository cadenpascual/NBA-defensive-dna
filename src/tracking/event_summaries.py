# src/tracking/event_summaries.py
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple

def event_clock_span(event: dict) -> Tuple[Optional[float], Optional[float]]:
    frames = event.get("frames", [])
    if not frames:
        return None, None
    gcs = [fr.get("game_clock") for fr in frames if fr.get("game_clock") is not None]
    if not gcs:
        return None, None
    return float(np.max(gcs)), float(np.min(gcs))

def first_ball_xy(event: dict) -> Tuple[Optional[float], Optional[float]]:
    frames = event.get("frames", [])
    if not frames:
        return None, None
    ball = (frames[0].get("ball") or {})
    x = ball.get("x")
    y = ball.get("y")
    return (float(x) if x is not None else None,
            float(y) if y is not None else None)
