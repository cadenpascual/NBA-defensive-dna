from __future__ import annotations
import pandas as pd

# EVENTMSGTYPE (broad category)
MSG_SHOT_MADE = 1
MSG_SHOT_MISSED = 2
MSG_FREE_THROW = 3
MSG_REBOUND = 4
MSG_TURNOVER = 5
MSG_FOUL = 6
MSG_VIOLATION = 7
MSG_SUBSTITUTION = 8
MSG_TIMEOUT = 9
MSG_JUMP_BALL = 10
MSG_EJECTION = 11
MSG_PERIOD_START = 12
MSG_PERIOD_END = 13
MSG_ADMIN = 18  # generally dropped as non-play/admin marker :contentReference[oaicite:3]{index=3}

def coarse_event_type(eventmsgtype: int | float | None) -> str | None:
    """
    Coarse semantic bucket used BEFORE clustering.
    """
    if eventmsgtype is None or pd.isna(eventmsgtype):
        return None
    msg = int(eventmsgtype)

    if msg in (MSG_SHOT_MADE, MSG_SHOT_MISSED):
        return "shot"
    if msg == MSG_FREE_THROW:
        return "free_throw"
    if msg == MSG_REBOUND:
        return "rebound"
    if msg == MSG_TURNOVER:
        return "turnover"
    if msg == MSG_FOUL:
        return "foul"
    if msg == MSG_VIOLATION:
        return "violation"
    if msg in (MSG_SUBSTITUTION, MSG_TIMEOUT, MSG_EJECTION):
        return "dead_ball"
    if msg in (MSG_PERIOD_START, MSG_PERIOD_END):
        return "period_boundary"
    if msg == MSG_JUMP_BALL:
        return "jump_ball"
    if msg == MSG_ADMIN:
        return "admin"
    return "other"