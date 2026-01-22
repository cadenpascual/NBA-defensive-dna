from __future__ import annotations
from src.tracking.event_summaries import first_ball_xy

# Court constants (feet)
BASELINE_Y = 0.0
BASKET_X = 0.0

def classify_play_start(event: dict, restart_trigger: str | None) -> str:
    """
    Assigns one of:
      - missed_free_throw
      - turnover_start
      - baseline_inbound
      - normal_play
    """

    x0, y0 = first_ball_xy(event)

    # Safety fallback
    if x0 is None or y0 is None:
        return "normal_play"

    # --- Rule-based logic ---

    # 1) Missed FT → live rebound or inbound
    if restart_trigger == "missed_free_throw":
        return "missed_free_throw"

    # 2) Turnover restart
    if restart_trigger == "turnover":
        return "turnover_start"

    # 3) Baseline inbound (under basket)
    # Ball starts very close to baseline and near basket horizontally
    if abs(y0 - BASELINE_Y) < 3.0 and abs(x0 - BASKET_X) < 8.0:
        return "baseline_inbound"

    # 4) Everything else
    return "normal_play"
