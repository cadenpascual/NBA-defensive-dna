from __future__ import annotations
import re
import pandas as pd

_FT_OF_RE = re.compile(r"(\d+)\s*OF\s*(\d+)", re.IGNORECASE)

def best_desc(row: pd.Series) -> str:
    """Pick the first non-empty description among home/visitor/neutral."""
    for col in ("HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"):
        v = row.get(col)
        if v is not None and not pd.isna(v) and str(v).strip():
            return str(v)
    return ""

def is_last_free_throw(desc: str) -> bool:
    """
    True if desc contains 'k of n' with k==n.
    Example: 'MISS Bogut Free Throw 2 of 2' -> True
    """
    m = _FT_OF_RE.search(desc.upper())
    if not m:
        return False
    k, n = int(m.group(1)), int(m.group(2))
    return k == n

def detect_restart_triggers(pbp_g: pd.DataFrame) -> pd.DataFrame:
    """
    Adds restart_trigger to the FIRST non-free-throw row after:
      - missed LAST free throw  -> 'missed_free_throw'
      - turnover               -> 'turnover'
      - made FG                -> 'made_basket' (useful for baseline inbound later)

    Requires columns: PERIOD, game_clock, EVENTNUM, EVENTMSGTYPE, descriptions.
    """
    pbp = pbp_g.copy()

    # stable order within same second: EVENTNUM ascending
    pbp = pbp.sort_values(["PERIOD", "game_clock", "EVENTNUM"], ascending=[True, False, True]).reset_index(drop=True)

    pbp["restart_trigger"] = None

    i = 0
    while i < len(pbp) - 1:
        row = pbp.iloc[i]
        msg = int(row["EVENTMSGTYPE"]) if not pd.isna(row["EVENTMSGTYPE"]) else None
        desc = best_desc(row).upper()

        # ---- Case A: Missed LAST free throw ----
        if msg == 3 and "MISS" in desc and is_last_free_throw(desc):
            # assign trigger to the first subsequent row that is NOT a free throw
            j = i + 1
            while j < len(pbp) and int(pbp.iloc[j]["EVENTMSGTYPE"]) == 3:
                j += 1
            if j < len(pbp):
                pbp.at[j, "restart_trigger"] = "missed_free_throw"
            i = j
            continue

        # ---- Case B: Turnover ----
        if msg == 5:
            pbp.at[i + 1, "restart_trigger"] = "turnover"
            i += 1
            continue

        # ---- Case C: Made field goal ----
        if msg == 1:
            pbp.at[i + 1, "restart_trigger"] = "made_basket"
            i += 1
            continue

        i += 1

    return pbp
