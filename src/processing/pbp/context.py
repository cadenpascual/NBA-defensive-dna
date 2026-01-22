import pandas as pd

def pbp_context(row: pd.Series) -> dict:
    """
    Minimal context for play-type labeling.
    """
    msg = int(row["EVENTMSGTYPE"]) if pd.notna(row.get("EVENTMSGTYPE")) else None
    action = int(row["EVENTMSGACTIONTYPE"]) if pd.notna(row.get("EVENTMSGACTIONTYPE")) else None

    home = str(row.get("HOMEDESCRIPTION") or "")
    away = str(row.get("VISITORDESCRIPTION") or "")
    desc = home if home else away

    return {
        "pbp_msgtype": msg,
        "pbp_actiontype": action,
        "pbp_desc": desc,
        "pbp_team1": int(row["PLAYER1_TEAM_ID"]) if pd.notna(row.get("PLAYER1_TEAM_ID")) else None,
        "pbp_team2": int(row["PLAYER2_TEAM_ID"]) if pd.notna(row.get("PLAYER2_TEAM_ID")) else None,
    }