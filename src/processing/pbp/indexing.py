import pandas as pd

PBP_KEEP_COLS = [
    "GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE",
    "PLAYER1_TEAM_ID", "PLAYER2_TEAM_ID",
    "HOMEDESCRIPTION", "VISITORDESCRIPTION",
    "PCTIMESTRING", "PERIOD"
]

def build_pbp_index(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Return pbp indexed by (GAME_ID, EVENTNUM), keeping only relevant columns.
    Deduplicates by keeping the last row for each (GAME_ID, EVENTNUM).
    """
    df = pbp.copy()

    # keep only what we need (ignore missing columns gracefully)
    keep = [c for c in PBP_KEEP_COLS if c in df.columns]
    df = df[keep]

    df["GAME_ID"] = df["GAME_ID"].astype(int)
    df["EVENTNUM"] = df["EVENTNUM"].astype(int)

    # if duplicates exist, keep last
    df = df.drop_duplicates(subset=["GAME_ID", "EVENTNUM"], keep="last")

    return df
