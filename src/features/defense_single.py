import numpy as np
import pandas as pd

# find defensive features from tracking data
def compute_pre_shot_defense_features(
    event_frames,
    release_frame_idx,
    shooter_id,
    offense_team_id,
    fps=25,
    window_seconds=1.0,
    smooth_window=5,
):
    """
    Compute pre-shot shooter + defender features from tracking frames.

    Parameters
    ----------
    event_frames : list[dict]
        List of frames like event["frames"], each frame has:
          - "players": list of {teamid, playerid, x, y, z}
          - "game_clock" (optional)
          - "shot_clock" (optional)
    release_frame_idx : int
        Index in event_frames corresponding to shot release (or closest match).
    shooter_id : int
        Shooter player id.
    offense_team_id : int
        Team id of offense (shooter team). Defenders are players with teamid != offense_team_id.
    fps : int
        Frames per second (25 for SportVU).
    window_seconds : float
        Seconds BEFORE release to use. 1.0s => 25 frames.
    smooth_window : int
        Rolling window length (frames) for smoothing speed/accel. Use odd number like 5.

    Returns
    -------
    dict
        Features including closest defender distance, closing speed, speed/accel stats, etc.
        Returns np.nan for features if data is insufficient.
    """
    dt = 1.0 / fps
    n_back = int(round(window_seconds * fps))

    # Window indices: strictly before (and including) release frame
    start = max(0, release_frame_idx - n_back)
    end = release_frame_idx
    idxs = list(range(start, end + 1))

    # Helper: extract xy for a specific player in a frame
    def get_player_xy(frame, pid):
        for p in frame.get("players", []):
            if p.get("playerid") == pid:
                return np.array([p.get("x", np.nan), p.get("y", np.nan)], dtype=float)
        return None

    # Helper: get defenders list (xy, id) in a frame
    def get_defenders_xy(frame):
        defs = []
        for p in frame.get("players", []):
            if p.get("teamid") != offense_team_id:
                defs.append((np.array([p.get("x", np.nan), p.get("y", np.nan)], dtype=float), p.get("playerid")))
        return defs

    # Collect shooter trajectory + defender trajectories (for chosen closest defender)
    shooter_xy = []
    gc = []
    sc = []

    # We'll decide closest defender at release, then track that same defender over the window
    frame_release = event_frames[release_frame_idx]
    shooter_xy_release = get_player_xy(frame_release, shooter_id)
    if shooter_xy_release is None:
        return {"error": "shooter_not_found", "release_frame_idx": release_frame_idx}

    defenders_release = get_defenders_xy(frame_release)
    if len(defenders_release) == 0:
        return {"error": "no_defenders_found", "release_frame_idx": release_frame_idx}

    # Identify closest defender at release
    dists_release = [(float(np.linalg.norm(shooter_xy_release - dxy)), did) for dxy, did in defenders_release]
    close_def_dist_release, close_def_id = min(dists_release, key=lambda t: t[0])

    # Gather trajectories over the window for shooter + that defender
    def_xy = []

    for k in idxs:
        fr = event_frames[k]
        sx = get_player_xy(fr, shooter_id)
        dx = get_player_xy(fr, close_def_id)

        # If missing, append nan to keep alignment
        shooter_xy.append(sx if sx is not None else np.array([np.nan, np.nan], dtype=float))
        def_xy.append(dx if dx is not None else np.array([np.nan, np.nan], dtype=float))

        gc.append(fr.get("game_clock", np.nan))
        sc.append(fr.get("shot_clock", np.nan))

    shooter_xy = np.vstack(shooter_xy)  # (T,2)
    def_xy = np.vstack(def_xy)          # (T,2)
    T = shooter_xy.shape[0]

    # If too few frames, bail
    if T < 5:
        return {"error": "too_few_frames", "T": T, "release_frame_idx": release_frame_idx}

    # Distance time series between shooter and closest defender
    dist = np.linalg.norm(shooter_xy - def_xy, axis=1)

    # Central difference velocity (vector) for shooter/defender
    def central_vel(pos):
        v = np.full_like(pos, np.nan, dtype=float)
        v[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
        return v

    v_sh = central_vel(shooter_xy)
    v_df = central_vel(def_xy)

    speed_sh = np.linalg.norm(v_sh, axis=1)
    speed_df = np.linalg.norm(v_df, axis=1)

    # Smooth speeds to stabilize acceleration
    def smooth_1d(x, w):
        if w is None or w <= 1:
            return x
        # simple centered rolling mean
        return pd.Series(x).rolling(window=w, center=True, min_periods=max(2, w//2)).mean().to_numpy()

    speed_sh_s = smooth_1d(speed_sh, smooth_window)
    speed_df_s = smooth_1d(speed_df, smooth_window)

    # Central difference acceleration magnitude from smoothed speed
    accel_sh = np.full(T, np.nan, dtype=float)
    accel_df = np.full(T, np.nan, dtype=float)
    accel_sh[1:-1] = (speed_sh_s[2:] - speed_sh_s[:-2]) / (2.0 * dt)
    accel_df[1:-1] = (speed_df_s[2:] - speed_df_s[:-2]) / (2.0 * dt)

    # Closing speed: derivative of distance (negative means closing in)
    dist_s = smooth_1d(dist, smooth_window)
    closing = np.full(T, np.nan, dtype=float)
    closing[1:-1] = (dist_s[2:] - dist_s[:-2]) / (2.0 * dt)

    # Summary features
    def nan_stats(x):
        return {
            "mean": float(np.nanmean(x)),
            "max": float(np.nanmax(x)),
            "min": float(np.nanmin(x)),
            "std": float(np.nanstd(x)),
        }

    feats = {
        "close_def_id": int(close_def_id),
        "close_def_dist_release": float(close_def_dist_release),
        "close_def_dist_min": float(np.nanmin(dist)),
        "close_def_dist_mean": float(np.nanmean(dist)),

        # closing speed: negative = closing in. we'll provide min (most negative) and mean
        "close_def_closing_speed_mean": float(np.nanmean(closing)),
        "close_def_closing_speed_min": float(np.nanmin(closing)),

        # defender speed/accel stats
        "def_speed_mean": nan_stats(speed_df_s)["mean"],
        "def_speed_max": nan_stats(speed_df_s)["max"],
        "def_accel_mean": nan_stats(np.abs(accel_df))["mean"],
        "def_accel_max": nan_stats(np.abs(accel_df))["max"],

        # shooter speed/accel stats
        "shooter_speed_mean": nan_stats(speed_sh_s)["mean"],
        "shooter_speed_max": nan_stats(speed_sh_s)["max"],
        "shooter_accel_mean": nan_stats(np.abs(accel_sh))["mean"],
        "shooter_accel_max": nan_stats(np.abs(accel_sh))["max"],

        # time info (optional)
        "window_frames": T,
        "game_clock_release": float(gc[-1]) if len(gc) else np.nan,
        "shot_clock_release": float(sc[-1]) if len(sc) else np.nan,
    }

    return feats
