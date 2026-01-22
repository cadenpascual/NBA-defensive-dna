def event_signature(ev: dict):
    frames = ev.get("frames", [])
    gcs = [fr.get("game_clock") for fr in frames if fr.get("game_clock") is not None]
    if not gcs:
        return None
    return (ev.get("gameid"), ev.get("quarter"),
            round(max(gcs), 2), round(min(gcs), 2), len(frames))

def dedupe_tracking_events(tracking_events: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for ev in tracking_events:
        sig = event_signature(ev)
        if sig is None:
            continue
        if sig in seen:
            continue
        seen.add(sig)
        out.append(ev)
    return out
