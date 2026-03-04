from pathlib import Path
import numpy as np
import pandas as pd

def extract_events(lap, signal_col):
    # Gets a list of distances where "events" happen.
    # i.e.: upwards threshold crossing of signal_col (spaced by min separation)
    if signal_col == "brake":
        threshold = 0.1
    elif signal_col == "throttle":
        threshold = 0.1
    else:
        threshold = 0.5
    min_sep = 40.0

    df = lap.sort_values("distance")
    distance = df["distance"].to_numpy(dtype=float)
    signal = df[signal_col].to_numpy(dtype=float)

    if len(signal) < 2:
        return []

    # Find specifically where signal crosses the threshold:
    above = signal >= threshold
    prev_above = np.roll(above, 1)
    prev_above[0] = False
    
    idx = np.where(above & ~prev_above)[0]

    events = []
    last_dist = -1e18
    for i in idx:
        dist = float(distance[i])
        if dist - last_dist > min_sep:
            events.append(float(dist))
            last_dist = dist
    return events

def nearest_event(lap: float, refs: list[float]):
    tolerance = float(80.0)
    if not refs:
        return None
    array = np.asarray(refs)
    i = int(np.argmin(np.abs(array - float(lap))))
    best = float(array[i])
    return best if abs(best - float(lap)) <= tolerance else None

def build_references(scored_laps: pd.DataFrame, track: str, mode: str):
    top_percent = float(0.2)

    if mode not in ["brake", "throttle"]:
        raise ValueError(f"{mode} not 'brake' or 'throttle'.")

    if mode == "brake":
        prob_col = "should_brake" if "should_brake" in scored_laps.columns else "p_brake_zone"
    else:
        prob_col = "should_throttle" if "should_throttle" in scored_laps.columns else "p_throttle_zone"

    df = scored_laps.loc[scored_laps["track"].astype(str) == str(track)].copy()
    lap_times = (df.groupby("lap_id")["laptime"].first().sort_values())

    n_keep = max(1, int(round(len(lap_times) * float(top_percent))))
    keep_ids = set(lap_times.index[:n_keep])

    event_list: list[list[float]] = []
    df_filter = df.loc[df["lap_id"].isin(keep_ids)]
    for _, lap_df in df_filter.groupby("lap_id", sort=False):
        event_list.append(extract_events(lap_df, prob_col))
    
    max_len = max((len(ev) for ev in event_list), default=0)
    true_events: list[float] = []

    for i in range(max_len):
        values = [event[i] for event in event_list if len(event) > i]
        if values:
            true_events.append(float(np.median(values)))

    return true_events

def advice(lap: pd.DataFrame, ref_brake: list[float], ref_throttle: list[float]):
    df = lap.sort_values("distance")
    brake = extract_events(df, "brake")
    throttle = extract_events(df, "throttle")

    rows: list[dict] = []
    def add(mode: str, events: list[float], refs: list[float]):
        for i, ref_d in enumerate(refs):
            lap_d = nearest_event(ref_d, events)
            if lap_d is None:
                continue
            
            delta = float(lap_d) - float(ref_d)
            
            if mode == "brake":
                if delta < 0:
                    advice_text = f"Brake {abs(delta):.1f}m earlier"
                else:
                    advice_text = f"Brake {abs(delta):.1f}m later"
            elif mode == "throttle":
                if delta < 0:
                    advice_text = f"Throttle {abs(delta):.1f}m earlier"
                else:
                    advice_text = f"Throttle {abs(delta):.1f}m later"

            rows.append({
                "mode": mode,
                "zone_index": i+1,
                "lap_distance": round(float(lap_d), 1),
                "ref_distance": round(float(ref_d), 1),
                "delta": round(delta, 1),
                "advice": advice_text
            })

    add("brake", brake, ref_brake)
    add("throttle", throttle, ref_throttle)

    out_df = pd.DataFrame(rows, columns=[
        "mode", "zone_index", "lap_distance", "ref_distance", "delta", "advice"
    ])
    
    return out_df.sort_values(["mode", "zone_index"]).reset_index(drop=True)

def write_advice(advice_df: pd.DataFrame, out_path: Path, track_name: str, lap_id: str) -> Path:
    lines = [f"Track: {track_name}", f"Lap: {lap_id}", ""]
    if advice_df.empty:
        lines.append("No advice events found.")
    else:
        for _, r in advice_df.iterrows():
            lines.append(
                f"{r['mode']} zone {int(r['zone_index'])}: {r['advice']} "
                f"(lap={float(r['lap_distance']):.1f}m, ref={float(r['ref_distance']):.1f}m, delta={float(r['delta']):+.1f}m)"
            )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path