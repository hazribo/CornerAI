from pathlib import Path
import numpy as np
import pandas as pd

def extract_events(lap, signal_col):
    # Gets a list of distances where "events" happen.
    # i.e.: upwards threshold crossing of signal_col (spaced by min separation)
    threshold = float(0.6)
    min_sep = 40.0

    df = lap.sort_values("distance")
    distance = df["distance"].to_numpy(dtype=float)
    signal = df[signal_col].to_numpy(dtype=float)

    if len(signal) < 2:
        return []

    # detect binary vs probability series:
    uniq = np.unique(signal[~np.isnan(signal)])
    is_binary = np.all(np.isin(uniq, [0.0, 1.0]))

    if is_binary:
        idx = np.where((signal[1:] == 1.0) & (signal[:-1] == 0.0))[0] + 1
    else:
        idx = np.where((signal[1:] >= threshold) & (signal[:-1] < threshold))[0] + 1

    events = []
    last = -1e18
    for i in idx:
        d = float(distance[i])
        if d - last >= min_sep:
            events.append(d)
            last = d
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

    # get probability for correct attribute (brake or throttle)
    prob_col = "should_brake" if (mode == "brake" and "should_brake" in scored_laps.columns) else (
        "p_brake_zone" if mode == "brake" else "p_throttle_zone"
    )

    df = scored_laps.loc[scored_laps["track"].astype(str) == str(track)].copy()
    lap_times = (df.groupby("lap_id")["laptime"].first().sort_values())

    n_keep = max(1, int(round(len(lap_times) * float(top_percent))))
    keep_ids = set(lap_times.index[:n_keep])

    # get event lists (one per lap):
    event_list: list[list[float]] = []
    df_filter = df.loc[df["lap_id"].isin(keep_ids)]
    for _, lap_df in df_filter.groupby("lap_id", sort=False):
        event_list.append(extract_events(lap_df, prob_col))
    
    # get median by event index:
    # (this is to stop "pedantic" improvements like "brake 1 metre earlier")
    max_len = max((len(ev) for ev in event_list), default=0)
    true_events: list[float] = []

    for i in range(max_len):
        values = [event[i] for event in event_list if len(event) > i]
        if values:
            true_events.append(float(np.median(values)))

    return true_events

def advice(lap: pd.DataFrame, ref_brake: list[float], ref_throttle: list[float]):
    df = lap.sort_values("distance")
    brake_col = "should_brake" if "should_brake" in df.columns else "p_brake_zone"
    brake = extract_events(df, brake_col)
    if len(brake) == 0 and "p_brake_zone" in df.columns:
        brake = extract_events(df, "p_brake_zone")
    throttle = extract_events(df, "p_throttle_zone")
    rows: list[dict] = []

    def add(mode: str, events: list[float], refs: list[float]):
        remaining = list(events)

        for i, ref_d in enumerate(refs):
            lap_d = nearest_event(ref_d, remaining)
            if lap_d is None:
                continue
            remaining.remove(lap_d)

            delta = float(lap_d - ref_d)
            # "placeholder" advice for now:
            if mode == "brake":
                advice = f"Brake {delta:.1f}m earlier" if delta > 0 else f"Brake {delta*-1:.1f}m later"
            else:
                advice = f"Throttle {delta:.1f}m earlier" if delta > 0 else f"Throttle {delta*-1:.1f}m later"

            rows.append(
                {
                    "mode": mode,
                    "zone_index": i+1, # start counting from 1
                    "lap_distance": lap_d,
                    "ref_distance": ref_d,
                    "delta": delta,
                    "advice": advice
                }
            )
    add("brake", brake, ref_brake)
    add("throttle", throttle, ref_throttle)
    return pd.DataFrame(rows).sort_values(["mode", "zone_index"]).reset_index(drop=True)

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