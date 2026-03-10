from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def extract_events(lap, signal_col):
    # Gets a list of distances where "events" happen.
    # i.e.: upwards threshold crossing of signal_col (spaced by min separation)
    threshold = 0.6
    min_sep = 40.0

    df = lap.sort_values("cl_dist")
    distance = df["cl_dist"].to_numpy(dtype=float)
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

def build_references_from_gt(gt: pd.DataFrame, mode: str, min_prob: float = 0.4) -> list[float]:
    prob_col = "p_brake_exp" if mode == "brake" else "p_throttle_exp"
    if prob_col not in gt.columns:
        return []

    cl = gt["cl_dist"].to_numpy(dtype=float)
    prob = gt[prob_col].to_numpy(dtype=float)

    # min_distance: peaks must be at least 40m apart in cl_dist space
    bin_m = float(cl[1] - cl[0]) if len(cl) > 1 else 5.0
    min_dist_bins = max(1, int(40.0 / bin_m))

    peaks, _ = find_peaks(prob, height=min_prob, distance=min_dist_bins)
    return [float(cl[i]) for i in peaks]

def advice(lap: pd.DataFrame, ref_brake: list[float], ref_throttle: list[float]):
    df = lap.sort_values("cl_dist")
    brake = extract_events(df, "brake")
    throttle = extract_events(df, "throttle")

    rows: list[dict] = []
    def add(mode: str, events: list[float], refs: list[float]):
        used = set()
        for i, ref_d in enumerate(refs):
            available = [e for e in events if e not in used]
            lap_d = nearest_event(ref_d, available)
            if lap_d is None:
                continue
            used.add(lap_d)
            
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
        for row_num, (_, r) in enumerate(advice_df.iterrows(), start=1):
            lines.append(
                f"{r['mode']} zone {row_num}: {r['advice']} "
                f"(lap={float(r['lap_distance']):.1f}m, ref={float(r['ref_distance']):.1f}m, delta={float(r['delta']):+.1f}m)"
            )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path