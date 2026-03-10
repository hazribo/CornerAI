from pathlib import Path
import numpy as np
import pandas as pd

def _get_threshold_crossings(distance_array, signal_array, threshold=0.5):
    if len(signal_array) < 2:
        return []
        
    above = signal_array >= threshold
    prev_above = np.roll(above, 1)
    prev_above[0] = above[0]
    
    idx = np.where(above & ~prev_above)[0]
    return [float(distance_array[i]) for i in idx]

def _consolidate_alternating(primary_events, secondary_events, min_sep=40.0):
    """
    Groups events if they occur before an alternating counter-event. 
    E.g. Keeps only the first 'brake' event until a 'throttle' event happens.
    Also enforces physical separation so micro-blips don't trigger new zones.
    """
    valid = []
    # Merge both event types into a single distance-sorted timeline
    timeline = [(d, "P") for d in primary_events] + [(d, "S") for d in secondary_events]
    timeline.sort(key=lambda x: x[0])
    
    last_state = None
    last_valid_dist = -1e18
    
    for dist, event_type in timeline:
        if event_type == "P":
            # Only trigger if we aren't already in a primary state
            if last_state != "P":
                if (dist - last_valid_dist) > min_sep:
                    valid.append(dist)
                    last_valid_dist = dist
                last_state = "P"
        else:
            last_state = "S"
            
    return valid

def build_references_from_gt(gt: pd.DataFrame, mode: str, min_prob: float = 0.5) -> list[float]:
    dist = gt["cl_dist"].to_numpy(dtype=float)
    b_array = gt.get("brake_exp", pd.Series(np.zeros(len(gt)))).to_numpy(dtype=float)
    t_array = gt.get("throttle_exp", pd.Series(np.zeros(len(gt)))).to_numpy(dtype=float)
    raw_b = _get_threshold_crossings(dist, b_array, threshold=0.3)
    raw_t = _get_threshold_crossings(dist, t_array, threshold=0.3)
    
    if mode == "brake":
        return _consolidate_alternating(raw_b, raw_t, min_sep=40.0)
    else:
        return _consolidate_alternating(raw_t, raw_b, min_sep=40.0)

def nearest_event(lap: float, refs: list[float]):
    tolerance = float(120.0) 
    if not refs:
        return None
    array = np.asarray(refs)
    i = int(np.argmin(np.abs(array - float(lap))))
    best = float(array[i])
    return best if abs(best - float(lap)) <= tolerance else None

def advice(lap: pd.DataFrame, ref_brake: list[float], ref_throttle: list[float]):
    df = lap.sort_values("cl_dist")
    dist = df["cl_dist"].to_numpy(dtype=float)
    
    # Extract raw player events:
    raw_lap_b = _get_threshold_crossings(dist, df["brake"].to_numpy(dtype=float), threshold=0.6)
    raw_lap_t = _get_threshold_crossings(dist, df["throttle"].to_numpy(dtype=float), threshold=0.6)
    # Consolidate these events:
    lap_brakes = _consolidate_alternating(raw_lap_b, raw_lap_t, min_sep=40.0)
    lap_throttles = _consolidate_alternating(raw_lap_t, raw_lap_b, min_sep=40.0)

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
                advice_text = f"Brake {abs(delta):.1f}m later" if delta < 0 else f"Brake {abs(delta):.1f}m earlier"
            else:
                advice_text = f"Throttle {abs(delta):.1f}m later" if delta < 0 else f"Throttle {abs(delta):.1f}m earlier"

            rows.append({
                "mode": mode,
                "zone_index": i+1,
                "lap_distance": round(float(lap_d), 1),
                "ref_distance": round(float(ref_d), 1),
                "delta": round(delta, 1),
                "advice": advice_text
            })

    add("brake", lap_brakes, ref_brake)
    add("throttle", lap_throttles, ref_throttle)

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