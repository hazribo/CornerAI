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
    valid = []
    # Merge both event types into a single distance-sorted timeline
    timeline = [(d, "P") for d in primary_events] + [(d, "S") for d in secondary_events]
    timeline.sort(key=lambda x: x[0])
    
    last_state = None
    last_valid_dist = -1e18
    
    for dist, event_type in timeline:
        if event_type == "P":
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

def advice(lap: pd.DataFrame, ref_brake: list[float], ref_throttle: list[float], gt: pd.DataFrame = None):
    p_df = lap.sort_values("cl_dist").copy()
    dist = p_df["cl_dist"].to_numpy(dtype=float)
    
    # Get raw events for brake and throttle from player lap:
    raw_lap_b = _get_threshold_crossings(dist, p_df["brake"].to_numpy(dtype=float), threshold=0.6)
    raw_lap_t = _get_threshold_crossings(dist, p_df["throttle"].to_numpy(dtype=float), threshold=0.6)
    
    # Build corner segments using the ground truth:
    corner_segments = []
    sorted_brakes = sorted(ref_brake)
    
    for i, b_dist in enumerate(sorted_brakes):
        # Start segment 50m before braking zone:
        seg_start = b_dist - 50.0  
        # End segment 50m before the next braking zone (or end of lap):
        if i < len(sorted_brakes) - 1:
            seg_end = sorted_brakes[i+1] - 50.0
        else:
            # For last corner -> end at end of lap:
            seg_end = p_df["cl_dist"].max()
            
        # Find AI first throttle use:
        ai_throttles_in_seg = [t for t in ref_throttle if seg_start < t < seg_end]
        ai_throttle_dist = ai_throttles_in_seg[0] if ai_throttles_in_seg else None
            
        corner_segments.append({
            "corner_id": i + 1,
            "ai_brake": b_dist,
            "ai_throttle": ai_throttle_dist,
            "seg_start": seg_start,
            "seg_end": seg_end
        })

    rows = []
    for corner in corner_segments:
        # Match player's braking point to get the exact brake distance:
        player_b_dist = nearest_event(corner["ai_brake"], raw_lap_b)
        # Get entry speed for exact point player braked:
        p_entry_speed = 0
        if player_b_dist is not None:
            # Find the closest telemetry row to the player's braking distance
            p_b_slice = p_df.iloc[(p_df["cl_dist"] - player_b_dist).abs().argsort()[:1]]
            p_entry_speed = p_b_slice["speed"].iloc[0] if not p_b_slice.empty else 0
        
        # Get entry speed for where the AI braked:
        ai_entry_speed = 0
        if gt is not None and not gt.empty:
            ai_slice = gt.iloc[(gt["cl_dist"] - corner["ai_brake"]).abs().argsort()[:1]]
            ai_entry_speed = ai_slice["speed_exp"].iloc[0] if not ai_slice.empty else 0

        if player_b_dist is not None:
            raw_delta = player_b_dist - corner["ai_brake"]
            
            # Convert km/h to m/s:
            p_v_ms = float(p_entry_speed) / 3.6
            ai_v_ms = float(ai_entry_speed) / 3.6
            # Assume constant F1 deceleration of ~4.5G (44.1 m/s^2)
            a = 44.1
            # Calculate stopping distance for player and AI:
            p_stop_dist = (p_v_ms**2) / (2 * a)
            ai_stop_dist = (ai_v_ms**2) / (2 * a)
            # Use these to calculate the stopping offset:
            physics_offset = p_stop_dist - ai_stop_dist
            adjusted_delta = raw_delta + physics_offset
            
            if adjusted_delta < -3.0: 
                dist_str = f"Brake {abs(adjusted_delta):.1f}m later"
            elif adjusted_delta > 3.0:
                dist_str = f"Brake {abs(adjusted_delta):.1f}m earlier"
            else:
                dist_str = "Optimal braking."
        else:
            dist_str = "Missed braking zone."

        # Getting back on throttle is usually the same regardless of speed (generalising), so not needed below.
        throttle_str = ""
        if corner["ai_throttle"] is not None:
            player_t_dist = nearest_event(corner["ai_throttle"], raw_lap_t)
            if player_t_dist is not None:
                t_delta = player_t_dist - corner["ai_throttle"]
                throttle_str = f"Throttle {abs(t_delta):.1f}m later" if t_delta < 0 else f"Throttle {abs(t_delta):.1f}m earlier"
            else:
                throttle_str = "Missed throttle zone."

        # Calculate player time:
        p_slice = p_df[(p_df["cl_dist"] >= corner["seg_start"]) & (p_df["cl_dist"] <= corner["seg_end"])].copy()

        # Calculate player time:
        p_time = 0.0
        if len(p_slice) > 1:
            dist_diffs = p_slice["cl_dist"].diff().fillna(0.0).to_numpy()
            speeds_ms = (p_slice["speed"].clip(lower=1.0) / 3.6).to_numpy() 
            p_time = np.sum(dist_diffs / speeds_ms) # dt = dx / v

        # Calculate AI time:
        ai_time = 0.0
        if gt is not None and not gt.empty:
            ai_slice = gt[(gt["cl_dist"] >= corner["seg_start"]) & (gt["cl_dist"] <= corner["seg_end"])].copy()
            if len(ai_slice) > 1:
                ai_dist_diffs = ai_slice["cl_dist"].diff().fillna(0.0).to_numpy()
                ai_speeds_ms = (ai_slice["speed_exp"].clip(lower=1.0) / 3.6).to_numpy()
                ai_time = np.sum(ai_dist_diffs / ai_speeds_ms)

        # Skip corners where telemetry is missing/broken:
        if p_time == 0 or ai_time == 0:
            continue

        time_lost = p_time - ai_time

        # Get braking marks for advice:
        if player_b_dist is not None:
            brake_marks = f"\n     Braking mark: {player_b_dist:.1f}m (Expected: {corner['ai_brake']:.1f}m)"
        else:
            brake_marks = f"\n     Braking mark: Missed (Expected: {corner['ai_brake']:.1f}m)"
        
        # Combine all feedback strings:
        advice_text = f"{dist_str} (Entry speed: {p_entry_speed:.0f}km/h vs Expected: {ai_entry_speed:.0f}km/h){brake_marks}\n     {throttle_str}"
        
        rows.append({
            "corner_id": corner["corner_id"],
            "time_lost_s": time_lost,
            "advice": advice_text,
            "seg_start": corner["seg_start"],
            "seg_end": corner["seg_end"]
        })

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df

    # Sort by worst segment first:
    out_df = out_df.sort_values("time_lost_s", ascending=False).reset_index(drop=True)
    return out_df

def write_advice(advice_df: pd.DataFrame, out_path: Path, track_name: str, lap_id: str) -> Path:
    lines = [f"Track: {track_name}", f"Lap: {lap_id}", ""]
    if advice_df.empty:
        lines.append("No advice events found.")
    else:
        for row_num, (_, r) in enumerate(advice_df.iterrows(), start=1):
            time_str = f"Lost {r['time_lost_s']:.2f}s" if r['time_lost_s'] > 0 else f"Gained {abs(r['time_lost_s']):.2f}s"
            lines.append(
                f"Priority {row_num} (Zone {r['corner_id']}): {time_str}"
                f"\n     {r['advice']}"
                f"\n     (Segment size: {r['seg_start']:.0f}m to {r['seg_end']:.0f}m)\n"
            )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path