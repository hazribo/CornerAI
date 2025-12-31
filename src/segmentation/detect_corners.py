import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks

F125_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "f1-25" / "laps"
HISTORICAL_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "historical"

frames: list[pd.DataFrame] = []
def load_laps():
    if F125_PROCESSED_DIR.exists():
        for track_dir in sorted([p for p in F125_PROCESSED_DIR.iterdir() if p.is_dir()]):
            csvs = sorted(track_dir.rglob("*.csv"))
            if not csvs:
                continue

            fp = csvs[0]  # pick one lap
            try:
                df = pd.read_csv(fp)
                df["track"] = track_dir.name
                frames.append(df)
            except Exception as e:
                print(f"fail: {e}")

    if HISTORICAL_PROCESSED_DIR.exists():
        for track_dir in sorted([p for p in HISTORICAL_PROCESSED_DIR.iterdir() if p.is_dir()]):
            csvs = sorted(track_dir.rglob("*.csv"))
            if len(csvs) < 150:
                continue  # skip one-off events (e.g. 70th anniversary)

            fp = csvs[0]  # pick one lap
            try:
                df = pd.read_csv(fp)
                df["track"] = track_dir.name
                frames.append(df)
            except Exception as e:
                print(f"fail: {e}")

    all_laps = pd.concat(frames, ignore_index=True)
    return all_laps

def detect_corners(tel: pd.DataFrame):
    tel = tel.sort_values("distance", kind="mergesort").dropna(subset=["distance", "speed", "x", "y"])
    tel = tel.loc[~tel["distance"].duplicated(keep="first")].reset_index(drop=True)
    speed = tel["speed"].to_numpy(dtype=float)
    distance = tel["distance"].to_numpy(dtype=float)
    x = tel["x"].to_numpy(dtype=float)
    y = tel["y"].to_numpy(dtype=float)

    # calculate speed peaks minima:
    speed_peaks, _ = find_peaks(-speed)

    # calculate turn peaks (can't use steering natively as not included in ff1):
    dx = np.gradient(x, distance)
    dy = np.gradient(y, distance)
    heading = np.unwrap(np.arctan2(dy, dx))
    turn_rate = np.abs(np.gradient(heading, distance))
    turn_peaks, _ = find_peaks(turn_rate, height=1)

    # group both, remove identical corners:
    all_peaks = np.unique(np.concatenate([speed_peaks, turn_peaks]))
    all_peaks = np.sort(all_peaks)

    corner_info = []
    for idx in all_peaks:
        corner_info.append({
            "index": int(idx),
            "distance": float(distance[idx]),
            "speed": float(speed[idx]),
            "x": float(x[idx]),
            "y": float(y[idx]),
        })
    
    track_name = tel["track"].iloc[0]
    return {"count": len(all_peaks), "corners": corner_info, "track": track_name}

if __name__ == "__main__":
    all_laps = load_laps()
    print(f"Combined {len(frames)} laps.\n")

    for lap in frames:
        corners = detect_corners(lap)
        track_name = corners["track"]
        corner_count = corners["count"]
        print(f"{track_name:30s} | Corners detected: {corner_count:2d}")