import pandas as pd
import json
from pathlib import Path

F125_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "f1-25" / "laps"
HISTORICAL_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "historical"
CORNER_JSON_DIR = Path(__file__).resolve().parents[2] / "data" / "corner_info" 

def load_laps(min_year = None):
    frames: list[pd.DataFrame] = []
    if F125_PROCESSED_DIR.exists():
        for track_dir in sorted([p for p in F125_PROCESSED_DIR.iterdir() if p.is_dir()]):
            csvs = sorted(track_dir.rglob("*.csv"))
            for fp in csvs:
                try:
                    df = pd.read_csv(fp)
                    df["track"] = track_dir.name
                    df["year"] = 2025
                    df["_lap_file"] = str(fp)
                    frames.append(df)
                except Exception as e:
                    print(f"fail: {e}")

    if HISTORICAL_PROCESSED_DIR.exists():
        for track_dir in sorted([p for p in HISTORICAL_PROCESSED_DIR.iterdir() if p.is_dir()]):
            csvs = sorted(track_dir.rglob("*.csv"))
            for fp in csvs:
                try:
                    df = pd.read_csv(fp)
                    df["track"] = track_dir.name
                    # first 4 digits of the file name:
                    year = int("".join(ch for ch in fp.stem[:4] if ch.isdigit()))
                    df["year"] = year
                    df["_lap_file"] = str(fp)
                    if min_year is not None and year < min_year:
                        continue
                    frames.append(df)
                except Exception as e:
                    print(f"fail: {e}")

    all_laps = pd.concat(frames, ignore_index=True)
    return all_laps

def load_corner_json(json_path: Path) -> pd.DataFrame:
    if not CORNER_JSON_DIR.exists():
        return pd.DataFrame(columns=["track", "id", "type", "apex_m"])

    if json_path is not None:
        files = [Path(json_path)]
    else:
        files = sorted(p for p in CORNER_JSON_DIR.rglob("*.json") if p.is_file())
        files = [p for p in files if p.name.lower() not in {"corner_info.json"}]

    rows: list[dict] = []
    for fp in files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"fail reading {fp}: {e}")
            continue

        track_name = data.get("track", {}).get("name") if isinstance(data.get("track"), dict) else None
        track_name = track_name or fp.stem  # fallback

        for c in data.get("corners", []):
            distances = c.get("distances_m", {}) or {}
            apex = distances.get("apex", None)
            if apex is None:
                continue

            rows.append(
                {
                    "track": track_name,
                    "id": c.get("id"),
                    "type": c.get("type"),
                    "apex_m": float(apex),
                }
            )

    return pd.DataFrame(rows)

if __name__ == "__main__":
    track = "Australian_Grand_Prix"
    all_laps = load_laps(min_year=2021)
    australia_laps = all_laps[all_laps["track"] == track]

    print(f"Loaded rows: {len(australia_laps):,}")
    if australia_laps.empty:
        raise SystemExit("No lap CSVs found for the requested range.")

    australia_fp = CORNER_JSON_DIR / "australia_2021.json"
    corners = load_corner_json(australia_fp)
    print(f"Loaded corners: {len(corners):,} from {australia_fp.name}\n")
    print(f"Corners:\n{corners}")

    distance_col = "distance"
    corners_use = corners.drop(columns=["track"], errors="ignore").copy()
    corners_use["apex_m"] = pd.to_numeric(corners_use["apex_m"], errors="coerce")
    corners_use = corners_use.dropna(subset=["apex_m"]).sort_values("apex_m").reset_index(drop=True)

    laps = australia_laps.copy()
    laps[distance_col] = pd.to_numeric(laps[distance_col], errors="coerce")
    laps["speed"] = pd.to_numeric(laps["speed"], errors="coerce")
    laps = laps.dropna(subset=[distance_col, "_lap_file"]).copy()

    group_cols = ["_lap_file"]

    laps_sorted = (
        laps.sort_values(by=[distance_col] + group_cols, kind="mergesort")
            .reset_index(drop=True)
    )
    lap_keys = (
        laps_sorted[group_cols]
        .drop_duplicates()
        .sort_values(by=group_cols, kind="mergesort")
        .reset_index(drop=True)
    )
    lap_corners = (
        lap_keys.merge(corners_use, how="cross")
        .sort_values(by=["apex_m"] + group_cols, kind="mergesort")
        .reset_index(drop=True)
    )
    apex_samples = pd.merge_asof(
        lap_corners,
        laps_sorted,
        left_on="apex_m",
        right_on=distance_col,
        by=group_cols,
        direction="nearest",
        suffixes=("_corner", ""),
    )

    apex_samples["apex_speed"] = apex_samples["speed"]

    cols = ["_lap_file", "id", "type", "apex_m", "matched_distance_m", "distance_error_m", "apex_speed"]
    cols = [c for c in cols if c in apex_samples.columns]

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_colwidth", 120)

    print("Apex samples (corner speed per file):")
    print(apex_samples.sort_values(["_lap_file", "apex_m"])[cols].to_string(index=False))