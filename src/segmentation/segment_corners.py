import re
import pandas as pd
import yaml
from pathlib import Path

F125_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "f1-25" / "laps"
HISTORICAL_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "historical"
CORNER_INFO_DIR = Path(__file__).resolve().parents[2] / "data" / "corner_info"
CORNER_STATS_DIR = Path(__file__).resolve().with_name("corner_stats.yaml")

TRACK_ALIASES = {
    "abu_dhabi_grand_prix": "abu_dhabi",
    "australian_grand_prix": "australia",
    "austrian_grand_prix": "austria", "styrian_grand_prix": "austria",
    "azerbaijan_grand_prix": "azerbaijan",
    "bahrain_grand_prix": "bahrain",  "sakhir_grand_prix": "sakhir", # sakhir 2020 has alternate layout, bahrain 2020 also existed.
    "belgian_grand_prix": "spa",
    "brazilian_grand_prix": "brazil", "são_paulo_grand_prix": "brazil",
    "british_grand_prix": "britain", "70th_anniversary_grand_prix": "britain",
    "canadian_grand_prix": "canada",
    "chinese_grand_prix": "shanghai",
    "dutch_grand_prix": "zandvoort",
    "eifel_grand_prix": "eifel",
    "emilia_romagna_grand_prix": "imola",
    "french_grand_prix": "france",
    "german_grand_prix": "germany",
    "hungarian_grand_prix": "hungary",
    "italian_grand_prix": "monza",
    "japanese_grand_prix": "suzuka",
    "las_vegas_grand_prix": "vegas",
    "mexican_grand_prix": "mexico", "mexico_city_grand_prix": "mexico",
    "miami_grand_prix": "miami",
    "monaco_grand_prix": "monaco",
    "portuguese_grand_prix": "portugal",
    "qatar_grand_prix": "qatar",
    "russian_grand_prix": "russia",
    "saudi_arabian_grand_prix": "jeddah",
    "singapore_grand_prix": "singapore",
    "spanish_grand_prix": "spain",
    "turkish_grand_prix": "turkey",
    "tuscan_grand_prix": "mugello",
    "united_states_grand_prix": "cota"
}

# normalise track name for easy lookup:
def _norm_track_name(name):
    s = str(name or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s

# get track alias:
def _resolve_track_key(track):
    norm = _norm_track_name(track)
    return TRACK_ALIASES.get(norm, norm)

# get correct year for track layout:
def _get_track_layout(available_years, requested_year):
    if not available_years:
        raise KeyError(f"No available data for this track layout.")
    years = sorted(set(int(y) for y in available_years))
    if requested_year in years:
        return requested_year
    past = [y for y in years if y <= requested_year]
    return max(past) if past else max(years)

def load_laps(min_year=None):
    frames: list[pd.DataFrame] = []

    if F125_PROCESSED_DIR.exists():
        for track_dir in sorted([p for p in F125_PROCESSED_DIR.iterdir() if p.is_dir()]):
            for fp in sorted(track_dir.rglob("*.csv")):
                try:
                    df = pd.read_csv(fp)
                    df["track"] = track_dir.name
                    df["year"] = 2025
                    df["_lap_file"] = str(fp)
                    frames.append(df)
                except Exception as e:
                    print(f"fail reading {fp}: {e}")

    if HISTORICAL_PROCESSED_DIR.exists():
        for track_dir in sorted([p for p in HISTORICAL_PROCESSED_DIR.iterdir() if p.is_dir()]):
            for fp in sorted(track_dir.rglob("*.csv")):
                try:
                    df = pd.read_csv(fp)
                    df["track"] = track_dir.name
                    year = int("".join(ch for ch in fp.stem[:4] if ch.isdigit()))
                    df["year"] = year
                    df["_lap_file"] = str(fp)
                    if min_year is not None and year < min_year:
                        continue
                    frames.append(df)
                except Exception as e:
                    print(f"fail reading {fp}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

def load_corner_yaml(track, year=None):
    with CORNER_STATS_DIR.open("r", encoding="utf-8") as f:
        stats = yaml.safe_load(f) or {}

    track_key = _resolve_track_key(track)
    track_block = stats.get(track_key, None)

    if year is None:
        raise ValueError("year is required.")

    resolved_year = int(year)
    groups = None

    if isinstance(track_block, dict):
        year_to_groups: dict[int, list[dict]] = {}
        for k, v in track_block.items():
            try:
                ky = int(k)
            except Exception:
                continue
            if isinstance(v, list) and len(v) > 0:
                year_to_groups[ky] = v

        if not year_to_groups:
            return (
                pd.DataFrame(columns=["label", "turns_str", "region_start_m", "region_end_m"]),
                resolved_year,
            )

        resolved_year = _get_track_layout(list(year_to_groups.keys()), int(year))
        groups = year_to_groups[resolved_year]

    elif isinstance(track_block, list):
        groups = track_block

    else:
        return (
            pd.DataFrame(columns=["label", "turns_str", "region_start_m", "region_end_m"]),
            resolved_year,
        )

    rows: list[dict] = []
    for g in groups or []:
        label = g.get("label", None)
        turns = g.get("turns", []) or []
        region = g.get("region_size", None)

        if not (isinstance(label, str) and label.strip()):
            continue
        if not (isinstance(region, (list, tuple)) and len(region) == 2):
            continue

        try:
            region_start_m = float(region[0])
            region_end_m = float(region[1])
        except Exception:
            continue

        turns_str = "-".join(str(t) for t in turns) if turns else "unknown"
        rows.append(
            {
                "label": label.strip(),
                "turns_str": turns_str,
                "region_start_m": region_start_m,
                "region_end_m": region_end_m,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["label", "region_start_m", "region_end_m"]).reset_index(drop=True)

    return df, resolved_year

def segment_and_write_laps(track, requested_year, laps, regions, resolved_year):
    regions_use = (
        regions[["label", "turns_str", "region_start_m", "region_end_m"]]
        .dropna(subset=["label", "region_start_m", "region_end_m"])
        .drop_duplicates()
        .sort_values(["label", "region_start_m", "region_end_m"])
        .reset_index(drop=True)
    )

    laps = laps.copy()
    laps["distance"] = pd.to_numeric(laps["distance"], errors="coerce")
    laps = laps.dropna(subset=["distance", "_lap_file"]).copy()

    desired_cols = [
        "time", "distance", "x", "y", "z", "speed", "throttle", "brake",
        "rpm", "gear", "drs", "source",
    ]

    track_key = _resolve_track_key(track)

    wrote = 0
    for lap_file, g in laps.groupby("_lap_file", sort=False):
        g = g.sort_values("distance").reset_index(drop=True)
        lap_stem = Path(lap_file).stem

        for _, r in regions_use.iterrows():
            label = str(r["label"])
            turns_str = str(r["turns_str"])
            start_m = float(r["region_start_m"])
            end_m = float(r["region_end_m"])

            out_root = CORNER_INFO_DIR / label
            out_root.mkdir(parents=True, exist_ok=True)

            seg = g[(g["distance"] >= start_m) & (g["distance"] <= end_m)].copy()
            if seg.empty:
                continue

            peak_speed = None
            if "speed" in seg.columns:
                sp = pd.to_numeric(seg["speed"], errors="coerce")
                if sp.notna().any():
                    peak_speed = float(sp.max())

            seg = seg.drop(columns=["track", "year", "_lap_file"], errors="ignore")
            seg = seg.reindex(columns=desired_cols)

            safe_turns = re.sub(r"[^0-9A-Za-z_\-]+", "_", turns_str)
            out_fp = out_root / f"{track_key}_{int(requested_year)}_{lap_stem}_T{safe_turns}.csv"
            if out_fp.exists():
                continue

            meta = (
                f"# track={track_key}, requested_year={int(requested_year)}, layout_year={int(resolved_year)}, "
                f"label={label}, turns={turns_str}, region_m=[{start_m:.1f},{end_m:.1f}], "
                f"peak_speed={'' if peak_speed is None else f'{peak_speed:.3f}'}"
            )

            with out_fp.open("w", encoding="utf-8", newline="") as f:
                f.write(meta + "\n")
                seg.to_csv(f, index=False)

            wrote += 1

    print(f"Wrote {wrote:,} segment CSV(s) under: {CORNER_INFO_DIR}")
    return CORNER_INFO_DIR

if __name__ == "__main__":
    tracks = list(TRACK_ALIASES.keys())

    all_laps = load_laps()
    if all_laps.empty:
        raise Exception("No lap CSVs found.")

    for track in tracks:
        track_laps = all_laps[(all_laps["track"]).map(_norm_track_name) == track].copy()
        if track_laps.empty:
            print(f"No lap CSVs found for {track}.")
            continue
        else:
            print(f"Loaded rows: {len(track_laps):,}")

        years = pd.to_numeric(track_laps["year"], errors="coerce").dropna().astype(int)
        for requested_year in sorted(years.unique()):
            year_laps = track_laps[track_laps["year"] == requested_year].copy()
            regions, layout_year = load_corner_yaml(track=track, year=requested_year)
            print(f"Loaded regions: {len(regions):,} for {track} ({layout_year} layout)")
            if regions.empty:
                print(f"No labeled regions with region_size in corner_stats.yaml for {track} {layout_year}.")
                continue

            segment_and_write_laps(track, requested_year, year_laps, regions, layout_year)