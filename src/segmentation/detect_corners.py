import pandas as pd
from pathlib import Path
from itertools import chain

F125_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "f1-25" / "laps"
HISTORICAL_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "historical"

# load one csv files from data/processed/f1-25 and data/processed/historical for each track.
# combine these into single dataframe for corner detection
all_laps = pd.DataFrame()
frames: list[pd.DataFrame] = []

def load_laps():
    f125_files = sorted(F125_PROCESSED_DIR.rglob("*.csv"))
    hist_files = sorted(HISTORICAL_PROCESSED_DIR.rglob("*.csv"))
    all_files = list(chain(f125_files, hist_files))

    #DEBUG:
    #print(f"Found {len(f125_files)} F1-25 processed CSVs under {F125_PROCESSED_DIR}")
    #print(f"Found {len(hist_files)} historical processed CSVs under {HISTORICAL_PROCESSED_DIR}")

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
                print("fail.")

    if HISTORICAL_PROCESSED_DIR.exists():
        for track_dir in sorted([p for p in HISTORICAL_PROCESSED_DIR.iterdir() if p.is_dir()]):
            csvs = sorted(track_dir.rglob("*.csv"))
            if len(csvs) < 150:
                continue  # skip one-off events / small-sample tracks

            fp = csvs[0]  # pick one lap
            try:
                df = pd.read_csv(fp)
                df["track"] = track_dir.name
                frames.append(df)
            except Exception as e:
                print("fail.")

    all_laps = pd.concat(frames, ignore_index=True)
    return all_laps

if __name__ == "__main__":
    all_laps = load_laps()
    print(f"Combined {len(frames)} laps.")