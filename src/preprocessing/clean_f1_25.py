import pandas as pd
from pathlib import Path

# Directories
F125_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "f1-25" / "opponent_laps" / "f1_2025"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "f1-25"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean(filepath):
    df_raw = pd.read_csv(filepath, skiprows=7, header=0) # first 7 rows are metadata from Telemetry Tool
    df_standard = pd.DataFrame()
    
    df_standard["sector"] = df_raw["sector [int]"]
    df_standard["time"] = df_raw["laptime [s]"]
    df_standard["distance"] = df_raw["lapdistance [m]"]
    df_standard["speed"] = df_raw["speed [m/s]"]
    # Normalise throttle, brake, steering to 0-1:
    df_standard["throttle"] = df_raw["throttle [%]"].astype(float) / 100.0
    df_standard["brake"] = df_raw["brake [%]"].astype(float) / 100.0
    df_standard["steer"] = df_raw["steer [%]"].astype(float) / 100.0
    # Gear & RPM likely won't matter, but is collected by Telemetry Tool:
    df_standard["gear"] = df_raw["gear [int]"]
    df_standard["drs"] = df_raw["drs"]
    df_standard["rpm"] = df_raw["revs [int]"]
    df_standard["x"] = df_raw["x [m]"]
    df_standard["y"] = df_raw["y [m]"]
    df_standard["z"] = df_raw["z [m]"]
    
    df_standard = df_standard.sort_values("distance").reset_index(drop=True)
    df_standard["source"] = "f125"
    return df_standard

def process_all_files():
    print("\nProcessing F1 25 telemetry files...")
    for csv_file in F125_DIR.glob("*.csv"):
        try:
            df = clean(csv_file)
            output_path = OUTPUT_DIR / f"{csv_file.stem}_clean.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved {csv_file.name}: ({len(df)} rows).")
        except Exception as e:
            print(f"FAILED TO SAVE {csv_file.name}: {e}")
    
    print(f"\nCleaned files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    process_all_files()