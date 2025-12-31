import pandas as pd
from pathlib import Path

F125_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "f1-25" / "opponent_laps" / "f1_2025"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "f1-25"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean(filepath):
    df_raw = pd.read_csv(filepath, skiprows=7, header=0) # first 7 rows are metadata from Telemetry Tool
    df = pd.DataFrame()

    
    df["distance"] = df_raw["lapdistance [m]"]
    df["x"] = df_raw["x [m]"]
    df["y"] = df_raw["z [m]"] # swap y and z
    df["z"] = df_raw["y [m]"]
    df["speed"] = df_raw["speed [m/s]"] # it says m/s but it's in km/h? weird telemetry tool issue
    # Normalise throttle, brake, steering to 0-1:
    df["throttle"] = df_raw["throttle [%]"].astype(float) / 100.0
    df["brake"] = df_raw["brake [%]"].astype(float) / 100.0

    # No steering in ff1 dataset... so don't include for now here:
    #df["steer"] = df_raw["steer [%]"].astype(float) / 100.0
    # uncomment at a later date if steering info can be gathered from ff1.

    df["gear"] = df_raw["gear [int]"]
    df["drs"] = df_raw["drs"]
    df["rpm"] = df_raw["revs [int]"]
    df["time"] = df_raw["laptime [s]"]
    
    df = df.sort_values("distance").reset_index(drop=True)
    df["source"] = "f125"
    return df

def process_all_files():
    print("\nProcessing F1 25 telemetry files...")
    for csv_file in F125_DIR.rglob("*.csv"):
        try:
            df = clean(csv_file)
            # mirror directory stucture for output:
            rel = csv_file.relative_to(F125_DIR)              
            out_dir = OUTPUT_DIR / rel.parent                   
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"{csv_file.stem}_clean.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved {csv_file.name}: ({len(df)} rows).")
        except Exception as e:
            print(f"FAILED TO SAVE {csv_file.name}: {e}")
    
    print(f"\nCleaned files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    process_all_files()