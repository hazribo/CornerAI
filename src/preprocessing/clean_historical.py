import pandas as pd
from pathlib import Path

FASTF1_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "historical" / "csv"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "historical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean(filepath):
    df_raw = pd.read_csv(filepath)
    df = pd.DataFrame()
    
    df["distance"] = df_raw["Distance"]
    # Convert x, y, z from decimetres to metres:
    df["x"] = df_raw["X"] / 10
    df["y"] = df_raw["Y"] / 10
    df["z"] = df_raw["Z"] / 10
    df["speed"] = df_raw["Speed"] # km/h
    # Normalise throttle, brake to 0-1:
    df["throttle"] = df_raw["Throttle"].apply(lambda x: 1.0 if x > 95 else 0.0)
    df["brake"] = df_raw["Brake"].apply(lambda x: 1.0 if x == True or x == "True" else 0.0)
    df["gear"] = df_raw["nGear"]
    df["drs"] = df_raw["DRS"].apply(lambda x: 1.0 if x == 12 else 0.0)
    df["rpm"] = df_raw["RPM"]
    if "Time" in df_raw.columns:
        try:
            df["time"] = pd.to_timedelta(df_raw["Time"]).dt.total_seconds()
        except:
            df["time"] = 0.0

    df = df.sort_values("distance").reset_index(drop=True)
    df["source"] = "fastf1"
    return df

def process_all_files():
    print("\nProcessing FastF1 telemetry files...")
    for csv_file in FASTF1_DIR.rglob("*.csv"):
        try:
            df = clean(csv_file)
            # mirror directory stucture for output:
            rel = csv_file.relative_to(FASTF1_DIR)              
            out_dir = OUTPUT_DIR / rel.parent                   
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = out_dir / f"{csv_file.stem}_clean.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved {output_path} ({len(df)} rows).")
        except Exception as e:
            print(f"FAILED {csv_file}: {e}")
    
    print(f"\nCleaned files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    process_all_files()