import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

F125_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "f1-25" / "opponent_laps" / "f1_2025"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "f1-25"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean(filepath):
    df_raw = pd.read_csv(filepath, skiprows=7, header=0) # first 7 rows are metadata from Telemetry Tool
    df = pd.DataFrame()
    
    df["distance"] = df_raw["lapdistance [m]"]
    df["x"] = df_raw["z [m]"] # swap x and z
    df["y"] = df_raw["x [m]"] # swap y and z (x)
    df["z"] = df_raw["y [m]"]
    # Add speed, and normalise speed to 0-1 for norm_speed:
    speed = pd.to_numeric(df_raw["speed [m/s]"], errors="coerce")
    max_speed = speed.max(skipna=True)
    df["speed"] = speed
    df["norm_speed"] = speed / max_speed
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
    
    df = interpolate_lap(df, interval_m=5.0)
    return df

def interpolate_lap(df: pd.DataFrame, interval_m: float = 5.0) -> pd.DataFrame:
    if df.empty or "distance" not in df.columns:
        return df

    # Sort by increasing distance so interpolation works:
    df = df.drop_duplicates(subset=["distance"], keep="first").sort_values("distance")
    
    dist_raw = df["distance"].to_numpy(dtype=float)
    min_dist, max_dist = dist_raw[0], dist_raw[-1]
    
    dist_new = np.arange(min_dist, max_dist, interval_m)
    if dist_new[-1] != max_dist:
        dist_new = np.append(dist_new, max_dist)
        
    out_dict = {"distance": dist_new}
    
    float_cols = ["x", "y", "z", "norm_speed", "time"]
    int_cols = ["speed", "throttle", "brake", "gear", "drs", "rpm"]
    
    for col in float_cols + int_cols:
        if col in df.columns:
            y_raw = df[col].to_numpy(dtype=float)
            
            f_interp = interp1d(dist_raw, y_raw, kind='linear', bounds_error=False, fill_value="extrapolate")
            y_new = f_interp(dist_new)
            
            if col in int_cols:
                out_dict[col] = np.round(y_new).astype(int)
            else:
                out_dict[col] = y_new
                
    out_df = pd.DataFrame(out_dict)
    
    if "source" in df.columns:
        out_df["source"] = df["source"].iloc[0]
        
    return out_df

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