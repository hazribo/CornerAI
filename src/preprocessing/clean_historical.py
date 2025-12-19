import pandas as pd
from pathlib import Path

# Directories
FASTF1_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "historical" / "csv"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "historical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean(filepath):
    df_raw = pd.read_csv(filepath)
    df_standard = pd.DataFrame()
    
    df_standard['distance'] = df_raw['Distance']
    df_standard['x'] = df_raw['X']
    df_standard['y'] = df_raw['Y']
    df_standard['z'] = df_raw['Z']
    df_standard['speed'] = df_raw['Speed'] / 3.6 # Convert KM/H to M/S
    # Normalise throttle, brake to 0-1:
    df_standard['throttle'] = df_raw['Throttle'] / 100.0
    df_standard['brake'] = df_raw['Brake'].apply(lambda x: 1.0 if x == True or x == 'True' else 0.0)
    df_standard['gear'] = df_raw['nGear']
    df_standard['drs'] = df_raw['DRS']
    df_standard['rpm'] = df_raw['RPM']
    if 'Time' in df_raw.columns:
        try:
            df_standard['time'] = pd.to_timedelta(df_raw['Time']).dt.total_seconds()
        except:
            df_standard['time'] = 0.0

    df_standard = df_standard.sort_values('distance').reset_index(drop=True)
    df_standard['source'] = 'fastf1'
    return df_standard

def process_all_files():
    print("\nProcessing FastF1 telemetry files...")
    for csv_file in FASTF1_DIR.glob("*.csv"):
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