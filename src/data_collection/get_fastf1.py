import fastf1 as ff1
import os
import warnings
import logging
import shutil
import pandas as pd
from pathlib import Path

logging.getLogger("fastf1").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=FutureWarning)
cache_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "historical"
os.makedirs(cache_dir, exist_ok=True)
ff1.Cache.enable_cache(cache_dir)

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
SESSION = "Q"

for year in YEARS:
    schedule = None
    try:
        schedule = ff1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        print(f"Could not load schedule for {year}. Reason: {e}")
        continue

    for _, event in schedule.iterrows():
        # if event folder exists + has all 10 ff1pkl files, skip loading:
        event_folder = cache_dir / f"{year}/{year}-{event.EventDate.strftime('%m-%d')}_{event.EventName.replace(' ', '_')}"
        if event_folder.exists():
            quali_folder = event_folder / f"{year}-{(event.EventDate - pd.Timedelta(days=1)).strftime('%m-%d')}_Qualifying"
            if quali_folder.exists() and len(os.listdir(quali_folder)) == 10:
                print(f"Skipping {year} {event.EventName}. Already loaded.")
                continue
        try:
            quali = ff1.get_session(year, event.EventName, SESSION)
            quali.load(laps=True)
        except Exception as e:
            print(f"Skipping {year} {event.EventName}. Reason: {e}")
            continue
        
        try:
            print(f"Loaded {year} {event.EventName}.")
        except Exception as e:
            print(f"Could not load {year} {event.EventName}. Reason: {e}")
            # Delete potentially corrupted cache folder:
            shutil.rmtree(event_folder)
            continue