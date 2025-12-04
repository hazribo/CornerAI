import fastf1 as ff1
import numpy as np
import pandas as pd
import os
import warnings
import logging
import time

logging.getLogger("fastf1").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=FutureWarning)
cache_dir = "../data/raw/historical/"
os.makedirs(cache_dir, exist_ok=True)
ff1.Cache.enable_cache(cache_dir)

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
SESSION = "Q"

all_data = []
for year in YEARS:
    schedule = None
    try:
        schedule = ff1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        print(f"Could not load schedule for {year}. Reason: {e}")
        continue

    for _, event in schedule.iterrows():
        try:
            quali = ff1.get_session(year, event.EventName, SESSION)
            quali.load(laps=True)
        except Exception as e:
            print(f"Skipping {year} {event.EventName}. Reason: {e}")
            continue
        
        try:
            all_data.append(quali.laps)
            print(f"Loaded {year} {event.EventName}.")
            #time.sleep(1) # might help rate limits?
        except Exception as e:
            print(f"Could not load {year} {event.EventName}. Reason: {e}")
            continue