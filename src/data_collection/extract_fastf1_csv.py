import fastf1 as ff1
import os
import warnings
import logging

logging.getLogger("fastf1").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", category=FutureWarning)
cache_dir = "./data/raw/historical/"
ff1.Cache.enable_cache(cache_dir)
ff1.Cache.offline = True # can only use cached data

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
SESSION = "Q"

# no try-except needed: only using data from cached files
for year in YEARS:
    schedule = ff1.get_event_schedule(year, include_testing=False)

    for _, event in schedule.iterrows():
        quali = ff1.get_session(year, event.EventName, SESSION)
        quali.load(laps=True)
        
        # Save each driver's individual lap telemetry data to csv.
        for driver in quali.laps["Driver"].dropna().unique():
            driver_laps = (quali.laps.pick_driver(driver).pick_quicklaps()) # pick_quicklaps() removes in/out laps

            for _, lap in driver_laps.iterlaps():
                try:
                    tel = lap.get_telemetry().copy() 
                except Exception:
                    continue # skips if get_telemetry() fails - some laps have no add_driver_ahead() data and so break.
                # get laptime for filename:
                laptime = lap["LapTime"].total_seconds()
                laptime = f"{laptime:.3f}" # laptime must be to 3dp

                # Save to CSV file:
                output_dir = f"./data/raw/historical/csv/{event.EventName}".replace(' ', '_')
                os.makedirs(output_dir, exist_ok=True)
                output_path = f"{output_dir}/{year}_{driver}_{laptime}.csv"
                tel.to_csv(output_path, index=False)
                print(f"Saved {output_path}.")

        print(f"Loaded {year} {event.EventName}.")