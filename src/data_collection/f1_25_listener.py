import socket
import struct
from datetime import datetime
from pathlib import Path
import csv
import pandas as pd

UDP_IP = "127.0.0.1"
UDP_PORT = 20777
HEADER_SIZE = 29
output_dir = Path(__file__).resolve().parents[2] / "data" / "testing"
output_dir.mkdir(parents=True, exist_ok=True)

udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp.bind((UDP_IP, UDP_PORT))
print("Listening on " + UDP_IP + ":" + str(UDP_PORT))

current_lap = 0
lap_data = []
session_dir = None
recording = False
last_lap_distance = 0.0
session_time = 0.0
lap_start_time = 0.0
current_sector = 1

current_telemetry = {}

def save_lap_csv(filename, car_index, lap_number, lap_time_ms, data_points):
    if not data_points:
        return
    
    df_raw = pd.DataFrame(data_points)
    df = pd.DataFrame()

    df["distance"] = df_raw["lap_distance"]
    df["x"] = df_raw["z_pos"] # swap x and z
    df["y"] = df_raw["x_pos"] # swap y and z (x)
    df["z"] = df_raw["y_pos"]
    # Normalise speed to 0-1:
    speed = pd.to_numeric(df_raw["speed"], errors="coerce")
    max_speed = speed.max(skipna=True)
    df["speed"] = speed / max_speed
    # Normalise throttle, brake, steering to 0-1:
    df["throttle"] = df_raw["throttle"].astype(float) / 100.0
    df["brake"] = df_raw["brake"].astype(float) / 100.0

    #df["steer"] = df_raw["steer [%]"].astype(float) / 100.0
    # uncomment at a later date if steering info can be gathered from ff1.

    df["gear"] = df_raw["gear"]
    df["drs"] = df_raw["drs"]
    df["rpm"] = df_raw["rpm"]
    df["time"] = df_raw["laptime"]
    
    df = df.sort_values("distance").reset_index(drop=True)
    df["source"] = "f125"

    # Assign all other attributes to prevent errors.
    # TODO: fix this so this isn't needed
    df["track"] = "1 melbourne" # change this later - testing purposes.                         
    df["difficulty"] = "0"
    df["year"] = int(2026)
    df["lap_id"] = "player_lap"

    df.to_csv(filename, index=False)
    print(f"Saved {filename} with {len(df)} points")

while True:
    data, addr = udp.recvfrom(4096)
    
    if len(data) < HEADER_SIZE:
        continue

    header = struct.unpack("<HBBBBBQfIIBB", data[:HEADER_SIZE])
    packet_id = header[5]
    session_time = header[7]
    player_car_index = header[10]

    if packet_id == 2:  # Lap Data
        player_offset = HEADER_SIZE + (player_car_index * 57)

        if len(data) >= player_offset + 57:
            lap_info = struct.unpack("<IIHBHBHBHBfffBBBBBBBBBBBBBBBHHBfB", data[player_offset:player_offset+57])
               
            lap_number = lap_info[14]
            lap_time_ms = lap_info[0]
            lap_distance = lap_info[10]
            sector = lap_info[13] + 1  # Convert 0-indexed to 1-indexed

            if session_dir is None:
                session_dir = output_dir / datetime.now().strftime("%Y_%m_%d_%H%M%S")
                session_dir.mkdir(parents=True, exist_ok=True)
                lap_data = []
                lap_start_time = session_time
                current_sector = sector
                print("New session started.")
            
            if lap_number < current_lap:
                session_dir = output_dir / datetime.now().strftime("%Y_%m_%d_%H%M%S")
                session_dir.mkdir(parents=True, exist_ok=True)
                lap_data = []
                lap_start_time = session_time
                current_sector = sector
                print("New session started.")
            
            if lap_number == current_lap and lap_distance < last_lap_distance - 500:
                lap_data = []
                lap_start_time = session_time
                current_sector = sector
                print(f"Lap {lap_number} restarted - cleared telemetry")
            
            if last_lap_distance < 0 and lap_distance >= 0:
                lap_data = []
                lap_start_time = session_time

            last_lap_distance = lap_distance

            # Store lap context for merging with telemetry/motion
            current_telemetry["lap_distance"] = lap_distance
            current_telemetry["sector"] = sector
            current_telemetry["laptime"] = (session_time - lap_start_time) if lap_start_time else 0

            if lap_number > current_lap and current_lap > 0:
                filename = session_dir / f"lap_{current_lap}.csv"
                save_lap_csv(filename, player_car_index, current_lap, lap_time_ms, lap_data)
                lap_data = []
                lap_start_time = session_time
            
            current_lap = lap_number
            current_sector = sector
            recording = current_lap > 0

    # Car Telemetry
    if packet_id == 6 and recording:
        player_offset = HEADER_SIZE + (player_car_index * 33)
        if len(data) >= player_offset + 33:
            tel = struct.unpack("<HfffBbHBBHHBBHfB", data[player_offset:player_offset+33])
            current_telemetry.update({
                "speed": tel[0],
                "throttle": tel[1],
                "steering_angle": tel[2],
                "brake": tel[3],
                "gear": tel[5],
                "rpm": tel[6],
                "drs": tel[7]
            })

    if packet_id == 0 and recording:
        player_offset = HEADER_SIZE + (player_car_index * 60)
        if len(data) >= player_offset + 60:
            motion = struct.unpack("<ffffffhhhhhhffffff", data[player_offset:player_offset+60])
            current_telemetry.update({
                "x_pos": motion[0],
                "y_pos": motion[1],
                "z_pos": motion[2]
            })
            # Only record if lap distance is positive:
            if current_telemetry.get("lap_distance", -1.0) >= 0.0:
                lap_data.append(current_telemetry.copy())