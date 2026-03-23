import socket
import struct
from datetime import datetime
import pandas as pd

# Add src/modelling to path to load model/advice files:
import sys
from pathlib import Path
modelling_dir = Path(__file__).resolve().parents[1] / "modelling"
sys.path.append(str(modelling_dir))
try:
    from game_model import RandomForestModel, Curvature, project_to_centreline, add_should_brake, add_should_throttle # type: ignore
    from game_advice import build_references_from_gt, advice, write_advice # type: ignore
except ImportError as e:
    print(f"Warning: {e}")
    
UDP_IP = "127.0.0.1"
UDP_PORT = 20777
HEADER_SIZE = 29
# Map all track IDs since they don't line up with calendar order:
TRACK_IDS = {
    0: "1 melbourne",
    2: "2 shanghai",
    3: "4 sakhir",
    4: "9 calalunya",
    5: "8 monaco",
    6: "10 montreal",
    7: "12 silverstone",
    9: "14 hungary",
    10: "13 spa",
    11: "16 monza",
    12: "18 singapore",
    13: "3 suzuka",
    14: "24 abu dhabi",
    15: "19 cota",
    16: "21 brazil",
    17: "11 austria",
    19: "20 mexico",
    20: "17 baku",
    26: "15 zandvoort",
    27: "7 imola",
    29: "5 jeddah",
    30: "6 miami",
    31: "22 vegas",
    32: "23 qatar",
}

output_dir = Path(__file__).resolve().parents[2] / "data" / "testing"
output_dir.mkdir(parents=True, exist_ok=True)
models_dir = Path(__file__).resolve().parents[2] / "data" / "models" / "f1-25"
model_path = models_dir / "game_model.joblib"
if model_path.exists():
    try:
        model = RandomForestModel.load_model(model_path)
        print(f"Loading model from {model_path}.")
    except FileExistsError as e:
        print(f"Warning: {e}")

current_lap = 0
lap_data = []
session_dir = None
recording = False
last_lap_distance = 0.0
session_time = 0.0
lap_start_time = 0.0
current_sector = 1
current_telemetry = {}
current_track_id = -1

udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp.bind((UDP_IP, UDP_PORT))
print("Listening on " + UDP_IP + ":" + str(UDP_PORT))                            

def get_advice(filename: Path, df: pd.DataFrame):
    target_track = df["track"].iloc[0                                         ]
    target_lap_id = str(filename)
    gt_path = models_dir / f"{target_track}_ground_truth.csv"

    gt = pd.read_csv(gt_path)
    ref_brake = build_references_from_gt(gt, mode="brake")
    ref_throttle = build_references_from_gt(gt, mode="throttle")

    player_lap = pd.read_csv(filename)
    player_lap = Curvature.add_curv_cols(df, n_cols=4)
    player_lap = model.predict_probability(player_lap)
    # Project player's coordinates to the nearest centreline point for cl_dist:
    cl = gt[["x_exp", "y_exp", "cl_dist"]].rename(columns={"x_exp": "x", "y_exp": "y"})
    player_lap = project_to_centreline(player_lap, cl)

    lap_df = add_should_brake(player_lap, gt)
    lap_df = add_should_throttle(player_lap, gt).sort_values("cl_dist")

    advice_df = advice(lap_df, ref_brake, ref_throttle, gt=gt)
    advice_path = output_dir / f"{target_lap_id}_advice.txt"
    write_advice(advice_df, advice_path, track_name=target_track, lap_id=filename)
    print(f"Saved advice to {advice_path}")

def save_lap_csv(filename, data_points):
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
    df["speed"] = speed
    df["norm_speed"] = speed / max_speed
    # Normalise throttle, brake, steering to 0-1:
    df["throttle"] = df_raw["throttle"].astype(float)
    df["brake"] = df_raw["brake"].astype(float)

    #df["steer"] = df_raw["steer [%]"].astype(float) / 100.0
    # uncomment at a later date if steering info can be gathered from ff1.

    df["gear"] = df_raw["gear"]
    df["drs"] = df_raw["drs"]
    df["rpm"] = df_raw["rpm"]
    df["time"] = df_raw["laptime"]
    
    df = df.sort_values("distance").reset_index(drop=True)
    df["source"] = "f125"

    df["track"] = TRACK_IDS.get(current_track_id)              
    df["difficulty"] = "999" # placeholder for "player"
    df["year"] = year
    df["lap_id"] = filename

    # Save telemetry to CSV:
    df.to_csv(filename, index=False)
    print(f"Saved {filename} with {len(df)} points")
    # Get advice for this lap; will also be saved:
    get_advice(filename, df)

while True:
    data, addr = udp.recvfrom(4096)
    
    # Ignore any data of invalid size:
    if len(data) < HEADER_SIZE:
        continue

    # Get all necessary data from header:
    header = struct.unpack("<HBBBBBQfIIBB", data[:HEADER_SIZE])
    year = header[2] # for df["year"]
    packet_id = header[5]
    session_time = header[7]
    player_car_index = header[10]

    # Motion Data:
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

    # Session Data:
    if packet_id == 1 and recording:
        if len(data) >= HEADER_SIZE + 8:
            session_info = struct.unpack("<BbbBHbB", data[HEADER_SIZE:HEADER_SIZE+8])
            current_track_id = session_info[6]

    # Lap Data
    if packet_id == 2:
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
                save_lap_csv(filename, lap_data)
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