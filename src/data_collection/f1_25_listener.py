import socket
import struct
from datetime import datetime
from pathlib import Path
import csv

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
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lap_time_seconds = lap_time_ms / 1000.0
    
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-2]
        writer.writerow(["player", "v3", "Player", car_index, timestamp])
        writer.writerow(["laptime [s]"])
        writer.writerow([f"{lap_time_seconds:.3f}"])
        
        for point in data_points:
            writer.writerow([
                round(point.get("laptime", 0), 3),
                round(point.get("lap_distance", 0), 2),
                point.get("speed", 0),
                round(point.get("throttle", 0) * 100, 8),
                round(point.get("brake", 0) * 100, 8),
                round(point.get("steering_angle", 0), 8),
                point.get("gear", 0),
                point.get("drs", 0),
                round(point.get("x_pos", 0), 8),
                round(point.get("y_pos", 0), 8),
                round(point.get("z_pos", 0), 8),
            ])
    print(f"Saved {filename} with {len(data_points)} points")

while True:
    data, addr = udp.recvfrom(4096)
    
    if len(data) < HEADER_SIZE:
        continue

    header = struct.unpack("<HBBBBBQfIIBB", data[:HEADER_SIZE])
    packet_id = header[5]
    session_time = header[7]
    player_car_index = header[10]

    if packet_id == 1:  # Session
        if len(data) >= HEADER_SIZE + 724:
            assists_data = struct.unpack("<BBBBBBBB", data[HEADER_SIZE+657:HEADER_SIZE+665])
            assists = [
                bool(assists_data[0]),
                bool(assists_data[1]),
                bool(assists_data[2]),
                bool(assists_data[5]),
                bool(assists_data[6]),
                bool(assists_data[7])
            ]

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
            lap_data.append(current_telemetry.copy())