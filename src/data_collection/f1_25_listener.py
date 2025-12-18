import socket
import os
import struct
from datetime import datetime
from pathlib import Path
import json

UDP_IP = "127.0.0.1"
UDP_PORT = 20777
HEADER_SIZE = 29
output_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "f1-25"
output_dir.mkdir(parents=True, exist_ok=True)

udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp.bind((UDP_IP, UDP_PORT))
print ("Listening on " + UDP_IP + ":" + str(UDP_PORT))

current_lap = 0
lap_data = []
assists = list(range(5)) # list of bools?
session_dir = None
recording = False

def ms_to_timestamp(ms: int) -> str:
    minutes, rem = divmod(ms, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"

while True:
    data, addr = udp.recvfrom(4096)
    
    if len(data) < HEADER_SIZE: # header in f1 25 is 29 bytes
        continue

    header = struct.unpack("<HBBBBBQfIIBB", data[:HEADER_SIZE])
    packet_format = header[0] # m_packetFormat (always = 2025)
    packet_id = header[5] # m_packetId
    player_car_index = header[10] # m_playerCarIndex

    if packet_id == 1: # 753 bytes total
        player_offset = HEADER_SIZE + (player_car_index * 103)
        if len(data) >= player_offset + 103:
            lap_info = struct.unpack("<BbbBHBbBHHBBBBBBfbBBBBBBbbbbBBBIIIBBBBBBBBBBBBBBIBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBff", data[player_offset:player_offset+103])
            
            #Assist settings:
            assists = [
                bool(lap_info[29] == 1),  # m_antiLockBrakes
                bool(lap_info[30] == 1),  # m_tractionControl
                bool(lap_info[31] == 1),  # m_dynamicRacingLine
                bool(lap_info[34] == 1),  # m_autoGearbox
                bool(lap_info[35] == 1)   # m_manualClutch
            ]

    # ePacketIdLapData - data about all the lap times of cars in the session
    if packet_id == 2: # 1285 bytes total
        player_offset = HEADER_SIZE + (player_car_index * 57) # each car = 57 bytes of data

        if len(data) >= player_offset + 57:
            lap_info = struct.unpack("<IIHBHBHBHBfffBBBBBBBBBBBBBBBHHBfB", data[player_offset:player_offset+57])
            
            lap_number = lap_info[14]  # m_currentLapNum
            lap_time = ms_to_timestamp(lap_info[0])

            # New directory for each new session (i.e. on file run or on lap 1):
            if session_dir is None:
                session_dir = output_dir / datetime.now().strftime("%Y_%m_%d_%H%M%S")
                session_dir.mkdir(parents=True, exist_ok=True)
                lap_data = []
                print("New session started.")
            if lap_number < current_lap:
                session_dir = output_dir / datetime.now().strftime("%Y_%m_%d_%H%M%S")
                session_dir.mkdir(parents=True, exist_ok=True)
                lap_data = []
                print("New session started.")

            if lap_number > current_lap and current_lap > 0:
                filename = session_dir / f"lap_{current_lap}.json"
                with open(filename, 'w') as f:
                    json.dump({
                        "car_index": player_car_index,
                        "lap_number": current_lap,
                        "lap_time": lap_time,
                        "data_points": len(lap_data),
                        "assists": assists,
                        "telemetry": lap_data
                    }, f, indent=2)
                print(f"Saved lap {current_lap} with {len(lap_data)} points")
                lap_data = []  # Reset for next lap
            
            current_lap = lap_number
            recording = current_lap > 0

    # ePacketIdCarTelemetry - telemetry data for all cars
    if packet_id == 6 and recording: # 1352 bytes total
        player_offset = HEADER_SIZE + (player_car_index * 33) # telemetry = 33 bytes for each car
        if len(data) >= player_offset + 33:
            telemetry_info = struct.unpack("<HfffBbHBBHHBBHfB", data[player_offset:player_offset+33])

            telemetry_data = {
                "speed": telemetry_info[0],
                "throttle": telemetry_info[1],
                "steering_angle": telemetry_info[2],
                "brake": telemetry_info[3],
                "gear": telemetry_info[5],
                "drs": telemetry_info[7]
            }
            lap_data.append(telemetry_data)

    if packet_id == 0 and recording: # 1349 bytes total
        player_offset = HEADER_SIZE + (player_car_index * 60)
        if len(data) >= player_offset + 60:
            telemetry_info = struct.unpack("<ffffffhhhhhhffffff", data[player_offset:player_offset+60])

        telemetry_data = {
            "x_pos": telemetry_info[0],
            "y_pos": telemetry_info[1],
            "z_pos": telemetry_info[2]
        }
        lap_data.append(telemetry_data)