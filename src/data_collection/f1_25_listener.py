import socket
import struct
from datetime import datetime
import pandas as pd
# imports for real-time/overlay:
import threading
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
# Add src/modelling to path to load model/advice files:
import sys
from pathlib import Path
modelling_dir = Path(__file__).resolve().parents[1] / "modelling"
sys.path.append(str(modelling_dir))
try:
    from game_model import RandomForestModel, Curvature, project_to_centreline, add_should_brake, add_should_throttle # type: ignore
    from game_advice import build_references_from_gt, advice, write_advice # type: ignore
    from track_plots import PlotTrackMaps # type: ignore
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

class UDPListener(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.gt_df = None
        self.gt_distances = np.array([])
        self.gt_speeds = np.array([])
        
        # All required variables/values:
        self.current_lap = 0
        self.lap_data = []
        self.session_dir = None
        self.recording = False
        self.last_lap_distance = 0.0
        self.lap_start_time = 0.0
        self.current_sector = 1
        self.current_telemetry = {}
        self.current_track_id = -1
        # Initialise UDP socket:
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.bind((UDP_IP, UDP_PORT))
        print("Listening on " + UDP_IP + ":" + str(UDP_PORT))
        
    def load_ground_truth(self, track_name):
        gt_path = models_dir / f"{track_name}_ground_truth.csv"
        if gt_path.exists():
            self.gt_df = pd.read_csv(gt_path)
            self.gt_distances = self.gt_df["cl_dist"].values
            self.gt_speeds = self.gt_df["speed_exp"].values
            
            # Use your existing advice logic to get the AI braking points:
            ref_brake = build_references_from_gt(self.gt_df, mode="brake")
            
            self.ai_braking_zones = []
            for b_dist in ref_brake:
                # Find AI speed exactly at this braking point:
                ai_slice = self.gt_df.iloc[(self.gt_df["cl_dist"] - b_dist).abs().argsort()[:1]]
                if not ai_slice.empty:
                    ai_v_ms = ai_slice["speed_exp"].iloc[0] / 3.6
                    self.ai_braking_zones.append({
                        "ai_brake_dist": float(b_dist),
                        "ai_v_ms": float(ai_v_ms)
                    })
                    
            print(f"Loaded ground truth for {track_name} with {len(self.ai_braking_zones)} braking zones")

    def run(self):
        while True:
            data, addr = self.udp.recvfrom(4096)
            
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
            if packet_id == 0 and self.recording:
                player_offset = HEADER_SIZE + (player_car_index * 60)
                if len(data) >= player_offset + 60:
                    motion = struct.unpack("<ffffffhhhhhhffffff", data[player_offset:player_offset+60])
                    self.current_telemetry.update({
                        "x_pos": motion[0],
                        "y_pos": motion[1],
                        "z_pos": motion[2]
                    })
                    # Only record if lap distance is positive:
                    if self.current_telemetry.get("lap_distance", -1.0) >= 0.0:
                        self.lap_data.append(self.current_telemetry.copy())

            # Session Data:
            if packet_id == 1 and self.recording:
                if len(data) >= HEADER_SIZE + 8:
                    session_info = struct.unpack("<BbbBHbB", data[HEADER_SIZE:HEADER_SIZE+8])
                    new_track_id = session_info[6]
                    if new_track_id != self.current_track_id:
                        self.current_track_id = new_track_id
                        track_name = TRACK_IDS.get(self.current_track_id)
                        if track_name:
                            self.load_ground_truth(track_name)

            # Lap Data
            if packet_id == 2:
                player_offset = HEADER_SIZE + (player_car_index * 57)

                if len(data) >= player_offset + 57:
                    lap_info = struct.unpack("<IIHBHBHBHBfffBBBBBBBBBBBBBBBHHBfB", data[player_offset:player_offset+57])
               
                    lap_number = lap_info[14]
                    lap_time_ms = lap_info[0]
                    lap_distance = lap_info[10]
                    sector = lap_info[13] + 1  # Convert 0-indexed to 1-indexed

                    if self.session_dir is None:
                        self.session_dir = output_dir / datetime.now().strftime("%Y_%m_%d_%H%M%S")
                        self.session_dir.mkdir(parents=True, exist_ok=True)
                        self.lap_data = []
                        self.lap_start_time = session_time
                        self.current_sector = sector
                        print("New session started.")
                    
                    if lap_number < self.current_lap:
                        self.session_dir = output_dir / datetime.now().strftime("%Y_%m_%d_%H%M%S")
                        self.session_dir.mkdir(parents=True, exist_ok=True)
                        self.lap_data = []
                        self.lap_start_time = session_time
                        self.current_sector = sector
                        print("New session started.")
                    
                    if lap_number == self.current_lap and lap_distance < self.last_lap_distance - 500:
                        self.lap_data = []
                        self.lap_start_time = session_time
                        self.current_sector = sector
                        print(f"Lap {lap_number} restarted - cleared telemetry")
                    
                    if self.last_lap_distance < 0 and lap_distance >= 0:
                        self.lap_data = []
                        self.lap_start_time = session_time

                    self.last_lap_distance = lap_distance

                    # Store lap context for merging with telemetry/motion:
                    self.current_telemetry["lap_distance"] = lap_distance
                    self.current_telemetry["sector"] = sector
                    self.current_telemetry["laptime"] = (session_time - self.lap_start_time) if self.lap_start_time else 0

                    if lap_number > self.current_lap and self.current_lap > 0:
                        filename = self.session_dir / f"lap_{self.current_lap}.csv"
                        self.save_lap_csv(filename, self.lap_data)
                        self.lap_data = []
                        self.lap_start_time = session_time
            
                    self.current_lap = lap_number
                    self.current_sector = sector
                    self.recording = self.current_lap > 0

            # Car Telemetry
            if packet_id == 6 and self.recording:
                player_offset = HEADER_SIZE + (player_car_index * 33)
                if len(data) >= player_offset + 33:
                    tel = struct.unpack("<HfffBbHBBHHBBHfB", data[player_offset:player_offset+33])
                    self.current_telemetry.update({
                        "speed": tel[0],
                        "throttle": tel[1],
                        "steering_angle": tel[2],
                        "brake": tel[3],
                        "gear": tel[5],
                        "rpm": tel[6],
                        "drs": tel[7]
                    })

    def get_advice(self, filename: Path, df: pd.DataFrame):
        target_track = df["track"].iloc[0]
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
        print(f"Saved advice to {advice_path}.")

        # Generate plot comparisons and also save to advice path:
        PlotTrackMaps.plot_lap_comparison(user_df=player_lap, gt_df=gt, track_name=str(target_track), out_dir=output_dir)
        print(f"Saved lap comparison plots to {advice_path}.")

    def save_lap_csv(self, filename, data_points):
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

        df["track"] = TRACK_IDS.get(self.current_track_id)              
        df["difficulty"] = "999" # placeholder for "player"
        df["year"] = "2026" # placeholder - year value doesn't really matter for game telemetry
        df["lap_id"] = filename

        # Save telemetry to CSV:
        df.to_csv(filename, index=False)
        print(f"Saved {filename} with {len(df)} points")
        # Get advice for this lap; will also be saved:
        self.get_advice(filename, df)

# Overlay polls UDPListener without interrupting it:
class Overlay(QWidget):
    def __init__(self, listener: UDPListener):
        super().__init__()
        self.listener = listener
        # Tracking values for sound cues:
        self.last_dist = 0.0
        self.beeped_this_corner = False
        self.setup_ui()
        
        # Refresh the UI at ~60fps (16ms):
        # TODO: test adjusting this for 144+hz displays/using vsync:
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_overlay)
        self.timer.start(16)

    def setup_ui(self):
        # Make window transparent, frameless, and click-through:
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(100, 100, 400, 150) # x, y location + width, height
        
        self.layout = QVBoxLayout()
        self.label = QLabel("Waiting for telemetry...", self)
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self.label.setStyleSheet("color: white; background-color: rgba(0, 0, 0, 150); padding: 10px; border-radius: 10px;")
        
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

    def update_overlay(self):
        tel = self.listener.current_telemetry
        live_dist = tel.get("lap_distance", 0)
        live_speed = tel.get("speed", 0)
        
        # Interpolate expected speed if GT data is loaded:
        if len(self.listener.gt_distances) > 0 and live_dist > 0:
            expected_speed = np.interp(live_dist, self.listener.gt_distances, self.listener.gt_speeds)
            diff = live_speed - expected_speed
            
            color = "lime" if diff >= 0 else "red"
            text = f"Spd: {live_speed:.0f} km/h | Tgt: {expected_speed:.0f} km/h<br>Delta: <span style='color:{color}'>{diff:+.0f}</span>"
            self.label.setText(text)
        else:
            self.label.setText(f"Speed: {live_speed:.0f} km/h (No Target)")

        # Identify upcoming braking zone:
        upcoming_zone = None
        if hasattr(self.listener, 'ai_braking_zones'):
            for zone in self.listener.ai_braking_zones:
                if 0 < (zone["ai_brake_dist"] - live_dist) < 300:
                    upcoming_zone = zone
                    break

        if upcoming_zone:
            live_v_ms = float(live_speed) / 3.6
            ai_v_ms = upcoming_zone["ai_v_ms"]
            # taken from game_model.py:
            a = 44.1 
            p_stop_dist = (live_v_ms**2) / (2 * a)
            ai_stop_dist = (ai_v_ms**2) / (2 * a)
            optimal_brake_dist = upcoming_zone["ai_brake_dist"] - p_stop_dist + ai_stop_dist

            # Trigger audio cue at braking zone entry:
            if self.last_dist < (optimal_brake_dist - 5) and live_dist >= (optimal_brake_dist - 5):
                if not self.beeped_this_corner:
                    QApplication.beep() # system default "beep" - can be replaced later.
                    self.beeped_this_corner = True
                    
        # Reset corner beep latch for next lap:
        elif self.beeped_this_corner and tel.get("throttle") == 1.0:
            self.beeped_this_corner = False
        self.last_dist = live_dist

if __name__ == "__main__":
    app = QApplication(sys.argv)
    listener = UDPListener()
    listener.start()
    overlay = Overlay(listener)
    overlay.show()
    sys.exit(app.exec())