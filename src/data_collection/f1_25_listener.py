import socket
import struct
from datetime import datetime
import pandas as pd
import time
import sys
import keyboard
from pathlib import Path
# imports for real-time/overlay:
import threading
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from scipy.spatial import cKDTree
# load model/advice/overlay files:
src_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(src_dir / "modelling")) 
sys.path.append(str(src_dir / "feedback")) 
sys.path.append(str(src_dir / "ui"))
sys.path.append(str(src_dir / "feedback"))
from game_model import RandomForestModel, Curvature, project_to_centreline, add_should_brake, add_should_throttle # type: ignore
from game_advice import build_references_from_gt, advice, write_advice # type: ignore
from track_plots import PlotTrackMaps # type: ignore
from overlay import Overlay, StatsOverlay, AdviceOverlay # type: ignore
from corner_info import get_corner_no # type: ignore
from session_plots import PlotSessionProgression # type: ignore 

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
        self.session_best_time = float("inf")
        
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
        self.overlay_enabled = True
        keyboard.add_hotkey("ctrl", self.toggle_overlay)
        # Initialise UDP socket:
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.bind((UDP_IP, UDP_PORT))
        print("Listening on " + UDP_IP + ":" + str(UDP_PORT))

        # Preload corner data:
        threading.Thread(target=self.preload_corners, daemon=True).start()

    def toggle_overlay(self):
        self.overlay_enabled = not self.overlay_enabled
        print(f"Overlay enabled: {self.overlay_enabled}")
        
    def load_ground_truth(self, track_name):
        gt_path = models_dir / f"{track_name}_ground_truth.csv"
        if gt_path.exists():
            self.gt_df = pd.read_csv(gt_path)
            self.gt_distances = self.gt_df["cl_dist"].values
            self.gt_speeds = self.gt_df["speed_exp"].values
            coords = self.gt_df[["x_exp", "y_exp"]].values
            self.gt_tree = cKDTree(coords)
            self.gt_cl_dists = self.gt_df["cl_dist"].values
            # brake and throttle states for overlay colouring:
            self.gt_brake_exp = self.gt_df["brake_exp"].values
            self.gt_throttle_exp = self.gt_df["throttle_exp"].values
            
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

    def new_session(self, session_time, sector):
        self.session_dir = output_dir / datetime.now().strftime("%Y_%m_%d_%H%M%S")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.lap_data = []
        self.current_lap = 0
        self.lap_start_time = session_time
        self.current_sector = sector
        self.session_best_time = float("inf")
        self.latest_advice = None
        print("New session started.")

    def run(self):
        while True:
            data, _ = self.udp.recvfrom(4096)
            
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
                # Map real-time position to the nearest ground truth centerline distance
                if hasattr(self, 'gt_tree'):
                    real_x = motion[2] # z_pos
                    real_y = motion[0] # x_pos
                    
                    # Find the index of the closest centerline coordinate:
                    _, nearest_idx = self.gt_tree.query([real_x, real_y])
                    # Get cl_dist, expected brake, and expected throttle from this index:
                    self.current_telemetry["cl_dist"] = float(self.gt_cl_dists[nearest_idx])
                    self.current_telemetry["exp_brake"] = float(self.gt_brake_exp[nearest_idx])
                    self.current_telemetry["exp_throttle"] = float(self.gt_throttle_exp[nearest_idx])
                    
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
                        self.new_session(session_time, sector)
                    
                    if lap_number < self.current_lap:
                        self.new_session(session_time, sector)
                    
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
                        self.save_lap_csv(filename, self.lap_data, exact_lap_time_ms=lap_time_ms)
                        self.lap_data = []
                        self.lap_start_time = session_time
            
                    self.current_lap = lap_number
                    self.current_sector = sector
                    self.recording = self.current_lap > 0

            # Event telemetry (Race Control):
            if packet_id == 3 and self.recording:
                if len(data) >= HEADER_SIZE + 4:
                    event_data = struct.unpack("<4s", data[HEADER_SIZE:HEADER_SIZE+4])
                    event_string_code = event_data[0].decode('utf-8', errors='ignore')
                    # List of events that usually trigger a top-center UI popup:
                    ui_popups = ["RCWN", "PENA", "DRSE", "DRSD", "CHQF"] 
                    if event_string_code in ui_popups:
                        self.current_telemetry["last_ui_popup_time"] = time.time()
                        print(f"Detected UI Popup: {event_string_code}")

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

    # Pre-load circuit corner data distances via FastF1 for all tracks:
    def preload_corners(self):
        print("Loading 2025 corner data (via FastF1)...")
        for _, track_name in TRACK_IDS.items():
            get_corner_no(track_name)
        print("All corner data loaded.")

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

        advice_df = advice(lap_df, ref_brake, ref_throttle, gt=gt, track_name=target_track)
        self.latest_advice = advice_df # save for overlay display
        advice_path = output_dir / f"{target_lap_id}_advice.txt"
        write_advice(advice_df, advice_path, track_name=target_track, lap_id=filename)
        print(f"Saved advice to {advice_path}.")

        # Generate plot comparisons and also save to advice path:
        PlotTrackMaps.plot_lap_comparison(user_df=player_lap, gt_df=gt, track_name=str(target_track), out_dir=output_dir)
        print(f"Saved lap comparison plots to {advice_path}.")

    def save_lap_csv(self, filename, data_points, exact_lap_time_ms=None):
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

        # Check to see if lap is a new PB; if so, save all telemetry features:
        if exact_lap_time_ms > 0:
            lap_time = exact_lap_time_ms / 1000.0
        else:
            lap_time = df["time"].max()
        if lap_time > 0 and lap_time < self.session_best_time:
            self.session_best_time = lap_time
            print(f"*** NEW SESSION PERSONAL BEST: {lap_time:.3f}s ***")
            # Save features with respect to centreline for accurate comparisons:
            df_pb = df_raw.sort_values("cl_dist")
            self.pb_distances = df_pb["cl_dist"].values
            self.pb_speeds = pd.to_numeric(df_pb["speed"], errors="coerce").fillna(0).values
            self.pb_brake = df_pb["brake"].astype(float).values
            self.pb_throttle = df_pb["throttle"].astype(float).values
            self.pb_times = df_pb["laptime"].astype(float).values 
        
        # Update session lap times plot:
        if self.current_lap > 0:
            self.session_lap_summary.append({
                "lap": self.current_lap,
                "time": lap_time,
                "overlay_active": self.overlay_enabled
            })
            
            PlotSessionProgression.plot_laps(
                self.session_lap_summary, 
                df["track"], 
                self.session_dir
            )

        # Save telemetry to CSV:
        df.to_csv(filename, index=False)
        print(f"Saved {filename} with {len(df)} points")
        # Get advice for this lap; will also be saved:
        self.get_advice(filename, df)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Initialise and start the background UDP thread:
    listener = UDPListener()
    listener.start()
    # Initialise the GUI on the main thread:
    overlay = Overlay(listener)
    stats_overlay = StatsOverlay(listener, overlay)
    advice_overlay = AdviceOverlay(listener)
    overlay.show(); stats_overlay.show(); advice_overlay.show()
    
    def check_overlay_visibility():
        if listener.overlay_enabled:
            if overlay.isHidden(): # Only call show() if currently hidden
                overlay.show()
                stats_overlay.show()
                advice_overlay.show()
        else:
            if not overlay.isHidden(): # Only call hide() if currently visible
                overlay.hide()
                stats_overlay.hide()
                advice_overlay.hide()
    # Check state every 30ms:
    visibility_timer = QTimer()
    visibility_timer.timeout.connect(check_overlay_visibility)
    visibility_timer.start(30)

    # Run:
    sys.exit(app.exec())