from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import numpy as np

class Overlay(QWidget):
    def __init__(self, listener):
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