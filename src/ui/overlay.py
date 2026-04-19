from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QGuiApplication, QPainter, QColor, QPen, QBrush
import numpy as np
import time
import keyboard # for hotkeys

class Overlay(QWidget):
    def __init__(self, listener):
        super().__init__()
        self.listener = listener
        # Tracking values for sound cues:
        self.last_dist = 0.0
        self.beeped_this_corner = False
        
        # Add UI tracking variables
        self.live_speed = 0
        self.expected_speed = 0
        self.diff = 0
        self.dist_to_brake = None

        # Mode settings:
        self.mode = "optimal" # default - compare against gt optimum
        self.mode_prefix = "Opt:"
        keyboard.add_hotkey("F9", self.toggle_mode)

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
        # Position overlay top-centre of screen:
        screen = QGuiApplication.primaryScreen().geometry()
        w, h = 600, 100
        self.setGeometry((screen.width() - w) // 2, 50, w, h)

    def toggle_mode(self):
        if self.mode == "optimal":
            if hasattr(self.listener, "pb_distances") and len(self.listener.pb_distances) > 0:
                self.mode = "pb"
                self.mode_prefix = "PB:"
                print("Delta mode switched to Personal Best.")
            else:
                print("Cannot switch mode: No laps recorded in this session.")

        else:
            self.mode = "optimal"
            self.mode_prefix = "Opt:"
            print("Delta mode switched to GT Optimum.")


    def update_overlay(self):
        tel = self.listener.current_telemetry
        screen = QGuiApplication.primaryScreen().geometry()
        w, h = 600, 100; center_x = (screen.width() - w) // 2
        # Get live centreline distance and speed; initialise dist_to_brake:
        live_dist = tel.get("cl_dist", 0) 
        self.live_speed = tel.get("speed", 0)
        self.dist_to_brake = None

        # Decide which reference arrays to use based on mode:
        if getattr(self, "mode", "optimal") == "pb" and hasattr(self.listener, "pb_distances"):
            ref_dists = self.listener.pb_distances
            ref_speeds = self.listener.pb_speeds
            ref_brake = self.listener.pb_brake
            ref_throttle = self.listener.pb_throttle
        else:
            ref_dists = self.listener.gt_distances
            ref_speeds = self.listener.gt_speeds
            ref_brake = getattr(self.listener, "gt_brake_exp", [])
            ref_throttle = getattr(self.listener, "gt_throttle_exp", [])
        
        # Interpolate all features:
        if len(ref_dists) > 0 and live_dist > 0:
            self.expected_speed = np.interp(live_dist, ref_dists, ref_speeds)
            self.diff = self.live_speed - self.expected_speed
            
            if len(ref_brake) > 0:
                self.exp_brake = np.interp(live_dist, ref_dists, ref_brake)
                self.exp_throttle = np.interp(live_dist, ref_dists, ref_throttle)
            else:
                self.exp_brake = self.exp_throttle = 0
        else:
            self.expected_speed = self.diff = self.exp_brake = self.exp_throttle = 0

        # Check for race control notices:
        last_popup_time = tel.get("last_ui_popup_time", 0)
        is_popup_active = (time.time() - last_popup_time) < 6.0 
        
        if is_popup_active:
            target_y = 150  # Shift down below alert
        else:
            target_y = 50   # Default position

        if self.y() != target_y:
            self.setGeometry(center_x, target_y, w, h)

        # Identify upcoming braking zone:
        upcoming_zone = None
        if hasattr(self.listener, 'ai_braking_zones'):
            for zone in self.listener.ai_braking_zones:
                if 0 < (zone["ai_brake_dist"] - live_dist) < 300:
                    upcoming_zone = zone
                    break

        if upcoming_zone:
            live_v_ms = float(self.live_speed) / 3.6
            ai_v_ms = upcoming_zone["ai_v_ms"]
            # taken from game_model.py:
            a = 44.1 
            p_stop_dist = (live_v_ms**2) / (2 * a)
            ai_stop_dist = (ai_v_ms**2) / (2 * a)
            optimal_brake_dist = upcoming_zone["ai_brake_dist"] - p_stop_dist + ai_stop_dist
            
            # Save distance to brake:
            self.dist_to_brake = optimal_brake_dist - live_dist

            if self.last_dist < (optimal_brake_dist - 5) and live_dist >= (optimal_brake_dist - 5):
                if not self.beeped_this_corner:
                    QApplication.beep() # system default "beep" - can be replaced later.
                    self.beeped_this_corner = True
                    
        # Reset corner beep latch for next lap:
        elif self.beeped_this_corner and tel.get("throttle") == 1.0:
            self.beeped_this_corner = False
            
        self.last_dist = live_dist
        # Trigger paintEvent - redraw UI with new values:
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Colour overlay based on expected car state:
        bg_alpha = 255
        if getattr(self, "exp_brake", 0) > 0.4:
            bg_color = QColor(255, 0, 0, bg_alpha) # Brake: Red
        elif getattr(self, "exp_throttle", 0) > 0.4:
            bg_color = QColor(0, 255, 0, bg_alpha) # Throttle: Green
        else:
            bg_color = QColor(100, 100, 100, bg_alpha) # Cornering: Grey
        painter.fillRect(self.rect(), bg_color)

        # Draw translucent background:
        painter.setBrush(QBrush(QColor(0, 0, 0, 180)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 15, 15)

        center_x = self.width() // 2
        bar_y = 55
        bar_height = 25
        max_bar_width = (self.width() - 40) // 2 # Max length left or right

        # Draw delta bar base:
        painter.setBrush(QBrush(QColor(50, 50, 50, 200)))
        painter.drawRoundedRect(20, bar_y, self.width() - 40, bar_height, 5, 5)

        # Draw active delta bar:
        if self.diff != 0:
            # Ensure bar is scaled (caps +- 20km/h):
            scale = max_bar_width / 20.0 
            bar_len = min(abs(self.diff) * scale, max_bar_width)
            
            painter.setPen(Qt.PenStyle.NoPen)
            if self.diff >= 0:
                # Faster than expected: green bar expanding right:
                painter.setBrush(QBrush(QColor(0, 255, 0, 220)))
                painter.drawRoundedRect(center_x, bar_y, int(bar_len), bar_height, 5, 5)
            else:
                # Slower than expected: red bar expanding left:
                painter.setBrush(QBrush(QColor(255, 0, 0, 220)))
                painter.drawRoundedRect(int(center_x - bar_len), bar_y, int(bar_len), bar_height, 5, 5)

        # Draw centre divider:
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawLine(center_x, bar_y - 5, center_x, bar_y + bar_height + 5)

        # Draw text:
        painter.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        text_y = 35
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(20, text_y, f"{self.live_speed:.0f} km/h")
        target_text = f"{getattr(self, 'mode_prefix', 'Opt:')} {self.expected_speed:.0f} km/h" if self.expected_speed > 0 else "No Target"
        painter.drawText(self.width() - 170, text_y, target_text)

        # Centre the delta text:
        delta_color = QColor(0, 255, 0) if self.diff >= 0 else QColor(255, 0, 0)
        painter.setPen(QPen(delta_color))
        painter.drawText(center_x - 15, text_y, f"{self.diff:+.0f}")

        # Draw braking point coutndown:
        if self.dist_to_brake is not None and 0 < self.dist_to_brake <= 150:
            painter.setPen(QPen(QColor(255, 165, 0))) # Orange "warning" colour?
            painter.drawText(center_x - 70, text_y - 20, f"BRAKE IN {self.dist_to_brake:.0f}m")