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

        # GT brake/throttle truths should be used for both modes:
        # Decide which reference arrays to use based on mode:
        if getattr(self, "mode", "optimal") == "pb" and hasattr(self.listener, "pb_distances"):
            ref_dists = self.listener.pb_distances
            ref_speeds = self.listener.pb_speeds
            ref_brake = self.listener.pb_brake
            ref_throttle = self.listener.pb_throttle
            if hasattr(self.listener, "pb_times") and live_dist > 0:
                live_time = tel.get("laptime", 0)
                expected_pb_time = np.interp(live_dist, ref_dists, self.listener.pb_times)
                self.time_delta = live_time - expected_pb_time
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

            zone_id = upcoming_zone["ai_brake_dist"]
            if getattr(self, "current_zone_id", None) != zone_id:
                self.current_zone_id = zone_id
                self.beeped_this_corner = False

            if self.dist_to_brake is not None and 0 < self.dist_to_brake <= 15:
                if not getattr(self, "beeped_this_corner", False):
                    QApplication.beep()
                    self.beeped_this_corner = True
            
        self.last_dist = live_dist
        # Trigger paintEvent - redraw UI with new values:
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Colour overlay based on expected car state:
        bg_alpha = 255
        if getattr(self, "exp_brake", 0) > 0.4:
            bg_colour = QColor(255, 0, 0, bg_alpha) # Brake: Red
        elif getattr(self, "exp_throttle", 0) > 0.4:
            bg_colour = QColor(0, 255, 0, bg_alpha) # Throttle: Green
        else:
            bg_colour = QColor(100, 100, 100, bg_alpha) # Cornering: Grey
        painter.setBrush(QBrush(bg_colour))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 15, 15)

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
        delta_colour = QColor(0, 255, 0) if self.diff >= 0 else QColor(255, 0, 0)
        painter.setPen(QPen(delta_colour))
        painter.drawText(center_x - 15, text_y, f"{self.diff:+.0f}")

        # Draw braking point coutndown:
        if self.dist_to_brake is not None and 0 < self.dist_to_brake <= 150:
            painter.setPen(QPen(QColor(255, 165, 0))) # Orange "warning" colour?
            painter.drawText(center_x - 70, text_y - 20, f"BRAKE IN {self.dist_to_brake:.0f}m")

class StatsOverlay(QWidget):
    def __init__(self, listener, main_overlay):
        super().__init__()
        self.listener = listener
        self.drag_position = None # for repositioning
        self.main_overlay = main_overlay # Keep reference to know the current mode
        self.setup_ui()
        # Timer settings:
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(33) # ~30fps, fine for the stats overlay

    def setup_ui(self):
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.FramelessWindowHint 
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        screen = QGuiApplication.primaryScreen().geometry()
        w, h = 250, 200
        # RHS - vertically aligned to centre:
        self.setGeometry(screen.width() - w - 20, (screen.height() - h) // 2, w, h)

    # FOR MOVING SESSION STATS BOX WITH MOUSE:
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Calculate the offset from the top-left of the widget to where we clicked
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self.drag_position is not None:
            # Move the window tracking the mouse position minus the original offset
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.drag_position = None
        event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Translucent dark background:
        painter.setBrush(QBrush(QColor(0, 0, 0, 200)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 10, 10)

        pb_time = getattr(self.listener, "session_best_time", float("inf"))
        laps = self.listener.current_lap
        time_diff = getattr(self.main_overlay, "time_delta", 0.0)

        # M:SS.ms formatter
        def format_time(t):
            m = int(t // 60)
            s = t % 60
            return f"{m}:{s:06.3f}"
        pb_str = format_time(pb_time) if pb_time != float("inf") else "No PB yet"
        
        # Define the base rows (always visible):
        stats_rows = [
            ("Target Mode:", self.main_overlay.mode.upper(), QColor(255, 215, 0)),
            ("Laps Driven:", str(laps), QColor(255, 255, 255)),
            ("Session PB:", pb_str, QColor(0, 255, 0) if pb_time != float("inf") else QColor(150, 150, 150)),
        ]

        # Add time delta to state if in PB mode:
        if self.main_overlay.mode == "pb":
            # Colour coding for time diff:
            if time_diff < -0.05:
                delta_str = f"{time_diff:+.3f}s"
                delta_colour = QColor(50, 255, 50)
            elif time_diff > 0.05:
                delta_str = f"{time_diff:+.3f}s"
                delta_colour = QColor(255, 50, 50)
            else:
                delta_str = f"{time_diff:+.3f}s"
                delta_colour = QColor(255, 255, 255)
            stats_rows.append(("Time Delta:", delta_str, delta_colour))

        # Render header:
        y_pos = 30
        painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(15, y_pos, "SESSION STATS")
        painter.drawLine(15, y_pos + 5, self.width() - 15, y_pos + 5)
        
        # Render all rows:
        y_pos += 35
        painter.setFont(QFont("Arial", 10, QFont.Weight.Normal))
        for label, value, colour in stats_rows:
            painter.setPen(QPen(QColor(200, 200, 200))) 
            painter.drawText(15, y_pos, label)
            painter.setPen(QPen(colour))                 
            painter.drawText(120, y_pos, value)
            y_pos += 35
            
        # F9 tooltip at bottom of stats box:
        painter.setFont(QFont("Arial", 8, italic=True))
        painter.setPen(QPen(QColor(150, 150, 150)))
        painter.drawText(15, self.height() - 15, "Press F9 to toggle PB/Optimal comparison.")

class AdviceOverlay(QWidget):
    def __init__(self, listener):
        super().__init__()
        self.listener = listener
        self.drag_position = None
        self.setup_ui()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000) # once a second

    def setup_ui(self):
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.FramelessWindowHint 
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        screen = QGuiApplication.primaryScreen().geometry()
        w, h = 250, 250
        # Default pos: under stats box
        stats_h = 200
        start_y = ((screen.height() - stats_h) // 2) + stats_h + 5
        self.setGeometry(screen.width() - w - 20, start_y, w, h)

    # MOUSE EVENTS FOR MOVING ADVICE BOX:
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self.drag_position is not None:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.drag_position = None
        event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setBrush(QBrush(QColor(0, 0, 0, 200)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 10, 10)

        advice_df = getattr(self.listener, "latest_advice", None)
        
        y_pos = 30
        painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(255, 215, 0)))
        painter.drawText(15, y_pos, "TOP 3 SUGGESTIONS")
        painter.drawLine(15, y_pos + 5, self.width() - 15, y_pos + 5)
        
        y_pos += 25
        
        if advice_df is None or advice_df.empty:
            painter.setFont(QFont("Arial", 10, italic=True))
            painter.setPen(QPen(QColor(150, 150, 150)))
            painter.drawText(15, y_pos, "Finish a lap to generate advice...")
            return

        painter.setFont(QFont("Arial", 9))
        for i, row in advice_df.head(3).iterrows():
            painter.setPen(QPen(QColor(255, 100, 100))) 
            painter.drawText(15, y_pos, f"Braking Zone {row['corner_id']} (Lost {row['time_lost_s']:.2f}s)")
            
            painter.setPen(QPen(QColor(220, 220, 220))) 
            for line in row["advice"].split('\n'):
                y_pos += 16
                painter.drawText(20, y_pos, line.strip())
            
            y_pos += 20 # Spacing before next corner