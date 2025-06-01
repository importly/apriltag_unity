#!/usr/bin/env python3
"""
Optimized RealSense Camera Heading Tracker - High Accuracy, Minimal Code
Focuses on core accuracy improvements without over-engineering
"""

import pyrealsense2 as rs
import numpy as np
import time
import math
from threading import Lock, Thread
import tkinter as tk
from collections import deque
import statistics


class OptimizedRealSenseHeadingTracker:
    def __init__(self):
        self.pipeline = None
        self.heading = 0.0
        self.last_timestamp = None
        self.lock = Lock()

        # Simplified calibration
        self.gyro_bias_y = 0.0
        self.is_calibrated = False
        self.running = False

        # Minimal filtering - only what's necessary
        self.gyro_samples = deque(maxlen=3)  # Very short buffer for outlier rejection only

        # Quality tracking
        self.confidence = 1.0
        self.angular_velocity = 0.0
        self.total_rotations = 0.0

    def initialize_camera(self):
        """Initialize with highest available precision"""
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()

            # Try highest rate first, fallback if needed
            for gyro_rate in [400, 200, 100]:
                try:
                    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, gyro_rate)
                    self.pipeline.start(config)
                    print(f"Initialized at {gyro_rate}Hz")
                    return True
                except:
                    self.pipeline = rs.pipeline()
                    config = rs.config()
                    continue
            return False
        except Exception as e:
            print(f"Init error: {e}")
            return False

    def calibrate(self):
        """Fast, robust calibration"""
        print("Calibrating (5 seconds, keep still)...")
        samples = []
        start_time = time.time()

        while time.time() - start_time < 5.0:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=50) # type: ignore
                for frame in frames:
                    if frame.get_profile().stream_type() == rs.stream.gyro: # type: ignore
                        gyro_data = frame.as_motion_frame().get_motion_data()
                        samples.append(gyro_data.y)
            except:
                continue

        if samples:
            # Use median for outlier rejection
            self.gyro_bias_y = statistics.median(samples)
            noise_level = statistics.stdev(samples) if len(samples) > 1 else 0
            print(f"Bias: {self.gyro_bias_y:.6f}, Noise: {noise_level:.6f}")
            self.is_calibrated = True
            return True
        return False

    def update_heading(self):
        """Core heading update - optimized for accuracy"""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=50)
            gyro_frame = None

            for frame in frames:
                if frame.get_profile().stream_type() == rs.stream.gyro: 
                    gyro_frame = frame
                    break

            if not gyro_frame:
                return False

            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            current_timestamp = gyro_frame.get_timestamp()

            if self.last_timestamp is None:
                self.last_timestamp = current_timestamp
                return True

            dt = (current_timestamp - self.last_timestamp) / 1000.0
            if dt <= 0 or dt > 0.1:  # Skip bad timestamps
                self.last_timestamp = current_timestamp
                return True

            # Get calibrated gyro reading
            gyro_y = gyro_data.y - self.gyro_bias_y

            # Simple outlier rejection with minimal lag
            self.gyro_samples.append(gyro_y)
            if len(self.gyro_samples) >= 3:
                sorted_samples = sorted(self.gyro_samples)
                gyro_y_filtered = sorted_samples[1]  # Use median of last 3
            else:
                gyro_y_filtered = gyro_y

            # Dead zone for noise (smaller than before)
            if abs(gyro_y_filtered) < 0.0005:
                gyro_y_filtered = 0.0

            with self.lock:
                # Direct integration - no complex weighting
                heading_change_deg = math.degrees(gyro_y_filtered * dt)
                self.heading = (self.heading + heading_change_deg) % 360.0

                # Update metrics
                self.angular_velocity = abs(gyro_y_filtered)
                self.total_rotations += abs(heading_change_deg) / 360.0

                # Simple confidence based on consistency
                if len(self.gyro_samples) >= 3:
                    sample_range = max(self.gyro_samples) - min(self.gyro_samples)
                    self.confidence = max(0.1, 1.0 - sample_range * 100)
                else:
                    self.confidence = 0.5

            self.last_timestamp = current_timestamp
            return True

        except:
            return False

    def get_heading(self):
        with self.lock:
            return self.heading

    def get_metrics(self):
        with self.lock:
            return {
                'confidence': self.confidence,
                'angular_velocity': self.angular_velocity,
                'total_rotations': self.total_rotations
            }

    def reset_heading(self):
        with self.lock:
            self.heading = 0.0
            self.total_rotations = 0.0
            self.gyro_samples.clear()

    def stop(self):
        self.running = False
        if self.pipeline:
            self.pipeline.stop()


class OptimizedHeadingGUI:
    def __init__(self, tracker):
        self.tracker = tracker
        self.root = tk.Tk()
        self.root.title("Optimized RealSense Heading Tracker")
        self.root.geometry("400x600")
        self.root.configure(bg='black')

        # Compass canvas
        self.canvas = tk.Canvas(self.root, width=320, height=320, bg='black', highlightthickness=0)
        self.canvas.pack(pady=10)

        # Info displays
        self.heading_label = tk.Label(self.root, text="Heading: 0.0°",
                                      font=("Arial", 20, "bold"), fg='white', bg='black')
        self.heading_label.pack(pady=5)

        self.confidence_label = tk.Label(self.root, text="Confidence: --",
                                         font=("Arial", 12), fg='gray', bg='black')
        self.confidence_label.pack()

        self.velocity_label = tk.Label(self.root, text="Angular Velocity: --",
                                       font=("Arial", 12), fg='gray', bg='black')
        self.velocity_label.pack()

        self.rotations_label = tk.Label(self.root, text="Total Rotations: 0.0",
                                        font=("Arial", 12), fg='gray', bg='black')
        self.rotations_label.pack()

        # Controls
        button_frame = tk.Frame(self.root, bg='black')
        button_frame.pack(pady=15)

        tk.Button(button_frame, text="Reset", command=self.tracker.reset_heading,
                  font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Quit", command=self.quit_app,
                  font=("Arial", 12)).pack(side=tk.LEFT, padx=5)

        self.center_x = self.center_y = 160
        self.compass_radius = 130

        self.update_display()

    def draw_compass(self, heading, confidence):
        """Draw compass with heading arrow"""
        self.canvas.delete("all")

        # Outer circle
        self.canvas.create_oval(30, 30, 290, 290, outline='white', width=2)

        # Cardinal directions and degree marks
        for angle in range(0, 360, 30):
            rad = math.radians(angle - 90)

            if angle % 90 == 0:
                # Cardinal points
                color = 'red' if angle == 0 else 'white'
                outer_r, inner_r, width = 115, 95, 2
                label = ['N', 'E', 'S', 'W'][angle // 90]
                text_r = 105
                text_x = self.center_x + text_r * math.cos(rad)
                text_y = self.center_y + text_r * math.sin(rad)
                self.canvas.create_text(text_x, text_y, text=label, fill=color, font=("Arial", 12, "bold"))
            else:
                # 30-degree marks
                color, outer_r, inner_r, width = 'gray', 115, 105, 1
                text_x = self.center_x + 100 * math.cos(rad)
                text_y = self.center_y + 100 * math.sin(rad)
                self.canvas.create_text(text_x, text_y, text=str(angle), fill='lightgray', font=("Arial", 8))

            # Draw tick
            outer_x = self.center_x + outer_r * math.cos(rad)
            outer_y = self.center_y + outer_r * math.sin(rad)
            inner_x = self.center_x + inner_r * math.cos(rad)
            inner_y = self.center_y + inner_r * math.sin(rad)
            self.canvas.create_line(outer_x, outer_y, inner_x, inner_y, fill=color, width=width)

        # Heading arrow
        arrow_color = 'lime' if confidence > 0.7 else 'yellow' if confidence > 0.4 else 'orange'
        rad = math.radians(heading - 90)

        # Arrow line
        arrow_length = 80
        end_x = self.center_x + arrow_length * math.cos(rad)
        end_y = self.center_y + arrow_length * math.sin(rad)
        self.canvas.create_line(self.center_x, self.center_y, end_x, end_y,
                                fill=arrow_color, width=4)

        # Arrowhead
        head_len = 15
        head_angle = math.radians(25)
        left_x = end_x - head_len * math.cos(rad - head_angle)
        left_y = end_y - head_len * math.sin(rad - head_angle)
        right_x = end_x - head_len * math.cos(rad + head_angle)
        right_y = end_y - head_len * math.sin(rad + head_angle)
        self.canvas.create_polygon(end_x, end_y, left_x, left_y, right_x, right_y,
                                   fill=arrow_color, outline=arrow_color)

        # Center dot
        self.canvas.create_oval(155, 155, 165, 165, fill='white')

    def update_display(self):
        """Update display at high frequency"""
        if self.tracker.running and self.tracker.is_calibrated:
            heading = self.tracker.get_heading()
            metrics = self.tracker.get_metrics()

            # Update labels
            self.heading_label.config(text=f"Heading: {heading:.1f}°")

            confidence = metrics['confidence']
            self.confidence_label.config(text=f"Confidence: {confidence:.1%}",
                                         fg='lime' if confidence > 0.7 else 'yellow' if confidence > 0.4 else 'orange')

            angular_vel_deg = math.degrees(metrics['angular_velocity'])
            self.velocity_label.config(text=f"Angular Velocity: {angular_vel_deg:.1f}°/s")

            self.rotations_label.config(text=f"Total Rotations: {metrics['total_rotations']:.1f}")

            # Draw compass
            self.draw_compass(heading, confidence)

        self.root.after(20, self.update_display)  # 50Hz for smooth display

    def quit_app(self):
        self.tracker.stop()
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    print("Optimized RealSense Heading Tracker")
    print("Key optimizations: Minimal filtering, direct integration, fast calibration")

    tracker = OptimizedRealSenseHeadingTracker()

    if not tracker.initialize_camera():
        print("Failed to initialize camera")
        return

    if not tracker.calibrate():
        print("Calibration failed")
        return

    tracker.running = True

    # Background tracking thread
    def track():
        while tracker.running:
            tracker.update_heading()
            time.sleep(0.002)  # 500Hz for maximum precision

    Thread(target=track, daemon=True).start()


if __name__ == "__main__":
    main()