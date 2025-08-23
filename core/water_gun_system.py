#!/usr/bin/env python3
import cv2
import numpy as np
import time
import threading
import logging
from dataclasses import dataclass
from typing import Tuple, Optional
import queue

@dataclass
class TrackingConfig:
    frame_width: int = 640
    frame_height: int = 480
    center_x: int = 320
    center_y: int = 240
    deadzone_radius: int = 30
    
    # PID parameters
    kp_pan: float = 0.8
    ki_pan: float = 0.1
    kd_pan: float = 0.2
    kp_tilt: float = 0.8
    ki_tilt: float = 0.1
    kd_tilt: float = 0.2
    
    # Servo limits (degrees)
    pan_min: int = -90
    pan_max: int = 90
    tilt_min: int = -30
    tilt_max: int = 60

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()
    
    def update(self, error: float) -> float:
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 0.01
        
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        output = (self.kp * error + 
                 self.ki * self.integral + 
                 self.kd * derivative)
        
        self.prev_error = error
        self.last_time = current_time
        
        return output

class VideoDetector:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        
        # Backup MobileNet SSD detector (if available)
        self.use_dnn = False
        try:
            self.net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'config.pbtxt')
            self.use_dnn = True
            logging.info("MobileNet SSD loaded successfully")
        except:
            logging.info("Using Haar cascades for detection")
    
    def detect_targets(self, frame: np.ndarray) -> list:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        targets = []
        
        # Face detection
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            center_x = x + w // 2
            center_y = y + h // 2
            targets.append((center_x, center_y, w * h, 'face'))
        
        # Body detection (if no faces found)
        if not targets:
            bodies = self.body_cascade.detectMultiScale(gray, 1.1, 3)
            for (x, y, w, h) in bodies:
                center_x = x + w // 2
                center_y = y + h // 2
                targets.append((center_x, center_y, w * h, 'body'))
        
        # Sort by area (largest first)
        targets.sort(key=lambda x: x[2], reverse=True)
        return targets
    
    def draw_targets(self, frame: np.ndarray, targets: list) -> np.ndarray:
        for i, (x, y, area, target_type) in enumerate(targets):
            color = (0, 255, 0) if i == 0 else (0, 255, 255)
            cv2.circle(frame, (x, y), 10, color, 2)
            cv2.putText(frame, f"{target_type}", (x-30, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw center crosshair
        cv2.line(frame, (self.config.center_x - 20, self.config.center_y), 
                (self.config.center_x + 20, self.config.center_y), (255, 255, 255), 1)
        cv2.line(frame, (self.config.center_x, self.config.center_y - 20), 
                (self.config.center_x, self.config.center_y + 20), (255, 255, 255), 1)
        
        # Draw deadzone
        cv2.circle(frame, (self.config.center_x, self.config.center_y), 
                  self.config.deadzone_radius, (128, 128, 128), 1)
        
        return frame

class WaterGunController:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.detector = VideoDetector(config)
        
        # PID controllers
        self.pan_pid = PIDController(config.kp_pan, config.ki_pan, config.kd_pan)
        self.tilt_pid = PIDController(config.kp_tilt, config.ki_tilt, config.kd_tilt)
        
        # Current servo positions
        self.current_pan = 0
        self.current_tilt = 0
        
        # Control state
        self.tracking_enabled = False
        self.auto_fire_enabled = False
        self.last_target_time = time.time()
        self.target_lost_timeout = 2.0
        
        # Communication queues
        self.servo_command_queue = queue.Queue()
        self.device_command_queue = queue.Queue()
        self.status_queue = queue.Queue()
        
        # Camera
        self.camera = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_camera(self, camera_index: int = 0) -> bool:
        try:
            # Try OpenCV first
            self.camera = cv2.VideoCapture(camera_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            if self.camera.isOpened():
                # Test if we can actually read frames
                ret, test_frame = self.camera.read()
                if ret:
                    self.logger.info("Camera initialized successfully with OpenCV")
                    return True
                else:
                    self.camera.release()
            
            # Fallback to libcamera for Pi 5
            self.logger.info("OpenCV failed, trying libcamera for Pi 5...")
            from camera_test import LibcameraCapture
            self.camera = LibcameraCapture(self.config.frame_width, self.config.frame_height)
            
            # Test libcamera capture
            ret, test_frame = self.camera.read()
            if ret:
                self.logger.info("Camera initialized successfully with libcamera")
                return True
            else:
                self.logger.error("Failed to capture with libcamera")
                return False
                
        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        if not self.camera:
            return None
        
        # Handle both OpenCV VideoCapture and custom LibcameraCapture
        if hasattr(self.camera, 'isOpened'):
            # OpenCV VideoCapture
            if not self.camera.isOpened():
                return None
        
        ret, frame = self.camera.read()
        if not ret or frame is None:
            return None
        
        with self.frame_lock:
            self.latest_frame = frame.copy()
        
        return frame
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def calculate_servo_adjustments(self, target_x: int, target_y: int) -> Tuple[float, float]:
        # Calculate errors from center
        pan_error = target_x - self.config.center_x
        tilt_error = self.config.center_y - target_y  # Inverted for tilt
        
        # Check if target is in deadzone
        distance = np.sqrt(pan_error**2 + tilt_error**2)
        if distance <= self.config.deadzone_radius:
            return 0, 0
        
        # PID control
        pan_adjustment = self.pan_pid.update(pan_error)
        tilt_adjustment = self.tilt_pid.update(tilt_error)
        
        # Convert pixel error to servo angle adjustment
        pan_adjustment = np.clip(pan_adjustment * 0.1, -5, 5)
        tilt_adjustment = np.clip(tilt_adjustment * 0.1, -5, 5)
        
        return pan_adjustment, tilt_adjustment
    
    def update_servo_positions(self, pan_adj: float, tilt_adj: float):
        self.current_pan += pan_adj
        self.current_tilt += tilt_adj
        
        # Apply limits
        self.current_pan = np.clip(self.current_pan, self.config.pan_min, self.config.pan_max)
        self.current_tilt = np.clip(self.current_tilt, self.config.tilt_min, self.config.tilt_max)
        
        # Send command to servo controller
        servo_command = {
            'pan': self.current_pan,
            'tilt': self.current_tilt,
            'timestamp': time.time()
        }
        
        try:
            self.servo_command_queue.put_nowait(servo_command)
        except queue.Full:
            pass
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.tracking_enabled:
            return self.detector.draw_targets(frame, [])
        
        # Detect targets
        targets = self.detector.detect_targets(frame)
        
        if targets:
            # Track the largest target
            target_x, target_y, area, target_type = targets[0]
            self.last_target_time = time.time()
            
            # Calculate servo adjustments
            pan_adj, tilt_adj = self.calculate_servo_adjustments(target_x, target_y)
            
            # Update servo positions
            if abs(pan_adj) > 0.1 or abs(tilt_adj) > 0.1:
                self.update_servo_positions(pan_adj, tilt_adj)
            
            # Auto-fire logic
            if self.auto_fire_enabled and abs(pan_adj) < 1 and abs(tilt_adj) < 1:
                self.fire_water()
        
        else:
            # Target lost - implement search pattern or return to center
            if time.time() - self.last_target_time > self.target_lost_timeout:
                self.return_to_center()
        
        # Draw visualization
        frame = self.detector.draw_targets(frame, targets)
        
        # Add status overlay
        status_text = f"Tracking: {'ON' if self.tracking_enabled else 'OFF'} | "
        status_text += f"Auto-fire: {'ON' if self.auto_fire_enabled else 'OFF'} | "
        status_text += f"Pan: {self.current_pan:.1f}° | Tilt: {self.current_tilt:.1f}°"
        
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame
    
    def return_to_center(self):
        center_pan_adj = -self.current_pan * 0.1
        center_tilt_adj = -self.current_tilt * 0.1
        
        if abs(center_pan_adj) > 0.1 or abs(center_tilt_adj) > 0.1:
            self.update_servo_positions(center_pan_adj, center_tilt_adj)
    
    def fire_water(self, duration: float = 1.0):
        device_command = {
            'action': 'fire',
            'duration': duration,
            'timestamp': time.time()
        }
        
        try:
            self.device_command_queue.put_nowait(device_command)
            self.logger.info("Water fired!")
        except queue.Full:
            pass
    
    def set_tracking_enabled(self, enabled: bool):
        self.tracking_enabled = enabled
        if enabled:
            self.logger.info("Tracking enabled")
        else:
            self.logger.info("Tracking disabled")
    
    def set_auto_fire_enabled(self, enabled: bool):
        self.auto_fire_enabled = enabled
        self.logger.info(f"Auto-fire {'enabled' if enabled else 'disabled'}")
    
    def manual_servo_control(self, pan: float, tilt: float):
        self.current_pan = np.clip(pan, self.config.pan_min, self.config.pan_max)
        self.current_tilt = np.clip(tilt, self.config.tilt_min, self.config.tilt_max)
        
        servo_command = {
            'pan': self.current_pan,
            'tilt': self.current_tilt,
            'timestamp': time.time()
        }
        
        try:
            self.servo_command_queue.put_nowait(servo_command)
        except queue.Full:
            pass
    
    def cleanup(self):
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    config = TrackingConfig()
    controller = WaterGunController(config)
    
    if not controller.initialize_camera():
        print("Failed to initialize camera")
        exit(1)
    
    print("Water Gun System Started")
    print("Press 't' to toggle tracking")
    print("Press 'f' to toggle auto-fire")
    print("Press 'SPACE' to fire manually")
    print("Press 'r' to return to center")
    print("Press 'q' to quit")
    
    try:
        while True:
            frame = controller.capture_frame()
            if frame is None:
                continue
            
            processed_frame = controller.process_frame(frame)
            cv2.imshow("Water Gun System", processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                controller.set_tracking_enabled(not controller.tracking_enabled)
            elif key == ord('f'):
                controller.set_auto_fire_enabled(not controller.auto_fire_enabled)
            elif key == ord(' '):
                controller.fire_water()
            elif key == ord('r'):
                controller.return_to_center()
    
    finally:
        controller.cleanup()