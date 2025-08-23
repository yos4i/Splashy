#!/usr/bin/env python3
"""
Tracker - Optimized High-Performance Face Tracking System
Balanced resolution and FPS with stable servo control
"""
import cv2
import numpy as np
import time
import sys
import os
import logging
import subprocess
import tempfile

# Import the exact working servo controller
sys.path.append('core')
from servo_controller import ServoController, ServoConfig

class OptimizedCapture:
    """High-performance camera capture with dynamic resolution scaling"""
    
    def __init__(self, width=2304, height=1296, quality=85):
        """Initialize with balanced resolution for better FPS"""
        self.width = width
        self.height = height 
        self.quality = quality
        self.temp_file = None
        
        # Performance optimizations for better FPS
        self.capture_interval = 0.03  # ~30+ FPS for balanced resolution
        self.last_capture_time = 0
        
        print(f"📹 Optimized Camera: {width}x{height} ({width*height//1000000:.1f}MP)")
        print(f"🚀 Target FPS: ~{1/self.capture_interval:.1f}")
        
        # Test camera connection
        self._test_camera()
    
    def _test_camera(self):
        """Test camera connection and capabilities"""
        try:
            result = subprocess.run(['libcamera-hello', '--list-cameras'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ Camera detected and ready")
            else:
                print("⚠️ Camera test warning - continuing anyway")
        except Exception as e:
            print(f"⚠️ Camera test failed: {e} - continuing anyway")
    
    def read(self):
        """Capture optimized high resolution frame"""
        current_time = time.time()
        
        # Minimal capture interval for better FPS
        time_since_last = current_time - self.last_capture_time
        if time_since_last < self.capture_interval:
            time.sleep(self.capture_interval - time_since_last)
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Optimized capture command for better FPS
            cmd = [
                'libcamera-still',
                '--output', temp_path,
                '--timeout', '50',  # Faster capture
                '--width', str(self.width),
                '--height', str(self.height),
                '--quality', str(self.quality),
                '--nopreview',
                '--immediate',
                '--encoding', 'jpg'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=1.5)
            
            if result.returncode == 0 and os.path.exists(temp_path):
                frame = cv2.imread(temp_path)
                os.unlink(temp_path)
                
                self.last_capture_time = time.time()
                
                if frame is not None:
                    return True, frame
                else:
                    print("❌ Failed to load captured image")
                    return False, None
            else:
                print(f"❌ Camera capture failed")
                return False, None
                
        except subprocess.TimeoutExpired:
            print("⚠️ Camera capture timeout")
            return False, None
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False, None
    
    def release(self):
        """Clean up camera resources"""
        print("🧹 Camera resources cleaned")

class StableFaceDetector:
    """Stable face detection optimized for consistent tracking"""
    
    def __init__(self):
        print("🔍 Loading stable face detection...")
        
        # Load Haar cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Optimized detection parameters for stability
        self.scale_factor = 1.1    # Good balance of accuracy and performance
        self.min_neighbors = 5     # Higher for stability (reduce false positives)
        self.min_face_size = (60, 60)   # Reasonable minimum
        self.max_face_size = (400, 400) # Reasonable maximum
        
        print("✅ Stable face detector ready")
    
    def detect_faces(self, frame):
        """Stable face detection with consistent results"""
        if frame is None:
            return []
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        # Detect frontal faces with stable parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size,
            maxSize=self.max_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to target format
        targets = []
        for i, (x, y, w, h) in enumerate(faces):
            center_x = x + w // 2
            center_y = y + h // 2
            area = w * h
            
            # Calculate stability score (larger faces are more stable)
            stability = min(1.0, area / 10000)
            
            targets.append({
                'center': (center_x, center_y),
                'bbox': (x, y, w, h),
                'area': area,
                'stability': stability,
                'id': i
            })
        
        # Sort by area (largest first) for consistent primary target
        targets.sort(key=lambda t: t['area'], reverse=True)
        
        return targets

class StableTracker:
    """High-performance tracker with stable servo control"""
    
    def __init__(self):
        print("\n🚀 INITIALIZING STABLE TRACKER")
        print("="*50)
        
        # Optimized resolution for better FPS
        self.width = 2304   # Good balance: high quality but better FPS
        self.height = 1296  # 56 FPS capable resolution
        self.total_pixels = self.width * self.height
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        
        print(f"📹 Optimized Resolution: {self.width}x{self.height}")
        print(f"🎯 Total Pixels: {self.total_pixels:,} ({self.total_pixels//1000000:.1f}MP)")
        print(f"📍 Center Point: ({self.center_x}, {self.center_y})")
        
        # Initialize optimized camera
        self.camera = OptimizedCapture(self.width, self.height)
        
        # Initialize stable face detector
        self.detector = StableFaceDetector()
        
        # Initialize servo controller
        print("🎛️ Starting stable servo controller...")
        logging.basicConfig(level=logging.INFO)
        self.servo_config = ServoConfig()
        self.servo = ServoController(self.servo_config)
        self.servo.start_control_loop()
        
        # STABLE tracking parameters
        self.tracking_enabled = False
        self.deadzone = 80  # Reasonable deadzone
        
        # STABLE sensitivity (prevents unnecessary movement)
        self.pan_sensitivity = 0.0015   # Balanced sensitivity
        self.tilt_sensitivity = 0.001   # More stable tilt
        
        # STABILITY features to prevent random movement
        self.last_target_center = None
        self.target_stability_threshold = 3  # Frames before trusting target
        self.stable_target_count = 0
        self.last_servo_move_time = 0
        self.min_servo_interval = 0.2  # Minimum 200ms between servo moves
        self.movement_threshold = 10   # Minimum pixel movement to trigger servo
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = time.time()
        self.servo_move_count = 0
        
        print("✅ STABLE TRACKER READY!")
        print(f"🎯 Deadzone: {self.deadzone}px")
        print(f"🛑 Movement threshold: {self.movement_threshold}px")
        print(f"⏱️ Min servo interval: {self.min_servo_interval}s")
        print("="*50)
    
    def should_move_servo(self, target_center):
        """Determine if servo should move (prevents unnecessary movement)"""
        if not target_center:
            return False, None, None
        
        current_time = time.time()
        
        # Check minimum time between moves
        if current_time - self.last_servo_move_time < self.min_servo_interval:
            return False, None, None
        
        target_x, target_y = target_center
        
        # Check if target moved significantly from last position
        if self.last_target_center:
            last_x, last_y = self.last_target_center
            movement_distance = np.sqrt((target_x - last_x)**2 + (target_y - last_y)**2)
            
            if movement_distance < self.movement_threshold:
                return False, None, None  # Target hasn't moved enough
        
        # Calculate errors from center
        error_x = target_x - self.center_x
        error_y = self.center_y - target_y
        distance = np.sqrt(error_x**2 + error_y**2)
        
        # Check if within deadzone
        if distance <= self.deadzone:
            return False, None, None
        
        # Calculate servo adjustments
        pan_adjustment = error_x * self.pan_sensitivity
        tilt_adjustment = error_y * self.tilt_sensitivity
        
        # Apply reasonable limits to prevent excessive movement
        pan_adjustment = max(-0.1, min(0.1, pan_adjustment))
        tilt_adjustment = max(-0.1, min(0.1, tilt_adjustment))
        
        # Get current positions
        current_pan = self.servo.current_pan
        current_tilt = self.servo.current_tilt
        
        # Calculate new positions
        new_pan = max(-1, min(1, current_pan + pan_adjustment))
        new_tilt = max(-1, min(1, current_tilt + tilt_adjustment))
        
        # Only move if change is significant
        pan_change = abs(new_pan - current_pan)
        tilt_change = abs(new_tilt - current_tilt)
        
        if pan_change < 0.005 and tilt_change < 0.005:
            return False, None, None  # Change too small
        
        print(f"📍 Target: ({target_x}, {target_y}) | Error: x={error_x:.0f}, y={error_y:.0f}")
        print(f"🎯 Servo: Pan {current_pan:.3f}→{new_pan:.3f} | Tilt {current_tilt:.3f}→{new_tilt:.3f}")
        
        return True, new_pan, new_tilt
    
    def draw_optimized_overlay(self, frame, targets):
        """Draw optimized overlay for better performance"""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Reasonable UI scaling
        ui_scale = max(1, w // 2000)
        font_scale = 0.6 * ui_scale
        thickness = max(1, ui_scale)
        
        # Draw center crosshair
        crosshair_size = 60
        crosshair_color = (0, 255, 255)
        cv2.line(display, (self.center_x - crosshair_size, self.center_y), 
                (self.center_x + crosshair_size, self.center_y), crosshair_color, 3)
        cv2.line(display, (self.center_x, self.center_y - crosshair_size), 
                (self.center_x, self.center_y + crosshair_size), crosshair_color, 3)
        
        # Draw center point
        cv2.circle(display, (self.center_x, self.center_y), 10, crosshair_color, -1)
        
        # Draw deadzone
        cv2.circle(display, (self.center_x, self.center_y), self.deadzone, (255, 255, 0), 2)
        
        # Draw faces
        for i, target in enumerate(targets[:3]):  # Limit to 3 faces for performance
            center = target['center']
            bbox = target['bbox']
            x, y, w, h = bbox
            area = target['area']
            
            # Color coding
            if i == 0:  # Primary target
                color = (0, 255, 0)
                thickness_face = 3
                label = "PRIMARY"
            else:
                color = (0, 255, 255)
                thickness_face = 2
                label = f"FACE {i+1}"
            
            # Face bounding box
            cv2.rectangle(display, (x, y), (x + w, y + h), color, thickness_face)
            
            # Face center
            cv2.circle(display, center, 8, color, -1)
            
            # Face info
            cv2.putText(display, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            cv2.putText(display, f"{w}x{h}", (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, color, 1)
            
            # Tracking line for primary target
            if i == 0 and self.tracking_enabled:
                cv2.line(display, center, (self.center_x, self.center_y), color, 3)
                
                distance = np.sqrt((center[0] - self.center_x)**2 + (center[1] - self.center_y)**2)
                status = "CENTERED" if distance <= self.deadzone else "TRACKING"
                
                cv2.putText(display, status, (center[0] + 30, center[1] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        # Compact status panel
        self.draw_compact_status(display, targets)
        
        return display
    
    def draw_compact_status(self, frame, targets):
        """Draw compact status panel"""
        h, w = frame.shape[:2]
        
        # Compact status panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Title
        cv2.putText(frame, "🎯 STABLE TRACKER", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Status
        if self.tracking_enabled:
            status_text = "🔴 TRACKING ACTIVE"
            status_color = (0, 255, 0)
        else:
            status_text = "⚪ STANDBY"
            status_color = (128, 128, 128)
        
        cv2.putText(frame, status_text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Performance info
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f} | Faces: {len(targets)} | Moves: {self.servo_move_count}", 
                   (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(frame, "T=Track | C=Center | Q=Quit", (20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def run(self):
        """Main optimized tracking loop"""
        print("\n" + "="*60)
        print("🚀 STABLE TRACKER - OPTIMIZED PERFORMANCE")
        print("="*60)
        print(f"📹 Resolution: {self.width}x{self.height} (optimized for FPS)")
        print("🎯 Stable servo control with movement filtering")
        print("🚀 Enhanced FPS with balanced quality")
        print("")
        print("🎮 CONTROLS:")
        print("   T - Toggle tracking ON/OFF")
        print("   C - Center servos")
        print("   Q - Quit")
        print("")
        print("▶️ Optimized for smooth, stable performance!")
        print("-"*60)
        
        # Center servos
        print("🎯 Centering servos...")
        self.servo.move_to_center()
        time.sleep(1.5)
        
        try:
            while True:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ Capture failed - retrying...")
                    time.sleep(0.05)
                    continue
                
                self.frame_count += 1
                
                # Detect faces
                targets = self.detector.detect_faces(frame)
                
                # Stable tracking with movement filtering
                if self.tracking_enabled and targets:
                    primary_target = targets[0]
                    
                    # Check if we should move servo
                    should_move, new_pan, new_tilt = self.should_move_servo(primary_target['center'])
                    
                    if should_move:
                        self.servo.set_position(new_pan, new_tilt)
                        self.last_servo_move_time = time.time()
                        self.servo_move_count += 1
                        self.last_target_center = primary_target['center']
                        print("🎯 STABLE SERVO MOVE")
                    
                elif self.tracking_enabled and not targets:
                    # Reset tracking state when no targets
                    self.last_target_center = None
                    self.stable_target_count = 0
                
                # Create display
                display_frame = self.draw_optimized_overlay(frame, targets)
                
                # Show video
                cv2.imshow("Stable Tracker - Optimized Performance", display_frame)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n⏹️ QUIT")
                    break
                elif key == ord('t') or key == ord('T'):
                    self.tracking_enabled = not self.tracking_enabled
                    status = "ENABLED" if self.tracking_enabled else "DISABLED"
                    print(f"\n🎯 TRACKING {status}")
                    # Reset tracking state
                    self.last_target_center = None
                    self.stable_target_count = 0
                elif key == ord('c') or key == ord('C'):
                    print("\n🎯 CENTERING")
                    self.servo.move_to_center()
                    self.last_target_center = None
                    self.last_servo_move_time = time.time()
                
                # Performance report every 120 frames
                if self.frame_count % 120 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    print(f"📊 Frame {self.frame_count} | FPS: {fps:.1f} | Faces: {len(targets)} | Servo moves: {self.servo_move_count}")
                
        except KeyboardInterrupt:
            print("\n⏹️ INTERRUPTED")
        
        finally:
            print("\n🛑 SHUTDOWN...")
            
            if self.tracking_enabled:
                self.tracking_enabled = False
            
            print("🎯 Centering servos...")
            self.servo.move_to_center()
            time.sleep(1.5)
            
            print("🧹 Cleanup...")
            self.servo.cleanup()
            self.camera.release()
            cv2.destroyAllWindows()
            
            # Final report
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                fps = self.frame_count / elapsed
                moves_per_minute = (self.servo_move_count / elapsed) * 60
                print(f"\n📊 SESSION COMPLETE:")
                print(f"   Duration: {elapsed:.1f}s")
                print(f"   Frames: {self.frame_count}")
                print(f"   Average FPS: {fps:.1f}")
                print(f"   Servo moves: {self.servo_move_count}")
                print(f"   Moves per minute: {moves_per_minute:.1f}")
            
            print("\n✅ STABLE TRACKER COMPLETE!")

def main():
    print("🚀 Initializing Stable Tracker...")
    tracker = StableTracker()
    tracker.run()

if __name__ == "__main__":
    main()