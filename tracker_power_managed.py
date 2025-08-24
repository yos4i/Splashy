#!/usr/bin/env python3
"""
Power-Managed Tracker - Stable face tracking with automatic servo power management
Servos automatically turn off when not needed to save power and reduce wear
"""
import cv2
import numpy as np
import time
import sys
import os
import logging
import subprocess
import tempfile

# Import the power-managed servo controller
sys.path.append('.')
from power_managed_servo import PowerManagedServoController, PowerManagedServoConfig

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

class PowerManagedTracker:
    """High-performance tracker with power-managed servo control"""
    
    def __init__(self):
        print("\n🔋 INITIALIZING POWER-MANAGED TRACKER")
        print("="*60)
        
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
        
        # Initialize power-managed servo controller
        print("🔋 Starting power-managed servo controller...")
        logging.basicConfig(level=logging.INFO)
        self.servo_config = PowerManagedServoConfig(
            auto_disable_delay=5.0,  # Turn off servos after 5 seconds idle
            movement_settle_time=1.0  # Wait 1 second after movement
        )
        self.servo = PowerManagedServoController(self.servo_config)
        
        # ULTRA-STRICT tracking parameters
        self.tracking_enabled = False
        self.deadzone = 80  # Reasonable deadzone
        
        # ULTRA-STRICT sensitivity (prevents unnecessary movement)
        self.pan_sensitivity = 0.0015   # Balanced sensitivity
        self.tilt_sensitivity = 0.001   # More stable tilt
        
        # ULTRA-STRICT stability features
        self.last_target_center = None
        self.target_stability_threshold = 8
        self.stable_target_count = 0
        self.consistent_targets = []
        self.last_servo_move_time = 0
        self.min_servo_interval = 0.5  # 500ms between moves
        self.movement_threshold = 25   # 25px minimum movement
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = time.time()
        self.servo_move_count = 0
        
        print("✅ POWER-MANAGED TRACKER READY!")
        print(f"🎯 Deadzone: {self.deadzone}px")
        print(f"🔋 Auto servo disable: {self.servo_config.auto_disable_delay}s")
        print(f"🛑 Movement threshold: {self.movement_threshold}px")
        print(f"⏱️ Min servo interval: {self.min_servo_interval}s")
        print("="*60)
    
    def validate_target_stability(self, target_center):
        """Ultra-strict target validation to prevent false positives"""
        if not target_center:
            self.stable_target_count = 0
            self.consistent_targets.clear()
            return False
        
        target_x, target_y = target_center
        
        # Add current target to recent history
        self.consistent_targets.append((target_x, target_y))
        
        # Keep only recent targets (last 10 frames)
        if len(self.consistent_targets) > 10:
            self.consistent_targets.pop(0)
        
        # Need minimum number of consistent targets
        if len(self.consistent_targets) < self.target_stability_threshold:
            return False
        
        # Check if recent targets are consistent (not jumping around)
        recent_targets = self.consistent_targets[-self.target_stability_threshold:]
        
        # Calculate variance in target positions
        x_positions = [t[0] for t in recent_targets]
        y_positions = [t[1] for t in recent_targets]
        
        x_variance = np.var(x_positions)
        y_variance = np.var(y_positions)
        
        # If targets are jumping around too much, don't trust them
        if x_variance > 100 or y_variance > 100:
            print(f"🚫 Target unstable: x_var={x_variance:.1f}, y_var={y_variance:.1f}")
            return False
        
        return True

    def should_move_servo(self, target_center):
        """ULTRA-STRICT servo movement decision with multiple safety checks"""
        if not target_center:
            return False, None, None
        
        current_time = time.time()
        
        # STRICT: Check minimum time between moves
        if current_time - self.last_servo_move_time < self.min_servo_interval:
            return False, None, None
        
        target_x, target_y = target_center
        
        # STRICT: Target must be stable for multiple frames
        if not self.validate_target_stability(target_center):
            return False, None, None
        
        # STRICT: Check if target moved significantly from last position
        if self.last_target_center:
            last_x, last_y = self.last_target_center
            movement_distance = np.sqrt((target_x - last_x)**2 + (target_y - last_y)**2)
            
            if movement_distance < self.movement_threshold:
                return False, None, None
        
        # Calculate errors from center
        error_x = target_x - self.center_x
        error_y = self.center_y - target_y
        distance = np.sqrt(error_x**2 + error_y**2)
        
        # STRICT: Check if within deadzone
        if distance <= self.deadzone:
            return False, None, None
        
        # STRICT: Only move if error is significant
        if distance < self.deadzone * 1.5:
            return False, None, None
        
        # Calculate servo adjustments
        pan_adjustment = error_x * self.pan_sensitivity
        tilt_adjustment = error_y * self.tilt_sensitivity
        
        # STRICT: Apply conservative limits
        max_adjustment = 0.05
        pan_adjustment = max(-max_adjustment, min(max_adjustment, pan_adjustment))
        tilt_adjustment = max(-max_adjustment, min(max_adjustment, tilt_adjustment))
        
        # Get current positions
        current_pan = self.servo.current_pan
        current_tilt = self.servo.current_tilt
        
        # Calculate new positions
        new_pan = max(-1, min(1, current_pan + pan_adjustment))
        new_tilt = max(-1, min(1, current_tilt + tilt_adjustment))
        
        # STRICT: Only move if change is significant
        pan_change = abs(new_pan - current_pan)
        tilt_change = abs(new_tilt - current_tilt)
        
        min_change = 0.01
        if pan_change < min_change and tilt_change < min_change:
            return False, None, None
        
        # Show why we're moving
        print(f"🔍 SERVO MOVE: Target({target_x}, {target_y}) | Dist: {distance:.0f}px | Pan: {current_pan:.3f}→{new_pan:.3f} | Tilt: {current_tilt:.3f}→{new_tilt:.3f}")
        
        return True, new_pan, new_tilt
    
    def draw_status_overlay(self, frame, targets):
        """Draw status overlay with power management info"""
        display = frame.copy()
        h, w = display.shape[:2]
        
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
        for i, target in enumerate(targets[:3]):
            center = target['center']
            bbox = target['bbox']
            x, y, w, h = bbox
            
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
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Tracking line for primary target
            if i == 0 and self.tracking_enabled:
                cv2.line(display, center, (self.center_x, self.center_y), color, 3)
                
                distance = np.sqrt((center[0] - self.center_x)**2 + (center[1] - self.center_y)**2)
                status = "CENTERED" if distance <= self.deadzone else "TRACKING"
                
                cv2.putText(display, status, (center[0] + 30, center[1] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Status panel with power info
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Title
        cv2.putText(frame, "🔋 POWER-MANAGED TRACKER", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Tracking status
        if self.tracking_enabled:
            status_text = "🔴 TRACKING ACTIVE"
            status_color = (0, 255, 0)
        else:
            status_text = "⚪ STANDBY"
            status_color = (128, 128, 128)
        
        cv2.putText(frame, status_text, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Power status
        servo_power = "ON" if self.servo.is_powered() else "OFF"
        power_color = (0, 255, 0) if self.servo.is_powered() else (128, 128, 128)
        
        # Performance info
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f} | Faces: {len(targets)} | Moves: {self.servo_move_count}", 
                   (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"🔋 Servo Power: {servo_power}", 
                   (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, power_color, 1)
        
        # Controls
        cv2.putText(frame, "T=Track | P=Power On/Off | C=Center | Q=Quit", (20, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return display
    
    def run(self):
        """Main power-managed tracking loop"""
        print("\n" + "="*70)
        print("🔋 POWER-MANAGED TRACKER - ENERGY EFFICIENT OPERATION")
        print("="*70)
        print(f"📹 Resolution: {self.width}x{self.height} (optimized for FPS)")
        print("🔋 Servos auto-disable when idle to save power")
        print("🎯 Ultra-strict movement filtering prevents jitter")
        print("")
        print("🎮 CONTROLS:")
        print("   T - Toggle tracking ON/OFF")
        print("   P - Manual servo power ON/OFF")
        print("   C - Center servos")  
        print("   Q - Quit")
        print("")
        print("▶️ Servos will auto-power off when idle!")
        print("-"*70)
        
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
                
                # Power-managed tracking
                if self.tracking_enabled and targets:
                    primary_target = targets[0]
                    
                    # Check if we should move servo
                    should_move, new_pan, new_tilt = self.should_move_servo(primary_target['center'])
                    
                    if should_move:
                        # This will auto-enable servos if needed
                        self.servo.set_position(new_pan, new_tilt)
                        self.last_servo_move_time = time.time()
                        self.servo_move_count += 1
                        self.last_target_center = primary_target['center']
                        print("🎯 POWER-MANAGED SERVO MOVE")
                    
                elif self.tracking_enabled and not targets:
                    # Reset tracking state when no targets
                    self.last_target_center = None
                    self.stable_target_count = 0
                    self.consistent_targets.clear()
                
                # Create display
                display_frame = self.draw_status_overlay(frame, targets)
                
                # Show video
                cv2.imshow("Power-Managed Tracker - Energy Efficient", display_frame)
                
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
                    self.consistent_targets.clear()
                elif key == ord('p') or key == ord('P'):
                    if self.servo.is_powered():
                        self.servo.force_disable_servos()
                        print(f"\n🔋 SERVO POWER DISABLED")
                    else:
                        self.servo.force_enable_servos()
                        print(f"\n🔋 SERVO POWER ENABLED")
                elif key == ord('c') or key == ord('C'):
                    print("\n🎯 CENTERING")
                    self.servo.move_to_center()
                    self.last_target_center = None
                    self.last_servo_move_time = time.time()
                    self.consistent_targets.clear()
                
                # Performance report every 120 frames
                if self.frame_count % 120 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    servo_status = "ON" if self.servo.is_powered() else "OFF"
                    print(f"📊 Frame {self.frame_count} | FPS: {fps:.1f} | Faces: {len(targets)} | Moves: {self.servo_move_count} | Servo: {servo_status}")
                
        except KeyboardInterrupt:
            print("\n⏹️ INTERRUPTED")
        
        finally:
            print("\n🛑 SHUTDOWN...")
            
            if self.tracking_enabled:
                self.tracking_enabled = False
            
            print("🔋 Cleaning up power-managed servos...")
            self.servo.cleanup()
            
            print("🧹 Camera cleanup...")
            self.camera.release()
            cv2.destroyAllWindows()
            
            # Final report
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                fps = self.frame_count / elapsed
                moves_per_minute = (self.servo_move_count / elapsed) * 60
                print(f"\n📊 POWER-MANAGED SESSION COMPLETE:")
                print(f"   Duration: {elapsed:.1f}s")
                print(f"   Frames: {self.frame_count}")
                print(f"   Average FPS: {fps:.1f}")
                print(f"   Servo moves: {self.servo_move_count}")
                print(f"   Moves per minute: {moves_per_minute:.1f}")
                print(f"   Power saved: Servos auto-disabled when idle! 🔋")
            
            print("\n✅ POWER-MANAGED TRACKER COMPLETE!")

def main():
    print("🔋 Initializing Power-Managed Tracker...")
    tracker = PowerManagedTracker()
    tracker.run()

if __name__ == "__main__":
    main()