#!/usr/bin/env python3
"""
Final Target Tracker - Uses RPi.GPIO like the working servo test
"""
import cv2
import numpy as np
import time
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tests'))

from camera_test import LibcameraCapture

# Try to import GPIO (same method as working servo test)
GPIO_AVAILABLE = False
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    print("✅ RPi.GPIO imported successfully")
except ImportError:
    print("❌ RPi.GPIO not available")

class DirectServoController:
    """Direct servo control using RPi.GPIO like the working test"""
    
    def __init__(self, pan_pin=12, tilt_pin=13):
        self.pan_pin = pan_pin
        self.tilt_pin = tilt_pin
        self.simulation_mode = not GPIO_AVAILABLE
        
        # Current angles
        self.pan_angle = 0.0
        self.tilt_angle = 15.0  # Start slightly up
        
        if not self.simulation_mode:
            try:
                # Setup GPIO (same as working servo test)
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.pan_pin, GPIO.OUT)
                GPIO.setup(self.tilt_pin, GPIO.OUT)
                
                # Create PWM instances (50Hz for servos)
                self.pan_pwm = GPIO.PWM(self.pan_pin, 50)
                self.tilt_pwm = GPIO.PWM(self.tilt_pin, 50)
                
                # Start PWM at center positions
                self.pan_pwm.start(7.5)  # Center = 7.5% duty cycle
                self.tilt_pwm.start(8.0)  # Slightly up
                
                print(f"✅ Direct servo control initialized: Pan=GPIO{pan_pin}, Tilt=GPIO{tilt_pin}")
                
            except Exception as e:
                print(f"❌ Failed to initialize GPIO servos: {e}")
                self.simulation_mode = True
        
        if self.simulation_mode:
            print("🎯 Servo controller in simulation mode")
    
    def angle_to_duty_cycle(self, angle):
        """Convert angle (-90 to 90) to PWM duty cycle (5.0 to 10.0)"""
        # Servo duty cycle: 5% = -90°, 7.5% = 0°, 10% = +90°
        # Linear mapping: duty = 7.5 + (angle * 2.5 / 90)
        duty = 7.5 + (angle * 2.5 / 90.0)
        return max(5.0, min(10.0, duty))  # Clamp to safe range
    
    def move_to_angles(self, pan_angle, tilt_angle):
        """Move servos to specific angles"""
        # Clamp angles to safe limits
        pan_angle = max(-90, min(90, pan_angle))
        tilt_angle = max(-30, min(60, tilt_angle))
        
        # Check if movement is significant
        pan_diff = abs(pan_angle - self.pan_angle)
        tilt_diff = abs(tilt_angle - self.tilt_angle)
        
        if pan_diff > 3 or tilt_diff > 3:  # Only move if change > 3 degrees
            print(f"🎯 SERVO MOVE: Pan {self.pan_angle:.1f}° → {pan_angle:.1f}° | Tilt {self.tilt_angle:.1f}° → {tilt_angle:.1f}°")
            
            if not self.simulation_mode:
                try:
                    # Convert angles to duty cycles
                    pan_duty = self.angle_to_duty_cycle(pan_angle)
                    tilt_duty = self.angle_to_duty_cycle(tilt_angle)
                    
                    # Move servos
                    self.pan_pwm.ChangeDutyCycle(pan_duty)
                    self.tilt_pwm.ChangeDutyCycle(tilt_duty)
                    
                    print(f"   PWM: Pan {pan_duty:.1f}% | Tilt {tilt_duty:.1f}%")
                    
                except Exception as e:
                    print(f"❌ Servo movement failed: {e}")
            
            # Update current positions
            self.pan_angle = pan_angle
            self.tilt_angle = tilt_angle
            
            # Delay for servo movement
            time.sleep(0.2)
    
    def center(self):
        """Center both servos"""
        print("🎯 CENTERING SERVOS")
        self.move_to_angles(0, 15)
    
    def cleanup(self):
        """Clean up GPIO resources"""
        if not self.simulation_mode:
            try:
                self.pan_pwm.stop()
                self.tilt_pwm.stop()
                GPIO.cleanup()
                print("🛑 GPIO cleaned up")
            except:
                pass

class FaceTracker:
    """Face detection and tracking"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("✅ Face detector ready")
    
    def detect_faces(self, frame):
        """Detect faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with more aggressive parameters for better detection
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,  # Smaller scale factor for more detection
            minNeighbors=4,   # Fewer neighbors required
            minSize=(60, 60), # Smaller minimum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        targets = []
        for (x, y, w, h) in faces:
            center_x = x + w // 2
            center_y = y + h // 2
            confidence = w * h
            targets.append({
                'center': (center_x, center_y),
                'bbox': (x, y, w, h),
                'confidence': confidence
            })
        
        # Sort by size (largest first)
        targets.sort(key=lambda t: t['confidence'], reverse=True)
        return targets

class FinalTargetTracker:
    """Final working target tracker"""
    
    def __init__(self):
        # Camera setup
        self.width = 1280
        self.height = 720
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        
        # Components
        print("🎥 Initializing camera...")
        self.camera = LibcameraCapture(self.width, self.height)
        
        print("🔍 Initializing face detector...")
        self.detector = FaceTracker()
        
        print("🎛️ Initializing servo controller...")
        self.servo = DirectServoController()
        
        # Tracking parameters
        self.tracking_enabled = False
        self.deadzone = 120  # Larger deadzone for stability
        self.pan_sensitivity = 0.10  # Pan sensitivity
        self.tilt_sensitivity = 0.08  # Tilt sensitivity (less sensitive)
        
        # State
        self.running = True
        self.target_lost_count = 0
        
        print(f"✅ Tracker ready: {self.width}x{self.height}, deadzone={self.deadzone}px")
    
    def calculate_movement(self, target_center):
        """Calculate servo movement to center target"""
        target_x, target_y = target_center
        
        # Calculate pixel errors from center
        error_x = target_x - self.center_x  # Positive = target is right
        error_y = self.center_y - target_y  # Positive = target is up (inverted Y)
        
        # Check deadzone
        distance = np.sqrt(error_x**2 + error_y**2)
        if distance <= self.deadzone:
            return None  # No movement needed - target is centered
        
        # Calculate angle adjustments
        pan_adjustment = error_x * self.pan_sensitivity
        tilt_adjustment = error_y * self.tilt_sensitivity
        
        # Apply to current servo positions
        new_pan = self.servo.pan_angle + pan_adjustment
        new_tilt = self.servo.tilt_angle + tilt_adjustment
        
        print(f"📍 Target at ({target_x}, {target_y}) | Error: x={error_x:.0f}, y={error_y:.0f} | Distance: {distance:.0f}")
        
        return new_pan, new_tilt
    
    def draw_interface(self, frame, targets):
        """Draw tracking interface"""
        display = frame.copy()
        
        # Draw center crosshair (larger and more visible)
        cv2.line(display, (self.center_x - 60, self.center_y), 
                (self.center_x + 60, self.center_y), (0, 255, 255), 4)
        cv2.line(display, (self.center_x, self.center_y - 60), 
                (self.center_x, self.center_y + 60), (0, 255, 255), 4)
        
        # Draw deadzone circle
        cv2.circle(display, (self.center_x, self.center_y), self.deadzone, (255, 255, 0), 3)
        
        # Draw detected faces
        for i, target in enumerate(targets):
            center = target['center']
            bbox = target['bbox']
            x, y, w, h = bbox
            
            # Color coding: primary target = bright green
            if i == 0:
                color = (0, 255, 0)  # Bright green
                thickness = 4
            else:
                color = (0, 255, 255)  # Yellow
                thickness = 2
            
            # Face bounding box
            cv2.rectangle(display, (x, y), (x + w, y + h), color, thickness)
            
            # Center point
            cv2.circle(display, center, 20, color, -1)
            
            # Face number
            cv2.putText(display, f"FACE {i+1}", (x, y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Tracking line for primary target
            if i == 0 and self.tracking_enabled:
                cv2.line(display, center, (self.center_x, self.center_y), color, 3)
                
                # Show distance
                distance = np.sqrt((center[0] - self.center_x)**2 + (center[1] - self.center_y)**2)
                cv2.putText(display, f"DIST: {distance:.0f}px", (center[0] + 30, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Status panel
        self.draw_status_panel(display, targets)
        return display
    
    def draw_status_panel(self, frame, targets):
        """Draw comprehensive status panel"""
        # Background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (650, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Title
        status_text = "🎯 TRACKING ACTIVE" if self.tracking_enabled else "⏸️ STANDBY MODE"
        status_color = (0, 255, 0) if self.tracking_enabled else (128, 128, 128)
        
        cv2.putText(frame, "FINAL TARGET TRACKER", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.putText(frame, status_text, (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Detection info
        cv2.putText(frame, f"Faces detected: {len(targets)}", (20, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Servo positions
        cv2.putText(frame, f"Servo angles: Pan {self.servo.pan_angle:.0f}° | Tilt {self.servo.tilt_angle:.0f}°", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(frame, "T=Track | C=Center | Q=Quit | ESC=Stop", (20, 145), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Deadzone info
        cv2.putText(frame, f"Deadzone: {self.deadzone}px", (20, 165), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    def run(self):
        """Main tracking loop"""
        print("\n" + "="*70)
        print("🎯 FINAL TARGET TRACKER - READY TO TRACK!")
        print("="*70)
        print("📹 HD camera with face detection")
        print("🎛️ Real servo control via RPi.GPIO")
        print("🎮 Controls:")
        print("   T - Toggle tracking ON/OFF")
        print("   C - Center servos")
        print("   Q - Quit")
        print("   ESC - Emergency stop")
        print("\n▶️ Position your face in front of camera and press T to start tracking!")
        print("-"*70)
        
        # Stats
        frame_count = 0
        start_time = time.time()
        
        # Initialize - center servos
        self.servo.center()
        time.sleep(1)
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ Camera failed")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Detect faces
                targets = self.detector.detect_faces(frame)
                
                # Tracking logic
                if self.tracking_enabled and targets:
                    primary_target = targets[0]
                    movement = self.calculate_movement(primary_target['center'])
                    
                    if movement:
                        new_pan, new_tilt = movement
                        self.servo.move_to_angles(new_pan, new_tilt)
                        print(f"🎯 TRACKING: Moving to follow face")
                    else:
                        print(f"✅ TARGET CENTERED (within deadzone)")
                    
                    self.target_lost_count = 0
                
                elif self.tracking_enabled and not targets:
                    # Target lost
                    self.target_lost_count += 1
                    if self.target_lost_count > 30:  # Lost for 30 frames
                        print("🔍 TARGET LOST - Centering servos")
                        self.servo.center()
                        self.target_lost_count = 0
                
                # Create display
                display_frame = self.draw_interface(frame, targets)
                
                # Show video
                cv2.imshow("Final Target Tracker", display_frame)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    print("\n⏹️ QUIT REQUESTED")
                    break
                elif key == ord('t') or key == ord('T'):
                    self.tracking_enabled = not self.tracking_enabled
                    status = "ENABLED ✅" if self.tracking_enabled else "DISABLED ❌"
                    print(f"\n🎯 TRACKING {status}")
                elif key == ord('c') or key == ord('C'):
                    print("\n🎯 MANUAL CENTER COMMAND")
                    self.servo.center()
                
                # Progress every 60 frames
                if frame_count % 60 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"📊 Frame {frame_count} | FPS: {fps:.1f} | Faces: {len(targets)} | Tracking: {self.tracking_enabled}")
                
                # Frame delay
                time.sleep(0.03)
                
        except KeyboardInterrupt:
            print("\n⏹️ INTERRUPTED BY USER")
        
        finally:
            print("\n🛑 SHUTTING DOWN SAFELY...")
            
            # Center servos before exit
            if not self.servo.simulation_mode:
                print("🎯 Centering servos for safe shutdown...")
                self.servo.center()
                time.sleep(1)
            
            # Cleanup
            self.servo.cleanup()
            self.camera.release()
            cv2.destroyAllWindows()
            
            # Final report
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                print(f"\n📊 FINAL REPORT:")
                print(f"   Session duration: {elapsed:.1f} seconds")
                print(f"   Frames processed: {frame_count}")
                print(f"   Average FPS: {fps:.1f}")
                print(f"   Servo mode: {'REAL' if not self.servo.simulation_mode else 'SIMULATION'}")
            
            print("\n✅ TARGET TRACKER STOPPED SAFELY")

def main():
    print("🎯 Initializing Final Target Tracker...")
    tracker = FinalTargetTracker()
    tracker.run()

if __name__ == "__main__":
    main()