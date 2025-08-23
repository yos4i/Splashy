#!/usr/bin/env python3
"""
Perfect Face Tracker - Servos follow faces with live video display
"""
import cv2
import numpy as np
import time
import sys
import os
import logging

# Import the exact working servo controller
sys.path.append('core')
from servo_controller import ServoController, ServoConfig

# Import camera
sys.path.append('tests')  
from camera_test import LibcameraCapture

class ImprovedFaceDetector:
    def __init__(self):
        # Load multiple cascade classifiers for better detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        print("✅ Enhanced face detector loaded")
    
    def detect_faces(self, frame):
        """Detect faces with improved parameters"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance image for better detection
        gray = cv2.equalizeHist(gray)
        
        # Detect frontal faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # More sensitive detection
            minNeighbors=3,       # Less strict neighbor requirement
            minSize=(40, 40),     # Smaller minimum size
            maxSize=(300, 300),   # Maximum size limit
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Also detect profile faces
        profiles = self.profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(40, 40),
            maxSize=(300, 300)
        )
        
        # Combine all detections
        all_faces = []
        
        # Add frontal faces
        for (x, y, w, h) in faces:
            all_faces.append((x, y, w, h, 'frontal'))
        
        # Add profile faces
        for (x, y, w, h) in profiles:
            all_faces.append((x, y, w, h, 'profile'))
        
        # Convert to target format
        targets = []
        for i, (x, y, w, h, face_type) in enumerate(all_faces):
            center_x = x + w // 2
            center_y = y + h // 2
            area = w * h
            targets.append({
                'center': (center_x, center_y),
                'bbox': (x, y, w, h),
                'area': area,
                'type': face_type,
                'id': i
            })
        
        # Sort by area (largest first)
        targets.sort(key=lambda t: t['area'], reverse=True)
        return targets

class PerfectFaceTracker:
    def __init__(self):
        # Camera setup
        self.width = 1280
        self.height = 720
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        
        print("📹 Starting HD camera...")
        self.camera = LibcameraCapture(self.width, self.height)
        
        print("🔍 Starting enhanced face detector...")
        self.detector = ImprovedFaceDetector()
        
        print("🎛️ Starting servo controller...")
        logging.basicConfig(level=logging.INFO)
        self.servo_config = ServoConfig()
        self.servo = ServoController(self.servo_config)
        self.servo.start_control_loop()
        
        # Tracking parameters
        self.tracking_enabled = False
        self.deadzone = 60  # Smaller deadzone for more responsive tracking
        self.pan_sensitivity = 0.003   # Increased sensitivity
        self.tilt_sensitivity = 0.002  # Slightly less for tilt
        
        # Target tracking
        self.last_target_time = time.time()
        self.target_lost_timeout = 2.0  # Return to center after 2s without target
        
        print(f"✅ Perfect Face Tracker ready!")
        print(f"   Camera: {self.width}x{self.height}")
        print(f"   Servo pins: Pan=GPIO{self.servo_config.pan_pin}, Tilt=GPIO{self.servo_config.tilt_pin}")
        print(f"   Deadzone: {self.deadzone}px")
    
    def calculate_servo_movement(self, target_center):
        """Calculate precise servo movement to follow face"""
        target_x, target_y = target_center
        
        # Calculate pixel errors from center
        error_x = target_x - self.center_x  # Positive = target is right of center
        error_y = self.center_y - target_y  # Positive = target is above center
        
        # Calculate distance from center
        distance = np.sqrt(error_x**2 + error_y**2)
        
        # Check if within deadzone (close enough to center)
        if distance <= self.deadzone:
            return None  # No movement needed
        
        # Calculate servo adjustments based on pixel error
        pan_adjustment = error_x * self.pan_sensitivity
        tilt_adjustment = error_y * self.tilt_sensitivity
        
        # Get current servo positions
        current_pan = self.servo.current_pan
        current_tilt = self.servo.current_tilt
        
        # Calculate new positions
        new_pan = current_pan + pan_adjustment
        new_tilt = current_tilt + tilt_adjustment
        
        # Clamp to servo limits
        new_pan = max(-1, min(1, new_pan))
        new_tilt = max(-1, min(1, new_tilt))
        
        print(f"📍 Target at ({target_x}, {target_y}) | Error: x={error_x:.0f}, y={error_y:.0f} | Distance: {distance:.0f}px")
        print(f"🎯 Servo move: Pan {current_pan:.3f} → {new_pan:.3f} | Tilt {current_tilt:.3f} → {new_tilt:.3f}")
        
        return new_pan, new_tilt
    
    def draw_fullscreen_overlay(self, frame, targets):
        """Draw comprehensive tracking overlay covering full screen"""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Draw center crosshair (larger and more visible)
        crosshair_color = (0, 255, 255)  # Bright cyan
        cv2.line(display, (self.center_x - 80, self.center_y), 
                (self.center_x + 80, self.center_y), crosshair_color, 4)
        cv2.line(display, (self.center_x, self.center_y - 80), 
                (self.center_x, self.center_y + 80), crosshair_color, 4)
        
        # Draw center circle
        cv2.circle(display, (self.center_x, self.center_y), 15, crosshair_color, -1)
        
        # Draw deadzone circle
        cv2.circle(display, (self.center_x, self.center_y), self.deadzone, (255, 255, 0), 3)
        cv2.putText(display, "DEADZONE", (self.center_x - 40, self.center_y + self.deadzone + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw all detected faces with full information
        for i, target in enumerate(targets):
            center = target['center']
            bbox = target['bbox']
            x, y, w, h = bbox
            face_type = target.get('type', 'unknown')
            area = target['area']
            
            # Color coding for different priorities
            if i == 0:  # Primary target (largest face)
                color = (0, 255, 0)      # Bright green
                thickness = 4
                label = f"PRIMARY TARGET"
            elif i < 3:  # Secondary targets
                color = (0, 255, 255)    # Yellow  
                thickness = 3
                label = f"FACE {i+1}"
            else:  # Other faces
                color = (255, 0, 255)    # Magenta
                thickness = 2
                label = f"FACE {i+1}"
            
            # Face bounding box
            cv2.rectangle(display, (x, y), (x + w, y + h), color, thickness)
            
            # Face center point
            cv2.circle(display, center, 15, color, -1)
            
            # Face information
            cv2.putText(display, label, (x, y - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(display, f"{face_type.upper()}", (x, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(display, f"Size: {w}x{h}", (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(display, f"Pos: ({center[0]}, {center[1]})", (x, y + h + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw tracking line for primary target
            if i == 0 and self.tracking_enabled:
                cv2.line(display, center, (self.center_x, self.center_y), color, 4)
                
                # Show distance and tracking status
                distance = np.sqrt((center[0] - self.center_x)**2 + (center[1] - self.center_y)**2)
                tracking_status = "CENTERED" if distance <= self.deadzone else "TRACKING"
                
                cv2.putText(display, f"{tracking_status}", 
                           (center[0] + 30, center[1] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(display, f"Dist: {distance:.0f}px", 
                           (center[0] + 30, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Comprehensive status panel
        self.draw_comprehensive_status(display, targets)
        
        return display
    
    def draw_comprehensive_status(self, frame, targets):
        """Draw detailed status information"""
        h, w = frame.shape[:2]
        
        # Main status panel (top-left)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Title
        cv2.putText(frame, "🎯 PERFECT FACE TRACKER", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Tracking status
        if self.tracking_enabled:
            status_text = "🔴 ACTIVELY TRACKING"
            status_color = (0, 255, 0)
        else:
            status_text = "⚪ STANDBY MODE"
            status_color = (128, 128, 128)
        
        cv2.putText(frame, status_text, (20, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Detection info
        cv2.putText(frame, f"👥 Faces detected: {len(targets)}", (20, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Servo information
        pan_deg, tilt_deg = self.servo.get_position_degrees()
        servo_status = "MOVING" if self.servo.is_moving else "IDLE"
        cv2.putText(frame, f"🎛️ Servos ({servo_status}): Pan {pan_deg:.1f}° | Tilt {tilt_deg:.1f}°", 
                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Primary target info
        if targets and self.tracking_enabled:
            primary = targets[0]
            tx, ty = primary['center']
            cv2.putText(frame, f"🎯 Primary target: ({tx}, {ty}) | Size: {primary['area']}", 
                       (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Controls
        cv2.putText(frame, "⌨️  T=Track | C=Center | Q=Quit | ESC=Stop", (20, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Detection statistics panel (top-right)
        if targets:
            stats_overlay = frame.copy()
            cv2.rectangle(stats_overlay, (w - 300, 10), (w - 10, 150), (0, 0, 0), -1)
            cv2.addWeighted(stats_overlay, 0.85, frame, 0.15, 0, frame)
            
            cv2.putText(frame, "📊 DETECTION STATS", (w - 290, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            for i, target in enumerate(targets[:4]):  # Show max 4 faces in stats
                y_pos = 60 + i * 20
                face_info = f"Face {i+1}: {target['area']} px²"
                color = (0, 255, 0) if i == 0 else (255, 255, 255)
                cv2.putText(frame, face_info, (w - 290, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def run(self):
        """Main tracking loop with enhanced display"""
        print("\n" + "="*80)
        print("🎯 PERFECT FACE TRACKER - ENHANCED EDITION")
        print("="*80)
        print("✅ Full-screen face detection and tracking")
        print("🎛️ Real servo control with precise following")
        print("📹 HD camera with enhanced face detection")
        print("")
        print("🎮 CONTROLS:")
        print("   T - Toggle tracking ON/OFF")
        print("   C - Center servos manually")
        print("   Q - Quit")
        print("   ESC - Emergency stop")
        print("")
        print("▶️ Position your face in camera view and press T to start tracking!")
        print("🎯 Multiple faces will be detected and highlighted")
        print("🤖 Servos will follow the largest face (primary target)")
        print("-"*80)
        
        frame_count = 0
        start_time = time.time()
        
        # Center servos at startup
        print("🎯 Centering servos...")
        self.servo.move_to_center()
        time.sleep(2)
        
        try:
            while True:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ Camera capture failed - retrying...")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Detect all faces
                targets = self.detector.detect_faces(frame)
                
                # Enhanced tracking logic
                if self.tracking_enabled and targets:
                    # Track the primary target (largest face)
                    primary_target = targets[0]
                    servo_movement = self.calculate_servo_movement(primary_target['center'])
                    
                    if servo_movement:
                        new_pan, new_tilt = servo_movement
                        self.servo.set_position(new_pan, new_tilt)
                        print("🎯 SERVO MOVING to follow primary target")
                    else:
                        if frame_count % 30 == 0:  # Print occasionally when centered
                            print("✅ PRIMARY TARGET CENTERED - no servo movement needed")
                    
                    self.last_target_time = time.time()
                    
                elif self.tracking_enabled and not targets:
                    # No targets detected - return to center after timeout
                    if time.time() - self.last_target_time > self.target_lost_timeout:
                        print("🔍 NO TARGETS - returning servos to center")
                        self.servo.move_to_center()
                        self.last_target_time = time.time()
                
                # Create enhanced display with full-screen overlay
                display_frame = self.draw_fullscreen_overlay(frame, targets)
                
                # Show the enhanced video
                cv2.imshow("Perfect Face Tracker - Enhanced Edition", display_frame)
                
                # Handle keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    print("\n⏹️ QUIT COMMAND RECEIVED")
                    break
                elif key == ord('t') or key == ord('T'):
                    self.tracking_enabled = not self.tracking_enabled
                    status = "ENABLED ✅" if self.tracking_enabled else "DISABLED ❌"
                    print(f"\n🎯 FACE TRACKING {status}")
                    if self.tracking_enabled:
                        print("🤖 Servos will now follow the largest detected face")
                    else:
                        print("⏸️ Servos will stay in current position")
                elif key == ord('c') or key == ord('C'):
                    print("\n🎯 MANUAL CENTER COMMAND")
                    self.servo.move_to_center()
                
                # Progress report every 90 frames (~3 seconds)
                if frame_count % 90 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    pan_deg, tilt_deg = self.servo.get_position_degrees()
                    print(f"📊 Frame {frame_count} | FPS: {fps:.1f} | Faces: {len(targets)} | Servos: Pan {pan_deg:.1f}° Tilt {tilt_deg:.1f}° | Tracking: {self.tracking_enabled}")
                
                # Small delay for stability
                time.sleep(0.025)  # ~40 FPS max
                
        except KeyboardInterrupt:
            print("\n⏹️ INTERRUPTED BY KEYBOARD")
        
        finally:
            print("\n🛑 SHUTTING DOWN SAFELY...")
            
            # Safe shutdown sequence
            if self.tracking_enabled:
                print("🛑 Disabling tracking...")
                self.tracking_enabled = False
                
            print("🎯 Centering servos for safe shutdown...")
            self.servo.move_to_center()
            time.sleep(2)
            
            # Cleanup resources
            print("🧹 Cleaning up resources...")
            self.servo.cleanup()
            self.camera.release()
            cv2.destroyAllWindows()
            
            # Final report
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                print(f"\n📊 SESSION COMPLETED:")
                print(f"   Duration: {elapsed:.1f} seconds")
                print(f"   Frames processed: {frame_count}")
                print(f"   Average FPS: {fps:.1f}")
                print(f"   Total face detections: Many! 😊")
            
            print("\n✅ PERFECT FACE TRACKER STOPPED SAFELY")
            print("🎯 Thank you for using the Perfect Face Tracker!")

def main():
    print("🚀 Initializing Perfect Face Tracker...")
    tracker = PerfectFaceTracker()
    tracker.run()

if __name__ == "__main__":
    main()