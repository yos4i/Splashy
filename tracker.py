#!/usr/bin/env python3
"""
Tracker - Ultra High-Resolution Face Tracking System
Maximum camera resolution with advanced detection and smooth servo control
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

class UltraHighResCapture:
    """Ultra high resolution camera capture using maximum Pi Camera v3 capabilities"""
    
    def __init__(self, width=4608, height=2592, quality=95):
        """Initialize with maximum resolution capabilities"""
        self.width = width
        self.height = height 
        self.quality = quality
        self.temp_file = None
        
        # Performance optimization
        self.capture_interval = 0.07  # ~14 FPS for max resolution (hardware limit)
        self.last_capture_time = 0
        
        print(f"📹 Ultra HD Camera initialized: {width}x{height} ({width*height//1000000:.1f}MP)")
        print(f"🎯 Target FPS: ~{1/self.capture_interval:.1f} (hardware limited)")
        
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
        """Capture ultra high resolution frame"""
        current_time = time.time()
        
        # Respect capture interval to avoid overwhelming the camera
        time_since_last = current_time - self.last_capture_time
        if time_since_last < self.capture_interval:
            time.sleep(self.capture_interval - time_since_last)
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Capture using maximum resolution with optimizations
            cmd = [
                'libcamera-still',
                '--output', temp_path,
                '--timeout', '100',  # Quick capture
                '--width', str(self.width),
                '--height', str(self.height),
                '--quality', str(self.quality),
                '--nopreview',  # No preview for speed
                '--immediate',  # Capture immediately
                '--encoding', 'jpg'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=2)
            
            if result.returncode == 0 and os.path.exists(temp_path):
                # Load and return the ultra high resolution image
                frame = cv2.imread(temp_path)
                os.unlink(temp_path)  # Clean up temp file
                
                self.last_capture_time = time.time()
                
                if frame is not None:
                    return True, frame
                else:
                    print("❌ Failed to load captured image")
                    return False, None
            else:
                print(f"❌ Camera capture failed: {result.stderr.decode()}")
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

class AdvancedFaceDetector:
    """Advanced face detection with multiple methods and optimizations"""
    
    def __init__(self):
        print("🔍 Loading advanced face detection systems...")
        
        # Load Haar cascade classifiers (fast, good for high-res)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Try to load DNN-based detector for better accuracy
        self.dnn_net = None
        try:
            # OpenCV DNN face detector (if available)
            model_file = "opencv_face_detector_uint8.pb"
            config_file = "opencv_face_detector.pbtxt"
            
            # These files would need to be downloaded separately
            # For now, we'll use enhanced Haar cascades
            print("📊 Using enhanced Haar cascade detection")
            
        except Exception as e:
            print("📊 Using Haar cascade detection (DNN not available)")
        
        # Detection parameters optimized for high resolution
        self.scale_factor = 1.05  # More sensitive for high-res
        self.min_neighbors = 3    # Less strict for better detection
        self.min_face_size = (80, 80)   # Larger minimum for high-res
        self.max_face_size = (800, 800) # Much larger maximum for high-res
        
        print("✅ Advanced face detector ready for ultra-high resolution")
    
    def detect_faces(self, frame):
        """Enhanced face detection optimized for ultra-high resolution"""
        if frame is None:
            return []
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhance image for better detection at high resolution
        gray = cv2.equalizeHist(gray)
        
        # For very high resolution, we might want to detect on a scaled version
        # then scale coordinates back up for better performance
        scale = 1.0
        if gray.shape[0] > 2000:  # If height > 2000px, scale down for detection
            scale = 2000 / gray.shape[0]
            detection_height = int(gray.shape[0] * scale)
            detection_width = int(gray.shape[1] * scale)
            gray_scaled = cv2.resize(gray, (detection_width, detection_height))
        else:
            gray_scaled = gray
            
        # Detect frontal faces
        faces = self.face_cascade.detectMultiScale(
            gray_scaled,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(int(self.min_face_size[0] * scale), int(self.min_face_size[1] * scale)),
            maxSize=(int(self.max_face_size[0] * scale), int(self.max_face_size[1] * scale)),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Detect profile faces
        profiles = self.profile_cascade.detectMultiScale(
            gray_scaled,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(int(self.min_face_size[0] * scale), int(self.min_face_size[1] * scale)),
            maxSize=(int(self.max_face_size[0] * scale), int(self.max_face_size[1] * scale))
        )
        
        # Scale coordinates back to original resolution
        all_faces = []
        
        # Add frontal faces
        for (x, y, w, h) in faces:
            if scale != 1.0:
                x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
            all_faces.append((x, y, w, h, 'frontal'))
        
        # Add profile faces
        for (x, y, w, h) in profiles:
            if scale != 1.0:
                x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
            all_faces.append((x, y, w, h, 'profile'))
        
        # Convert to target format with enhanced information
        targets = []
        for i, (x, y, w, h, face_type) in enumerate(all_faces):
            center_x = x + w // 2
            center_y = y + h // 2
            area = w * h
            
            # Calculate confidence based on size and position
            confidence = min(1.0, area / 50000)  # Normalize based on typical face size
            
            targets.append({
                'center': (center_x, center_y),
                'bbox': (x, y, w, h),
                'area': area,
                'type': face_type,
                'confidence': confidence,
                'id': i
            })
        
        # Sort by area (largest first) for better tracking priority
        targets.sort(key=lambda t: t['area'], reverse=True)
        
        return targets

class UltraTracker:
    """Ultra high-resolution face tracking system"""
    
    def __init__(self):
        print("\n🚀 INITIALIZING ULTRA TRACKER")
        print("="*60)
        
        # Ultra high resolution setup (maximum Pi Camera v3 capability)
        self.width = 4608   # 4K+ width
        self.height = 2592  # 4K+ height  
        self.total_pixels = self.width * self.height
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        
        print(f"📹 Ultra HD Camera: {self.width}x{self.height}")
        print(f"🎯 Total Pixels: {self.total_pixels:,} ({self.total_pixels//1000000:.1f} Megapixels)")
        print(f"📍 Center Point: ({self.center_x}, {self.center_y})")
        
        # Initialize ultra high-res camera
        self.camera = UltraHighResCapture(self.width, self.height)
        
        # Initialize advanced face detector
        self.detector = AdvancedFaceDetector()
        
        # Initialize servo controller
        print("🎛️ Starting precision servo controller...")
        logging.basicConfig(level=logging.INFO)
        self.servo_config = ServoConfig()
        self.servo = ServoController(self.servo_config)
        self.servo.start_control_loop()
        
        # Enhanced tracking parameters for ultra-high resolution
        self.tracking_enabled = False
        self.deadzone = 150  # Larger deadzone for ultra-high res
        
        # Ultra-precise sensitivity (smaller values for high resolution)
        self.pan_sensitivity = 0.0008   # More precise for ultra-high res
        self.tilt_sensitivity = 0.0005  # Even more precise for tilt
        
        # Enhanced tracking features
        self.last_target_time = time.time()
        self.target_lost_timeout = 2.5
        self.smoothing_factor = 0.3  # Smooth movement for stability
        self.last_target_center = None
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = time.time()
        self.detection_history = []
        
        print("✅ ULTRA TRACKER READY!")
        print(f"🎯 Deadzone: {self.deadzone}px")
        print(f"🎛️ Pan sensitivity: {self.pan_sensitivity}")
        print(f"🎛️ Tilt sensitivity: {self.tilt_sensitivity}")
        print("="*60)
    
    def calculate_precision_movement(self, target_center):
        """Calculate ultra-precise servo movement for high-resolution tracking"""
        target_x, target_y = target_center
        
        # Apply smoothing if we have previous target
        if self.last_target_center and self.smoothing_factor > 0:
            prev_x, prev_y = self.last_target_center
            target_x = prev_x + (target_x - prev_x) * (1 - self.smoothing_factor)
            target_y = prev_y + (target_y - prev_y) * (1 - self.smoothing_factor)
            target_x, target_y = int(target_x), int(target_y)
        
        self.last_target_center = (target_x, target_y)
        
        # Calculate pixel errors from center
        error_x = target_x - self.center_x
        error_y = self.center_y - target_y
        
        # Calculate distance from center
        distance = np.sqrt(error_x**2 + error_y**2)
        
        # Check if within deadzone
        if distance <= self.deadzone:
            return None  # No movement needed
        
        # Ultra-precise servo adjustments
        pan_adjustment = error_x * self.pan_sensitivity
        tilt_adjustment = error_y * self.tilt_sensitivity
        
        # Get current positions
        current_pan = self.servo.current_pan
        current_tilt = self.servo.current_tilt
        
        # Calculate new positions with limits
        new_pan = max(-1, min(1, current_pan + pan_adjustment))
        new_tilt = max(-1, min(1, current_tilt + tilt_adjustment))
        
        print(f"📍 Ultra-HD Target: ({target_x}, {target_y}) | Error: x={error_x:.0f}, y={error_y:.0f} | Dist: {distance:.0f}px")
        print(f"🎯 Precision Move: Pan {current_pan:.4f} → {new_pan:.4f} | Tilt {current_tilt:.4f} → {new_tilt:.4f}")
        
        return new_pan, new_tilt
    
    def draw_ultra_hd_overlay(self, frame, targets):
        """Draw comprehensive overlay optimized for ultra-high resolution"""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Scale UI elements for ultra-high resolution
        ui_scale = max(1, w // 1920)  # Scale UI based on width
        font_scale = ui_scale * 0.8
        thickness = max(1, ui_scale)
        
        # Draw ultra-precise center crosshair
        crosshair_size = 120 * ui_scale
        crosshair_color = (0, 255, 255)  # Bright cyan
        cv2.line(display, (self.center_x - crosshair_size, self.center_y), 
                (self.center_x + crosshair_size, self.center_y), crosshair_color, thickness * 2)
        cv2.line(display, (self.center_x, self.center_y - crosshair_size), 
                (self.center_x, self.center_y + crosshair_size), crosshair_color, thickness * 2)
        
        # Draw center point
        cv2.circle(display, (self.center_x, self.center_y), 20 * ui_scale, crosshair_color, -1)
        
        # Draw ultra-high res deadzone
        cv2.circle(display, (self.center_x, self.center_y), self.deadzone, (255, 255, 0), thickness)
        cv2.putText(display, f"DEADZONE ({self.deadzone}px)", 
                   (self.center_x - 100 * ui_scale, self.center_y + self.deadzone + 30 * ui_scale), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 255, 0), thickness)
        
        # Draw detected faces with ultra-high res details
        for i, target in enumerate(targets):
            center = target['center']
            bbox = target['bbox']
            x, y, w, h = bbox
            face_type = target.get('type', 'unknown')
            area = target['area']
            confidence = target.get('confidence', 0.5)
            
            # Color coding
            if i == 0:  # Primary target
                color = (0, 255, 0)
                thickness_face = thickness * 3
                label = "PRIMARY TARGET"
            elif i < 3:
                color = (0, 255, 255)
                thickness_face = thickness * 2
                label = f"FACE {i+1}"
            else:
                color = (255, 0, 255)
                thickness_face = thickness
                label = f"FACE {i+1}"
            
            # Face bounding box
            cv2.rectangle(display, (x, y), (x + w, y + h), color, thickness_face)
            
            # Face center point (larger for ultra-high res)
            cv2.circle(display, center, 25 * ui_scale, color, -1)
            
            # Ultra-detailed face information
            text_offset = 50 * ui_scale
            cv2.putText(display, label, (x, y - text_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            cv2.putText(display, f"{face_type.upper()}", (x, y - text_offset//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, color, thickness)
            cv2.putText(display, f"Size: {w}x{h}px ({area:,} px²)", (x, y + h + text_offset//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, color, max(1, thickness-1))
            cv2.putText(display, f"Pos: ({center[0]}, {center[1]})", (x, y + h + text_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, color, max(1, thickness-1))
            cv2.putText(display, f"Conf: {confidence:.2f}", (x, y + h + text_offset * 3//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, color, max(1, thickness-1))
            
            # Draw tracking line for primary target
            if i == 0 and self.tracking_enabled:
                cv2.line(display, center, (self.center_x, self.center_y), color, thickness * 2)
                
                # Enhanced tracking status
                distance = np.sqrt((center[0] - self.center_x)**2 + (center[1] - self.center_y)**2)
                tracking_status = "CENTERED" if distance <= self.deadzone else "TRACKING"
                
                cv2.putText(display, tracking_status, 
                           (center[0] + 40 * ui_scale, center[1] - 40 * ui_scale), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                cv2.putText(display, f"Dist: {distance:.0f}px", 
                           (center[0] + 40 * ui_scale, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, color, thickness)
        
        # Ultra-comprehensive status panel
        self.draw_ultra_status(display, targets, ui_scale, font_scale, thickness)
        
        return display
    
    def draw_ultra_status(self, frame, targets, ui_scale, font_scale, thickness):
        """Draw ultra-detailed status information for high resolution"""
        h, w = frame.shape[:2]
        
        # Main status panel (scaled for ultra-high res)
        panel_width = 600 * ui_scale
        panel_height = 300 * ui_scale
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Ultra tracker title
        cv2.putText(frame, "🚀 ULTRA TRACKER", (20, 50 * ui_scale), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (0, 255, 255), thickness * 2)
        
        # Resolution info
        cv2.putText(frame, f"📹 {self.width}x{self.height} ({self.total_pixels//1000000:.1f}MP)", 
                   (20, 90 * ui_scale), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), thickness)
        
        # Tracking status
        if self.tracking_enabled:
            status_text = "🔴 ULTRA PRECISION TRACKING"
            status_color = (0, 255, 0)
        else:
            status_text = "⚪ STANDBY MODE"
            status_color = (128, 128, 128)
        
        cv2.putText(frame, status_text, (20, 130 * ui_scale), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, status_color, thickness)
        
        # Ultra-detailed detection info
        cv2.putText(frame, f"👥 Ultra-HD Faces: {len(targets)}", (20, 170 * ui_scale), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), thickness)
        
        # Performance info
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"⚡ FPS: {fps:.2f} | Frame: {self.frame_count}", (20, 210 * ui_scale), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 255, 255), thickness)
        
        # Servo status
        pan_deg, tilt_deg = self.servo.get_position_degrees()
        servo_status = "MOVING" if self.servo.is_moving else "IDLE"
        cv2.putText(frame, f"🎛️ Servos ({servo_status}): Pan {pan_deg:.1f}° | Tilt {tilt_deg:.1f}°", 
                   (20, 250 * ui_scale), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 255, 255), thickness)
        
        # Ultra controls
        cv2.putText(frame, "⌨️ T=Track | C=Center | Q=Quit | ESC=Stop", (20, 290 * ui_scale), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (255, 255, 0), max(1, thickness-1))
    
    def run(self):
        """Main ultra-high resolution tracking loop"""
        print("\n" + "="*80)
        print("🚀 ULTRA TRACKER - MAXIMUM RESOLUTION EDITION")
        print("="*80)
        print(f"📹 Ultra HD: {self.width}x{self.height} ({self.total_pixels//1000000:.1f} Megapixels)")
        print("🎯 Maximum camera resolution with precision tracking")
        print("🎛️ Ultra-precise servo control with smoothing")
        print("")
        print("🎮 ULTRA CONTROLS:")
        print("   T - Toggle ultra-precision tracking ON/OFF")
        print("   C - Center servos manually")
        print("   Q - Quit")
        print("   ESC - Emergency stop")
        print("")
        print("▶️ Experience face tracking at maximum resolution!")
        print("-"*80)
        
        # Center servos
        print("🎯 Centering servos for ultra-precision...")
        self.servo.move_to_center()
        time.sleep(2)
        
        try:
            while True:
                # Capture ultra-high resolution frame
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ Ultra-HD capture failed - retrying...")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # Ultra-high resolution face detection
                targets = self.detector.detect_faces(frame)
                
                # Ultra-precision tracking
                if self.tracking_enabled and targets:
                    primary_target = targets[0]
                    servo_movement = self.calculate_precision_movement(primary_target['center'])
                    
                    if servo_movement:
                        new_pan, new_tilt = servo_movement
                        self.servo.set_position(new_pan, new_tilt)
                        print("🎯 ULTRA-PRECISION SERVO MOVEMENT")
                    else:
                        if self.frame_count % 30 == 0:
                            print("✅ ULTRA TARGET CENTERED - ultra-precise!")
                    
                    self.last_target_time = time.time()
                    
                elif self.tracking_enabled and not targets:
                    if time.time() - self.last_target_time > self.target_lost_timeout:
                        print("🔍 NO ULTRA TARGETS - returning to center")
                        self.servo.move_to_center()
                        self.last_target_time = time.time()
                        self.last_target_center = None
                
                # Create ultra-high resolution display
                display_frame = self.draw_ultra_hd_overlay(frame, targets)
                
                # Show ultra-high resolution video (OpenCV will scale for display)
                cv2.imshow("Ultra Tracker - Maximum Resolution", display_frame)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("\n⏹️ ULTRA QUIT COMMAND")
                    break
                elif key == ord('t') or key == ord('T'):
                    self.tracking_enabled = not self.tracking_enabled
                    status = "ENABLED ✅" if self.tracking_enabled else "DISABLED ❌"
                    print(f"\n🎯 ULTRA-PRECISION TRACKING {status}")
                elif key == ord('c') or key == ord('C'):
                    print("\n🎯 ULTRA-PRECISION CENTER")
                    self.servo.move_to_center()
                    self.last_target_center = None
                
                # Ultra progress report
                if self.frame_count % 60 == 0:  # Every 60 frames
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    pan_deg, tilt_deg = self.servo.get_position_degrees()
                    megapixels_processed = (self.frame_count * self.total_pixels) / 1000000
                    print(f"📊 Ultra Frame {self.frame_count} | FPS: {fps:.2f} | Faces: {len(targets)} | "
                          f"MP Processed: {megapixels_processed:.0f} | Servos: Pan {pan_deg:.1f}° Tilt {tilt_deg:.1f}°")
                
                # Small delay for stability
                time.sleep(0.02)
                
        except KeyboardInterrupt:
            print("\n⏹️ ULTRA INTERRUPTED")
        
        finally:
            print("\n🛑 ULTRA SHUTDOWN...")
            
            if self.tracking_enabled:
                self.tracking_enabled = False
            
            print("🎯 Ultra-centering servos...")
            self.servo.move_to_center()
            time.sleep(2)
            
            print("🧹 Ultra cleanup...")
            self.servo.cleanup()
            self.camera.release()
            cv2.destroyAllWindows()
            
            # Ultra final report
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                fps = self.frame_count / elapsed
                megapixels_processed = (self.frame_count * self.total_pixels) / 1000000
                print(f"\n📊 ULTRA SESSION COMPLETE:")
                print(f"   Duration: {elapsed:.1f}s")
                print(f"   Ultra Frames: {self.frame_count}")
                print(f"   Ultra FPS: {fps:.2f}")
                print(f"   Megapixels Processed: {megapixels_processed:.0f}")
                print(f"   Total Pixel Operations: {self.frame_count * self.total_pixels:,}")
            
            print("\n✅ ULTRA TRACKER COMPLETE!")
            print("🚀 Maximum resolution face tracking achieved!")

def main():
    print("🚀 Initializing Ultra Tracker - Maximum Resolution...")
    tracker = UltraTracker()
    tracker.run()

if __name__ == "__main__":
    main()