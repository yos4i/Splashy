#!/usr/bin/env python3
"""
Fixed Tracker - Resolves UI and face detection issues
"""
import cv2
import numpy as np
import time
import sys
import os
import logging

# Fix OpenCV display issues
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Import camera test for working camera functionality
sys.path.append('tests')
try:
    from camera_test import LibcameraCapture
    CAMERA_AVAILABLE = True
    print("✅ Using LibcameraCapture from working camera test")
except ImportError:
    CAMERA_AVAILABLE = False
    print("❌ LibcameraCapture not available")

class SimpleTracker:
    """Simple, working tracker based on your original setup"""
    
    def __init__(self):
        print("\n🎯 SIMPLE FIXED TRACKER")
        print("="*40)
        
        # Use working camera setup
        self.width = 1280
        self.height = 720
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        
        print(f"📹 Camera: {self.width}x{self.height}")
        print(f"📍 Center: ({self.center_x}, {self.center_y})")
        
        # Initialize camera
        if CAMERA_AVAILABLE:
            self.camera = LibcameraCapture(self.width, self.height)
            print("✅ Camera initialized")
        else:
            print("❌ No camera available")
            self.camera = None
        
        # Simple face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("✅ Face detector loaded")
        
        # Simple tracking state
        self.show_detection = True
        self.frame_count = 0
        self.start_time = time.time()
        
        print("="*40)
    
    def detect_faces_simple(self, frame):
        """Simple face detection"""
        if frame is None:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simple face detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert to simple format
        targets = []
        for i, (x, y, w, h) in enumerate(faces):
            center = (x + w//2, y + h//2)
            targets.append({
                'center': center,
                'bbox': (x, y, w, h),
                'area': w * h
            })
        
        return targets
    
    def draw_simple_overlay(self, frame, targets):
        """Simple overlay drawing"""
        display = frame.copy()
        
        # Draw center point
        cv2.circle(display, (self.center_x, self.center_y), 10, (0, 255, 255), -1)
        
        # Draw crosshair
        cv2.line(display, (self.center_x - 30, self.center_y), 
                (self.center_x + 30, self.center_y), (0, 255, 255), 2)
        cv2.line(display, (self.center_x, self.center_y - 30), 
                (self.center_x, self.center_y + 30), (0, 255, 255), 2)
        
        # Draw faces
        for i, target in enumerate(targets):
            center = target['center']
            bbox = target['bbox']
            x, y, w, h = bbox
            
            # Face rectangle
            color = (0, 255, 0) if i == 0 else (0, 255, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            
            # Face center
            cv2.circle(display, center, 5, color, -1)
            
            # Label
            label = "FACE" if i == 0 else f"FACE {i+1}"
            cv2.putText(display, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Simple status
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        cv2.putText(display, f"FPS: {fps:.1f} | Faces: {len(targets)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(display, "Q=Quit | D=Toggle Detection", 
                   (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return display
    
    def run(self):
        """Main loop"""
        print("\n🚀 Starting Fixed Tracker")
        print("Press Q to quit, D to toggle detection display")
        print("-" * 40)
        
        if not self.camera:
            print("❌ No camera available - cannot run")
            return
        
        try:
            while True:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # Face detection
                targets = []
                if self.show_detection:
                    targets = self.detect_faces_simple(frame)
                
                # Draw overlay
                display_frame = self.draw_simple_overlay(frame, targets)
                
                # Show frame
                try:
                    cv2.imshow("Fixed Tracker - Simple Test", display_frame)
                except Exception as e:
                    print(f"❌ Display error: {e}")
                    break
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("Quit requested")
                    break
                elif key == ord('d') or key == ord('D'):
                    self.show_detection = not self.show_detection
                    status = "ON" if self.show_detection else "OFF"
                    print(f"Detection display: {status}")
                
                # Progress report
                if self.frame_count % 60 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    print(f"📊 Frame {self.frame_count} | FPS: {fps:.1f} | Faces: {len(targets)}")
                
                # Small delay
                time.sleep(0.02)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            print("🧹 Cleaning up...")
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
            
            # Final stats
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                fps = self.frame_count / elapsed
                print(f"\n📊 Session complete:")
                print(f"   Duration: {elapsed:.1f}s")
                print(f"   Frames: {self.frame_count}")
                print(f"   Average FPS: {fps:.1f}")
            
            print("✅ Fixed tracker complete!")

def main():
    print("🔧 FIXED TRACKER TEST")
    print("Simplified version to test UI and face detection")
    
    try:
        tracker = SimpleTracker()
        tracker.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()