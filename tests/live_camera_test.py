#!/usr/bin/env python3
import cv2
import numpy as np
import time
from camera_test import LibcameraCapture
from color_detection_test import ColorDetector

class LiveCameraViewer:
    """Live camera viewer with green detection overlay - NO MOTORS"""
    
    def __init__(self):
        print("🎥 Initializing Live Camera Viewer...")
        print("📵 NO MOTOR CONTROL - Safe for testing!")
        
        # Initialize camera only
        self.camera = LibcameraCapture(1280, 720)
        self.detector = ColorDetector()
        
        # Display settings
        self.frame_count = 0
        self.start_time = time.time()
        self.show_detection = True
        self.show_crosshair = True
        
        print("✅ Camera viewer ready!")
        print("\nControls:")
        print("  CTRL+C - Exit")
        print("  Screenshots saved every 10 seconds")
    
    def draw_overlay(self, frame, targets):
        """Draw detection overlay on frame"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        if self.show_crosshair:
            # Draw center crosshair
            center_x, center_y = w // 2, h // 2
            cv2.line(display_frame, (center_x - 30, center_y), 
                    (center_x + 30, center_y), (255, 255, 255), 2)
            cv2.line(display_frame, (center_x, center_y - 30), 
                    (center_x, center_y + 30), (255, 255, 255), 2)
            
            # Draw deadzone circle
            cv2.circle(display_frame, (center_x, center_y), 50, (255, 255, 255), 1)
        
        if self.show_detection and targets:
            # Draw green detection boxes
            for i, target in enumerate(targets):
                center = target['center']
                bbox = target['bbox']
                area = target['area']
                x, y, w, h = bbox
                
                # Green box around detected objects
                color = (0, 255, 0)  # Bright green
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                
                # Center point
                cv2.circle(display_frame, center, 8, color, -1)
                
                # Label
                label = f"GREEN #{i+1}"
                cv2.putText(display_frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Status info overlay
        self.draw_status_overlay(display_frame, targets)
        
        return display_frame
    
    def draw_status_overlay(self, frame, targets):
        """Draw status information"""
        h, w = frame.shape[:2]
        
        # Semi-transparent panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status text
        fps = self.frame_count / max(1, time.time() - self.start_time)
        
        cv2.putText(frame, f"📹 LIVE CAMERA VIEW", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.putText(frame, f"Frame: {self.frame_count} | FPS: {fps:.1f}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if targets:
            cv2.putText(frame, f"🟢 Green Objects: {len(targets)}", 
                       (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show largest target position
            largest = targets[0]
            tx, ty = largest['center']
            cv2.putText(frame, f"Target: ({tx}, {ty})", 
                       (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, f"🔍 No green objects", 
                       (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # Safety indicator
        cv2.putText(frame, "📵 MOTORS DISABLED", 
                   (w - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def start_live_view(self, duration=60):
        """Start live camera viewing"""
        print(f"\n🎬 Starting live camera view for {duration} seconds...")
        print("Move the camera around and hold up green objects!")
        print("Screenshots will be saved automatically.\n")
        
        start_time = time.time()
        last_screenshot = 0
        
        try:
            while time.time() - start_time < duration:
                self.frame_count += 1
                
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Detect green objects
                targets, mask = self.detector.detect_green(frame)
                
                # Create display frame with overlays
                display_frame = self.draw_overlay(frame, targets)
                
                # Save screenshots every 10 seconds
                current_time = time.time()
                if current_time - last_screenshot >= 10:
                    timestamp = int(current_time)
                    filename = f'live_camera_{timestamp}.jpg'
                    cv2.imwrite(filename, display_frame)
                    print(f"📸 Screenshot saved: {filename}")
                    last_screenshot = current_time
                
                # Print live status
                if targets:
                    largest = targets[0]
                    tx, ty = largest['center']
                    print(f"Frame {self.frame_count:4d}: 🟢 Green detected at ({tx:3d}, {ty:3d}) | Total: {len(targets)} objects")
                else:
                    if self.frame_count % 20 == 0:  # Print every 20 frames when no detection
                        print(f"Frame {self.frame_count:4d}: 🔍 No green objects detected")
                
                # Small delay for reasonable frame rate
                time.sleep(0.02)  # ~50 FPS
                
        except KeyboardInterrupt:
            print("\n⏹️ Live view stopped by user")
        
        # Final screenshot
        if 'display_frame' in locals():
            final_filename = f'live_camera_final_{int(time.time())}.jpg'
            cv2.imwrite(final_filename, display_frame)
            print(f"📸 Final screenshot: {final_filename}")
        
        # Summary
        elapsed = time.time() - start_time
        fps = self.frame_count / elapsed
        print(f"\n📊 Session Summary:")
        print(f"   Duration: {elapsed:.1f} seconds")
        print(f"   Frames: {self.frame_count}")
        print(f"   Average FPS: {fps:.1f}")
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up camera"""
        self.camera.release()
        print("✅ Camera released")

def main():
    print("=" * 50)
    print("🎥 LIVE CAMERA TEST - NO MOTOR CONTROL")
    print("=" * 50)
    
    viewer = LiveCameraViewer()
    
    # Run for 60 seconds (you can change this)
    viewer.start_live_view(60)
    
    print("\n✅ Live camera test completed!")
    print("Check the saved .jpg files to see the live detection results!")

if __name__ == "__main__":
    main()