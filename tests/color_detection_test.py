#!/usr/bin/env python3
import cv2
import numpy as np
import time
from camera_test import LibcameraCapture

class ColorDetector:
    """Detect specific colors in camera frames"""
    
    def __init__(self):
        # Green color range in HSV
        self.green_lower = np.array([40, 40, 40])   # Lower HSV threshold for green
        self.green_upper = np.array([80, 255, 255]) # Upper HSV threshold for green
        
        # Minimum area for detection (to filter noise)
        self.min_area = 500
        
    def detect_green(self, frame):
        """Detect green objects in frame"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for green color
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        green_targets = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                green_targets.append({
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'area': area,
                    'type': 'green'
                })
        
        # Sort by area (largest first)
        green_targets.sort(key=lambda x: x['area'], reverse=True)
        
        return green_targets, mask
    
    def draw_detections(self, frame, targets):
        """Draw detection results on frame"""
        result_frame = frame.copy()
        
        for i, target in enumerate(targets):
            center = target['center']
            bbox = target['bbox']
            x, y, w, h = bbox
            
            # Draw bounding box
            color = (0, 255, 0) if i == 0 else (0, 200, 0)  # Brightest green for largest target
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            cv2.circle(result_frame, center, 8, color, -1)
            
            # Add label
            label = f"GREEN #{i+1} ({target['area']}px)"
            cv2.putText(result_frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw crosshair at frame center
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        cv2.line(result_frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 1)
        cv2.line(result_frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 1)
        
        return result_frame

def test_color_detection():
    """Test green color detection"""
    print("Starting green color detection test...")
    print("Hold up something green in front of the camera!")
    
    # Initialize camera and detector
    camera = LibcameraCapture(640, 480)
    detector = ColorDetector()
    
    try:
        for i in range(20):  # Take 20 test shots
            print(f"\nCapture {i+1}/20...")
            
            # Capture frame
            ret, frame = camera.read()
            if not ret:
                print("Failed to capture frame")
                continue
            
            # Detect green objects
            targets, mask = detector.detect_green(frame)
            
            if targets:
                print(f"✓ Found {len(targets)} green objects!")
                for j, target in enumerate(targets):
                    center_x, center_y = target['center']
                    area = target['area']
                    print(f"  Target {j+1}: Center({center_x}, {center_y}), Area: {area}px")
                
                # Calculate offset from frame center
                frame_center_x = frame.shape[1] // 2
                frame_center_y = frame.shape[0] // 2
                largest_target = targets[0]
                target_x, target_y = largest_target['center']
                
                offset_x = target_x - frame_center_x
                offset_y = target_y - frame_center_y
                
                print(f"  Offset from center: X={offset_x}, Y={offset_y}")
                
                # Simulate servo commands
                if abs(offset_x) > 30 or abs(offset_y) > 30:
                    direction = ""
                    if offset_x > 30:
                        direction += "RIGHT "
                    elif offset_x < -30:
                        direction += "LEFT "
                    if offset_y > 30:
                        direction += "DOWN "
                    elif offset_y < -30:
                        direction += "UP "
                    
                    print(f"  → SERVO: Move {direction}")
                else:
                    print("  → TARGET CENTERED - READY TO FIRE!")
                
            else:
                print("✗ No green objects detected")
            
            time.sleep(1)  # 1 second between captures
            
    finally:
        camera.release()
    
    print("\nColor detection test completed!")

if __name__ == "__main__":
    test_color_detection()