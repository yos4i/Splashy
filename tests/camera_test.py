#!/usr/bin/env python3
import cv2
import subprocess
import tempfile
import numpy as np
import time
import os

class LibcameraCapture:
    """Camera capture class using libcamera for Raspberry Pi 5"""
    
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.temp_file = None
        
    def read(self):
        """Capture a frame using libcamera-still"""
        try:
            # Create temporary file
            if self.temp_file is None:
                fd, self.temp_file = tempfile.mkstemp(suffix='.jpg')
                os.close(fd)
            
            # Capture frame with libcamera
            result = subprocess.run([
                'libcamera-still',
                '--output', self.temp_file,
                '--timeout', '100',  # Quick capture
                '--width', str(self.width),
                '--height', str(self.height),
                '--immediate',  # Skip preview
                '--nopreview'   # No preview window
            ], capture_output=True, timeout=2)
            
            if result.returncode == 0:
                # Read image with OpenCV
                frame = cv2.imread(self.temp_file)
                if frame is not None:
                    return True, frame
                    
        except Exception as e:
            print(f"Camera capture error: {e}")
            
        return False, None
    
    def release(self):
        """Clean up temporary file"""
        if self.temp_file and os.path.exists(self.temp_file):
            os.unlink(self.temp_file)

def test_camera():
    """Test the camera capture"""
    print("Testing Pi 5 camera with libcamera...")
    
    cap = LibcameraCapture(1280, 720)
    
    try:
        for i in range(5):
            print(f"Capture {i+1}/5...")
            ret, frame = cap.read()
            
            if ret:
                print(f"✓ SUCCESS: Frame captured {frame.shape}")
                
                # Simple face detection test
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                print(f"  Converted to grayscale: {gray.shape}")
                
            else:
                print("✗ FAILED to capture frame")
            
            time.sleep(0.5)
                
    finally:
        cap.release()
    
    print("Camera test completed!")

if __name__ == "__main__":
    test_camera()