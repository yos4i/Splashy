#!/usr/bin/env python3
import cv2
import time
import numpy as np

class OpenCVCameraTest:
    """Test OpenCV camera with different backends"""
    
    def test_camera_backends(self):
        """Test different camera backends"""
        backends = [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_GSTREAMER, "GStreamer"), 
            (cv2.CAP_FFMPEG, "FFmpeg"),
            (cv2.CAP_ANY, "Any")
        ]
        
        for backend_id, backend_name in backends:
            print(f"\n🎥 Testing {backend_name} backend...")
            
            try:
                cap = cv2.VideoCapture(0, backend_id)
                
                if not cap.isOpened():
                    print(f"❌ Failed to open camera with {backend_name}")
                    continue
                
                # Try to set higher resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Check actual values
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"✅ {backend_name} opened successfully")
                print(f"   Resolution: {width}x{height}")
                print(f"   FPS: {fps}")
                
                # Test capture
                ret, frame = cap.read()
                if ret:
                    print(f"   Frame captured: {frame.shape}")
                    
                    # Test multiple frames for FPS
                    frame_count = 0
                    start_time = time.time()
                    test_duration = 5
                    
                    while time.time() - start_time < test_duration:
                        ret, frame = cap.read()
                        if ret:
                            frame_count += 1
                        else:
                            break
                    
                    elapsed = time.time() - start_time
                    actual_fps = frame_count / elapsed
                    print(f"   Actual FPS: {actual_fps:.1f}")
                else:
                    print(f"❌ Failed to capture frame with {backend_name}")
                
                cap.release()
                
            except Exception as e:
                print(f"❌ Error with {backend_name}: {e}")

    def test_different_resolutions(self):
        """Test different resolutions with the best working backend"""
        print(f"\n🎥 Testing different resolutions with OpenCV...")
        
        resolutions = [
            (640, 480, "VGA"),
            (1280, 720, "HD 720p"),
            (1920, 1080, "Full HD 1080p"),
            (3840, 2160, "4K UHD")
        ]
        
        for width, height, name in resolutions:
            print(f"\n📐 Testing {name} ({width}x{height})")
            
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("❌ Failed to open camera")
                    continue
                
                # Set resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Check what we actually got
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"   Requested: {width}x{height}")
                print(f"   Actual: {actual_width}x{actual_height}")
                print(f"   FPS setting: {actual_fps}")
                
                # Test capture
                ret, frame = cap.read()
                if ret:
                    print(f"   ✅ Frame captured: {frame.shape}")
                    
                    # Quick FPS test
                    start_time = time.time()
                    frame_count = 0
                    
                    for _ in range(30):  # Capture 30 frames
                        ret, frame = cap.read()
                        if ret:
                            frame_count += 1
                        else:
                            break
                    
                    elapsed = time.time() - start_time
                    if frame_count > 0:
                        fps = frame_count / elapsed
                        print(f"   📊 Measured FPS: {fps:.1f}")
                    
                else:
                    print(f"   ❌ Failed to capture frame")
                
                cap.release()
                
            except Exception as e:
                print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🎥 OPENCV CAMERA PERFORMANCE TEST")
    print("=" * 60)
    
    tester = OpenCVCameraTest()
    
    # Test different backends
    tester.test_camera_backends()
    
    # Test different resolutions
    tester.test_different_resolutions()
    
    print(f"\n✅ Camera test completed!")