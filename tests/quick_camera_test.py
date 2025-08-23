#!/usr/bin/env python3
"""Quick 10-second camera test"""

import time
from live_camera_test import LiveCameraViewer

def quick_test():
    print("🎥 QUICK CAMERA TEST - 10 seconds")
    print("=" * 40)
    
    viewer = LiveCameraViewer()
    viewer.start_live_view(10)  # Only 10 seconds

if __name__ == "__main__":
    quick_test()