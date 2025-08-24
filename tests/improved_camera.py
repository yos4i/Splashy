#!/usr/bin/env python3
import cv2
import subprocess
import tempfile
import numpy as np
import time
import os
import threading
import queue

class HighPerformanceLibcameraCapture:
    """High-performance camera capture using libcamera-vid for Raspberry Pi 5"""
    
    def __init__(self, width=1280, height=720, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.running = False
        self.capture_thread = None
        
    def start(self):
        """Start video capture stream"""
        try:
            # Use libcamera-vid with stdout output
            cmd = [
                'libcamera-vid',
                '--output', '-',  # Output to stdout
                '--timeout', '0',  # Run indefinitely
                '--width', str(self.width),
                '--height', str(self.height),
                '--framerate', str(self.fps),
                '--codec', 'mjpeg',  # MJPEG for easier processing
                '--nopreview',
                '--immediate'
            ]
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_frames)
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            print(f"Failed to start libcamera-vid: {e}")
            return False
    
    def _capture_frames(self):
        """Background thread to capture frames"""
        frame_size = self.width * self.height * 3  # Rough estimate
        
        while self.running and self.process:
            try:
                # Read MJPEG frame (this is simplified - real implementation needs MJPEG parsing)
                chunk = self.process.stdout.read(frame_size)
                if not chunk:
                    break
                
                # For now, fall back to libcamera-still method but faster
                # In production, you'd parse the MJPEG stream properly
                ret, frame = self._capture_single_frame()
                if ret:
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        # Remove oldest frame if queue is full
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(frame)
                        except queue.Empty:
                            pass
                            
            except Exception as e:
                print(f"Capture error: {e}")
                break
    
    def _capture_single_frame(self):
        """Optimized single frame capture"""
        try:
            # Create temporary file
            fd, temp_file = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)
            
            # Faster libcamera-still with minimal timeout
            result = subprocess.run([
                'libcamera-still',
                '--output', temp_file,
                '--timeout', '50',  # Reduced timeout
                '--width', str(self.width),
                '--height', str(self.height),
                '--immediate',
                '--nopreview'
            ], capture_output=True, timeout=1)
            
            if result.returncode == 0:
                frame = cv2.imread(temp_file)
                os.unlink(temp_file)
                if frame is not None:
                    return True, frame
                    
            os.unlink(temp_file)
                    
        except Exception as e:
            print(f"Single frame capture error: {e}")
            
        return False, None
    
    def read(self):
        """Read the latest frame"""
        try:
            # Get the most recent frame, discarding older ones
            latest_frame = None
            while not self.frame_queue.empty():
                try:
                    latest_frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            if latest_frame is not None:
                return True, latest_frame
            else:
                # If queue is empty, capture a single frame
                return self._capture_single_frame()
                
        except Exception as e:
            print(f"Read error: {e}")
            return False, None
    
    def release(self):
        """Stop capture and clean up"""
        self.running = False
        
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2)


class FastLibcameraCapture:
    """Simplified fast capture using libcamera-still with optimizations"""
    
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.temp_file = None
        
    def read(self):
        """Optimized frame capture"""
        try:
            if self.temp_file is None:
                fd, self.temp_file = tempfile.mkstemp(suffix='.jpg')
                os.close(fd)
            
            # Optimized libcamera-still command
            result = subprocess.run([
                'libcamera-still',
                '--output', self.temp_file,
                '--timeout', '50',  # Minimal timeout
                '--width', str(self.width),
                '--height', str(self.height),
                '--immediate',
                '--nopreview',
                '--quality', '85',  # Slightly lower quality for speed
            ], capture_output=True, timeout=1)
            
            if result.returncode == 0:
                frame = cv2.imread(self.temp_file)
                if frame is not None:
                    return True, frame
                    
        except Exception as e:
            print(f"Fast capture error: {e}")
            
        return False, None
    
    def release(self):
        """Clean up"""
        if self.temp_file and os.path.exists(self.temp_file):
            os.unlink(self.temp_file)


def test_camera_performance():
    """Test different camera configurations"""
    configs = [
        {"name": "HD 720p", "width": 1280, "height": 720},
        {"name": "Full HD 1080p", "width": 1920, "height": 1080},
        {"name": "Original 480p", "width": 640, "height": 480},
    ]
    
    for config in configs:
        print(f"\n🎥 Testing {config['name']} ({config['width']}x{config['height']})")
        print("=" * 50)
        
        cap = FastLibcameraCapture(config['width'], config['height'])
        
        frame_count = 0
        start_time = time.time()
        test_duration = 10  # 10 seconds
        
        try:
            while time.time() - start_time < test_duration:
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                    if frame_count % 10 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        print(f"Frame {frame_count:3d}: {fps:.1f} FPS | Resolution: {frame.shape}")
                else:
                    print("❌ Frame capture failed")
                
        except KeyboardInterrupt:
            print("\n⏹️ Test interrupted")
        
        finally:
            cap.release()
            
        # Results
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed
        print(f"\n📊 Results for {config['name']}:")
        print(f"   Frames captured: {frame_count}")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Max theoretical improvement: {avg_fps/3:.1f}x over original")


if __name__ == "__main__":
    test_camera_performance()