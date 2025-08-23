#!/usr/bin/env python3
"""Simple web-based camera viewer"""

from flask import Flask, render_template_string, Response
import cv2
import time
import threading
from camera_test import LibcameraCapture
from color_detection_test import ColorDetector

app = Flask(__name__)

class WebCameraViewer:
    def __init__(self):
        self.camera = LibcameraCapture(640, 480)
        self.detector = ColorDetector()
        self.frame_count = 0
        self.latest_frame = None
        self.lock = threading.Lock()
        
    def generate_frames(self):
        """Generate frames for streaming"""
        while True:
            ret, frame = self.camera.read()
            if not ret:
                continue
                
            self.frame_count += 1
            
            # Detect green objects
            targets, mask = self.detector.detect_green(frame)
            
            # Draw overlays
            display_frame = self.draw_overlay(frame, targets)
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', display_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.1)  # ~10 FPS
    
    def draw_overlay(self, frame, targets):
        """Draw detection overlay"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Draw crosshair
        cv2.line(display_frame, (center_x - 30, center_y), 
                (center_x + 30, center_y), (255, 255, 255), 2)
        cv2.line(display_frame, (center_x, center_y - 30), 
                (center_x, center_y + 30), (255, 255, 255), 2)
        cv2.circle(display_frame, (center_x, center_y), 50, (255, 255, 255), 1)
        
        # Draw targets
        for i, target in enumerate(targets):
            center = target['center']
            bbox = target['bbox']
            x, y, w, h = bbox
            
            color = (0, 255, 0)
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            cv2.circle(display_frame, center, 8, color, -1)
            cv2.putText(display_frame, f"GREEN #{i+1}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Status info
        cv2.putText(display_frame, f"Frame: {self.frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if targets:
            cv2.putText(display_frame, f"Targets: {len(targets)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "No green objects", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 1)
        
        return display_frame

# Global viewer instance
viewer = WebCameraViewer()

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>🎥 Live Camera View</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #1a1a1a; 
            color: white;
            text-align: center;
        }
        .container { 
            max-width: 800px; 
            margin: 0 auto; 
        }
        .video-container {
            border: 3px solid #00ff00;
            border-radius: 10px;
            display: inline-block;
            margin: 20px 0;
        }
        img { 
            max-width: 100%; 
            height: auto; 
            display: block;
        }
        .status {
            background-color: #333;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .controls {
            margin: 20px 0;
        }
        .btn {
            background-color: #00ff00;
            color: black;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .btn:hover {
            background-color: #00cc00;
        }
    </style>
    <script>
        function refreshPage() {
            location.reload();
        }
        
        function takeScreenshot() {
            // This would need backend implementation
            alert('Screenshot feature coming soon!');
        }
        
        // Auto-refresh every 30 seconds
        setTimeout(refreshPage, 30000);
    </script>
</head>
<body>
    <div class="container">
        <h1>🎥 Water Gun Camera - Live View</h1>
        
        <div class="status">
            <h3>📵 MOTORS DISABLED - SAFE MODE</h3>
            <p>Live camera feed with green object detection</p>
        </div>
        
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" alt="Live Camera Feed">
        </div>
        
        <div class="controls">
            <button class="btn" onclick="refreshPage()">🔄 Refresh</button>
            <button class="btn" onclick="takeScreenshot()">📸 Screenshot</button>
        </div>
        
        <div class="status">
            <h4>What You're Seeing:</h4>
            <ul style="text-align: left; display: inline-block;">
                <li>🎯 White crosshair shows center point</li>
                <li>⭕ White circle shows "fire zone"</li>
                <li>🟢 Green boxes highlight detected objects</li>
                <li>📊 Frame counter and target count</li>
                <li>🔄 Page auto-refreshes every 30 seconds</li>
            </ul>
        </div>
        
        <p><em>Move the camera and hold up green objects to see live detection!</em></p>
    </div>
</body>
</html>
    ''')

@app.route('/video_feed')
def video_feed():
    return Response(viewer.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🌐 Starting Web Camera Viewer...")
    print("📵 NO MOTOR CONTROL - Safe for testing!")
    print(f"\n🔗 Open in your browser:")
    print(f"   http://192.168.0.3:8080")
    print(f"   http://localhost:8080")
    print("\n⏹️  Press Ctrl+C to stop")
    
    try:
        app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n👋 Web camera viewer stopped")
    finally:
        viewer.camera.release()
        print("✅ Camera released")