#!/usr/bin/env python3
import os
import cv2
import time
import json
import threading
import logging
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
import base64
import numpy as np
from datetime import datetime

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tests'))

from water_gun_system import WaterGunController, TrackingConfig
from servo_controller import ServoController, ServoConfig
from device_controller import DeviceController, DeviceConfig

app = Flask(__name__)
app.config['SECRET_KEY'] = 'water_gun_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

class WebInterface:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize system components
        self.tracking_config = TrackingConfig()
        self.servo_config = ServoConfig()
        self.device_config = DeviceConfig()
        
        self.water_gun = WaterGunController(self.tracking_config)
        self.servo_controller = ServoController(self.servo_config)
        self.device_controller = DeviceController(self.device_config)
        
        # Web interface state
        self.connected_clients = 0
        self.streaming_active = False
        self.log_messages = []
        self.max_log_messages = 100
        
        # Performance monitoring
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize camera
        if not self.water_gun.initialize_camera():
            self.logger.error("Failed to initialize camera")
    
    def start_system(self):
        """Start all system components"""
        try:
            self.servo_controller.start_control_loop()
            self.device_controller.start_control_loop()
            
            # Connect controllers to water gun system
            self.water_gun.servo_command_queue = self.servo_controller.command_queue
            self.water_gun.device_command_queue = self.device_controller.command_queue
            
            self.logger.info("Water gun system started successfully")
            self.add_log_message("System started successfully", "INFO")
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            self.add_log_message(f"System startup failed: {e}", "ERROR")
    
    def stop_system(self):
        """Stop all system components"""
        try:
            self.water_gun.cleanup()
            self.servo_controller.cleanup()
            self.device_controller.cleanup()
            
            self.logger.info("Water gun system stopped")
            self.add_log_message("System stopped", "INFO")
            
        except Exception as e:
            self.logger.error(f"Error stopping system: {e}")
    
    def add_log_message(self, message: str, level: str = "INFO"):
        """Add a log message to the web interface"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        
        self.log_messages.append(log_entry)
        
        # Limit log message count
        if len(self.log_messages) > self.max_log_messages:
            self.log_messages.pop(0)
        
        # Emit to connected clients
        socketio.emit('log_message', log_entry)
    
    def generate_frames(self):
        """Generate video frames for streaming"""
        while self.streaming_active:
            try:
                frame = self.water_gun.capture_frame()
                if frame is None:
                    continue
                
                # Process frame through water gun system
                processed_frame = self.water_gun.process_frame(frame)
                
                # Add FPS counter
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    fps = self.frame_count / elapsed
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                              (10, processed_frame.shape[0] - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Encode frame
                ret, buffer = cv2.imencode('.jpg', processed_frame, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                self.logger.error(f"Error generating frame: {e}")
                break

# Global web interface instance
web_interface = WebInterface()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    web_interface.streaming_active = True
    return Response(web_interface.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    """Get system status"""
    try:
        pan_deg, tilt_deg = web_interface.servo_controller.get_position_degrees()
        
        status = {
            'system': {
                'tracking_enabled': web_interface.water_gun.tracking_enabled,
                'auto_fire_enabled': web_interface.water_gun.auto_fire_enabled,
                'connected_clients': web_interface.connected_clients
            },
            'servos': {
                'pan_degrees': round(pan_deg, 1),
                'tilt_degrees': round(tilt_deg, 1),
                'is_moving': web_interface.servo_controller.is_moving
            },
            'devices': web_interface.device_controller.get_device_status(),
            'timestamp': time.time()
        }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/fire', methods=['POST'])
def fire_water():
    """Manual water firing"""
    try:
        data = request.get_json()
        duration = data.get('duration', 1.0)
        
        success = web_interface.device_controller.fire_pump(duration)
        
        if success:
            web_interface.add_log_message(f"Water fired manually ({duration}s)", "INFO")
            return jsonify({'success': True, 'message': f'Water fired for {duration}s'})
        else:
            return jsonify({'success': False, 'message': 'Fire command failed'}), 400
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/servo/control', methods=['POST'])
def control_servo():
    """Manual servo control"""
    try:
        data = request.get_json()
        pan = data.get('pan')
        tilt = data.get('tilt')
        
        if pan is not None and tilt is not None:
            web_interface.water_gun.manual_servo_control(pan, tilt)
            web_interface.add_log_message(f"Manual servo: Pan={pan}°, Tilt={tilt}°", "INFO")
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Pan and tilt values required'}), 400
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/servo/center', methods=['POST'])
def center_servo():
    """Center servo position"""
    try:
        web_interface.servo_controller.move_to_center()
        web_interface.add_log_message("Servos centered", "INFO")
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/tracking/toggle', methods=['POST'])
def toggle_tracking():
    """Toggle tracking mode"""
    try:
        current_state = web_interface.water_gun.tracking_enabled
        web_interface.water_gun.set_tracking_enabled(not current_state)
        
        new_state = "enabled" if not current_state else "disabled"
        web_interface.add_log_message(f"Tracking {new_state}", "INFO")
        
        return jsonify({'success': True, 'tracking_enabled': not current_state})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/autofire/toggle', methods=['POST'])
def toggle_autofire():
    """Toggle auto-fire mode"""
    try:
        current_state = web_interface.water_gun.auto_fire_enabled
        web_interface.water_gun.set_auto_fire_enabled(not current_state)
        
        new_state = "enabled" if not current_state else "disabled"
        web_interface.add_log_message(f"Auto-fire {new_state}", "INFO")
        
        return jsonify({'success': True, 'auto_fire_enabled': not current_state})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ir/toggle', methods=['POST'])
def toggle_ir():
    """Toggle IR illuminator"""
    try:
        web_interface.device_controller.toggle_ir_led()
        web_interface.add_log_message("IR illuminator toggled", "INFO")
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/emergency_stop', methods=['POST'])
def emergency_stop():
    """Emergency stop all systems"""
    try:
        web_interface.device_controller.emergency_stop()
        web_interface.water_gun.set_tracking_enabled(False)
        web_interface.water_gun.set_auto_fire_enabled(False)
        
        web_interface.add_log_message("EMERGENCY STOP ACTIVATED", "WARNING")
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    web_interface.connected_clients += 1
    web_interface.add_log_message(f"Client connected (total: {web_interface.connected_clients})", "INFO")
    
    # Send current status to new client
    status = {
        'tracking_enabled': web_interface.water_gun.tracking_enabled,
        'auto_fire_enabled': web_interface.water_gun.auto_fire_enabled,
        'log_messages': web_interface.log_messages[-10:]  # Last 10 messages
    }
    emit('status_update', status)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    web_interface.connected_clients = max(0, web_interface.connected_clients - 1)
    web_interface.add_log_message(f"Client disconnected (total: {web_interface.connected_clients})", "INFO")

@socketio.on('manual_control')
def handle_manual_control(data):
    """Handle manual control commands from websocket"""
    try:
        command = data.get('command')
        
        if command == 'fire':
            duration = data.get('duration', 1.0)
            web_interface.device_controller.fire_pump(duration)
        
        elif command == 'servo_move':
            pan = data.get('pan', 0)
            tilt = data.get('tilt', 0)
            web_interface.water_gun.manual_servo_control(pan, tilt)
        
        elif command == 'toggle_tracking':
            current_state = web_interface.water_gun.tracking_enabled
            web_interface.water_gun.set_tracking_enabled(not current_state)
        
        elif command == 'toggle_autofire':
            current_state = web_interface.water_gun.auto_fire_enabled
            web_interface.water_gun.set_auto_fire_enabled(not current_state)
        
        elif command == 'center_servos':
            web_interface.servo_controller.move_to_center()
        
        emit('command_acknowledged', {'command': command, 'success': True})
        
    except Exception as e:
        emit('command_acknowledged', {'command': command, 'success': False, 'error': str(e)})

# Status broadcasting thread
def status_broadcaster():
    """Periodically broadcast status updates"""
    while True:
        try:
            if web_interface.connected_clients > 0:
                pan_deg, tilt_deg = web_interface.servo_controller.get_position_degrees()
                
                status = {
                    'system': {
                        'tracking_enabled': web_interface.water_gun.tracking_enabled,
                        'auto_fire_enabled': web_interface.water_gun.auto_fire_enabled
                    },
                    'servos': {
                        'pan_degrees': round(pan_deg, 1),
                        'tilt_degrees': round(tilt_deg, 1)
                    },
                    'devices': web_interface.device_controller.get_device_status()
                }
                
                socketio.emit('status_update', status)
            
            time.sleep(1)  # Update every second
            
        except Exception as e:
            logging.error(f"Error in status broadcaster: {e}")
            time.sleep(5)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Start system components
        web_interface.start_system()
        
        # Start status broadcasting thread
        status_thread = threading.Thread(target=status_broadcaster, daemon=True)
        status_thread.start()
        
        print("Starting Water Gun Web Interface...")
        print("Access the dashboard at: http://localhost:5000")
        
        # Run Flask app
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        web_interface.stop_system()