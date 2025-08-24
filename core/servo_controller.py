#!/usr/bin/env python3
import time
import threading
import logging
import queue
from typing import Optional
from dataclasses import dataclass

try:
    from gpiozero import Servo, Device
    from gpiozero.pins.pigpio import PiGPIOFactory
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logging.warning("GPIO libraries not available - running in simulation mode")

@dataclass
class ServoConfig:
    # GPIO pins - User's actual wiring
    pan_pin: int = 18   # GPIO18 - Physical pin 12
    tilt_pin: int = 23  # GPIO23 - Physical pin 16
    
    # Servo parameters
    min_pulse_width: float = 0.5e-3   # 0.5ms
    max_pulse_width: float = 2.5e-3   # 2.5ms
    frame_width: float = 20e-3        # 20ms (50Hz)
    
    # Movement limits (in servo range -1 to 1)
    pan_min: float = -1.0
    pan_max: float = 1.0
    tilt_min: float = -1.0
    tilt_max: float = 1.0
    
    # Movement parameters
    max_speed: float = 2.0  # units per second
    acceleration: float = 4.0  # units per second squared
    settle_time: float = 0.1  # seconds to wait after movement

class ServoController:
    def __init__(self, config: ServoConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize GPIO factory for better PWM performance
        if GPIO_AVAILABLE:
            try:
                Device.pin_factory = PiGPIOFactory()
                self.logger.info("Using pigpio for servo control")
            except Exception as e:
                self.logger.warning(f"Failed to initialize pigpio: {e}")
        
        # Initialize servos
        self.pan_servo = None
        self.tilt_servo = None
        self.initialize_servos()
        
        # Current positions (in servo range -1 to 1)
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.target_pan = 0.0
        self.target_tilt = 0.0
        
        # Movement state
        self.is_moving = False
        self.movement_thread = None
        self.stop_movement = threading.Event()
        
        # Command queue
        self.command_queue = queue.Queue()
        
        # Control thread
        self.control_thread = None
        self.running = False
    
    def initialize_servos(self):
        if not GPIO_AVAILABLE:
            self.logger.info("Servo controller running in simulation mode")
            return
        
        try:
            self.pan_servo = Servo(
                self.config.pan_pin,
                min_pulse_width=self.config.min_pulse_width,
                max_pulse_width=self.config.max_pulse_width,
                frame_width=self.config.frame_width
            )
            
            self.tilt_servo = Servo(
                self.config.tilt_pin,
                min_pulse_width=self.config.min_pulse_width,
                max_pulse_width=self.config.max_pulse_width,
                frame_width=self.config.frame_width
            )
            
            # Move to center position
            self.pan_servo.value = 0
            self.tilt_servo.value = 0
            
            self.logger.info("Servos initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize servos: {e}")
            self.pan_servo = None
            self.tilt_servo = None
    
    def degrees_to_servo_value(self, degrees: float, servo_type: str = 'pan') -> float:
        """Convert degrees to servo value (-1 to 1)"""
        if servo_type == 'pan':
            # Pan: -90° to +90° maps to -1 to +1
            return max(min(degrees / 90.0, 1.0), -1.0)
        else:
            # Tilt: -30° to +60° maps to -1 to +1
            # Normalize to -30 to +60 range first
            normalized = (degrees + 30) / 90.0  # 0 to 1
            return max(min(normalized * 2.0 - 1.0, 1.0), -1.0)  # -1 to 1
    
    def servo_value_to_degrees(self, value: float, servo_type: str = 'pan') -> float:
        """Convert servo value (-1 to 1) to degrees"""
        if servo_type == 'pan':
            return value * 90.0
        else:
            # Convert from -1 to 1 back to -30 to +60
            normalized = (value + 1.0) / 2.0  # 0 to 1
            return normalized * 90.0 - 30.0
    
    def set_position_degrees(self, pan_degrees: float, tilt_degrees: float):
        """Set servo positions in degrees"""
        pan_value = self.degrees_to_servo_value(pan_degrees, 'pan')
        tilt_value = self.degrees_to_servo_value(tilt_degrees, 'tilt')
        
        self.set_position(pan_value, tilt_value)
    
    def set_position(self, pan: float, tilt: float):
        """Set target servo positions (-1 to 1 range)"""
        # Apply limits
        pan = max(min(pan, self.config.pan_max), self.config.pan_min)
        tilt = max(min(tilt, self.config.tilt_max), self.config.tilt_min)
        
        command = {
            'action': 'move',
            'pan': pan,
            'tilt': tilt,
            'timestamp': time.time()
        }
        
        try:
            self.command_queue.put_nowait(command)
        except queue.Full:
            self.logger.warning("Command queue full, dropping command")
    
    def get_position_degrees(self) -> tuple:
        """Get current position in degrees"""
        pan_deg = self.servo_value_to_degrees(self.current_pan, 'pan')
        tilt_deg = self.servo_value_to_degrees(self.current_tilt, 'tilt')
        return pan_deg, tilt_deg
    
    def move_to_center(self):
        """Move servos to center position"""
        self.set_position(0.0, 0.0)
    
    def smooth_move(self, target_pan: float, target_tilt: float):
        """Smoothly move servos to target position"""
        if self.is_moving:
            self.stop_movement.set()
            if self.movement_thread:
                self.movement_thread.join(timeout=1.0)
        
        self.target_pan = target_pan
        self.target_tilt = target_tilt
        self.stop_movement.clear()
        
        self.movement_thread = threading.Thread(
            target=self._smooth_movement_worker,
            args=(target_pan, target_tilt)
        )
        self.movement_thread.start()
    
    def _smooth_movement_worker(self, target_pan: float, target_tilt: float):
        """Worker thread for smooth servo movement"""
        self.is_moving = True
        start_pan = self.current_pan
        start_tilt = self.current_tilt
        
        # Calculate movement distances
        pan_distance = target_pan - start_pan
        tilt_distance = target_tilt - start_tilt
        total_distance = max(abs(pan_distance), abs(tilt_distance))
        
        if total_distance == 0:
            self.is_moving = False
            return
        
        # Calculate movement time based on max speed
        movement_time = total_distance / self.config.max_speed
        
        # Acceleration/deceleration phases
        accel_time = min(movement_time / 4, self.config.max_speed / self.config.acceleration)
        
        start_time = time.time()
        last_update = start_time
        
        while not self.stop_movement.is_set():
            current_time = time.time()
            elapsed = current_time - start_time
            
            if elapsed >= movement_time:
                # Movement complete
                self.current_pan = target_pan
                self.current_tilt = target_tilt
                self._update_servo_positions()
                break
            
            # Calculate progress with smooth acceleration/deceleration
            if elapsed < accel_time:
                # Acceleration phase
                progress = 0.5 * self.config.acceleration * elapsed * elapsed / total_distance
            elif elapsed > movement_time - accel_time:
                # Deceleration phase
                decel_elapsed = movement_time - elapsed
                progress = 1.0 - 0.5 * self.config.acceleration * decel_elapsed * decel_elapsed / total_distance
            else:
                # Constant speed phase
                accel_distance = 0.5 * self.config.acceleration * accel_time * accel_time / total_distance
                const_elapsed = elapsed - accel_time
                const_progress = self.config.max_speed * const_elapsed / total_distance
                progress = accel_distance + const_progress
            
            progress = min(progress, 1.0)
            
            # Update current positions
            self.current_pan = start_pan + pan_distance * progress
            self.current_tilt = start_tilt + tilt_distance * progress
            
            # Update servo positions (limit frequency)
            if current_time - last_update >= 0.02:  # 50Hz max update rate
                self._update_servo_positions()
                last_update = current_time
            
            time.sleep(0.01)
        
        self.is_moving = False
    
    def _update_servo_positions(self):
        """Update physical servo positions"""
        if not GPIO_AVAILABLE or not self.pan_servo or not self.tilt_servo:
            return
        
        try:
            self.pan_servo.value = self.current_pan
            self.tilt_servo.value = self.current_tilt
        except Exception as e:
            self.logger.error(f"Failed to update servo positions: {e}")
    
    def start_control_loop(self):
        """Start the servo control loop"""
        if self.running:
            return
        
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop_worker)
        self.control_thread.start()
        self.logger.info("Servo control loop started")
    
    def _control_loop_worker(self):
        """Main control loop worker"""
        while self.running:
            try:
                # Process commands from queue
                try:
                    command = self.command_queue.get(timeout=0.1)
                    self._process_command(command)
                except queue.Empty:
                    continue
                
            except Exception as e:
                self.logger.error(f"Error in control loop: {e}")
                time.sleep(0.1)
    
    def _process_command(self, command: dict):
        """Process a servo command"""
        action = command.get('action')
        
        if action == 'move':
            pan = command.get('pan', self.current_pan)
            tilt = command.get('tilt', self.current_tilt)
            self.smooth_move(pan, tilt)
        
        elif action == 'stop':
            if self.is_moving:
                self.stop_movement.set()
        
        elif action == 'center':
            self.smooth_move(0.0, 0.0)
    
    def stop_control_loop(self):
        """Stop the servo control loop"""
        self.running = False
        
        if self.is_moving:
            self.stop_movement.set()
        
        if self.control_thread:
            self.control_thread.join(timeout=2.0)
        
        if self.movement_thread:
            self.movement_thread.join(timeout=2.0)
        
        self.logger.info("Servo control loop stopped")
    
    def cleanup(self):
        """Clean up servo controller"""
        self.stop_control_loop()
        
        if GPIO_AVAILABLE and self.pan_servo and self.tilt_servo:
            try:
                # Return to center before cleanup
                self.pan_servo.value = 0
                self.tilt_servo.value = 0
                time.sleep(0.5)
                
                self.pan_servo.close()
                self.tilt_servo.close()
            except Exception as e:
                self.logger.error(f"Error during servo cleanup: {e}")

class ServoTester:
    """Test utility for servo controller"""
    
    def __init__(self, servo_controller: ServoController):
        self.servo = servo_controller
    
    def test_range_of_motion(self):
        """Test full range of motion"""
        print("Testing servo range of motion...")
        
        positions = [
            (0, 0, "Center"),
            (1, 0, "Pan Right"),
            (-1, 0, "Pan Left"),
            (0, 1, "Tilt Up"),
            (0, -1, "Tilt Down"),
            (1, 1, "Top Right"),
            (-1, 1, "Top Left"),
            (1, -1, "Bottom Right"),
            (-1, -1, "Bottom Left"),
            (0, 0, "Return to Center")
        ]
        
        for pan, tilt, description in positions:
            print(f"Moving to: {description} (Pan: {pan}, Tilt: {tilt})")
            self.servo.set_position(pan, tilt)
            time.sleep(2)
    
    def test_smooth_movement(self):
        """Test smooth movement patterns"""
        print("Testing smooth movement patterns...")
        
        # Figure-8 pattern
        import math
        for i in range(100):
            t = i * 0.1
            pan = 0.8 * math.sin(t)
            tilt = 0.6 * math.sin(2 * t)
            self.servo.set_position(pan, tilt)
            time.sleep(0.1)
        
        # Return to center
        self.servo.move_to_center()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = ServoConfig()
    servo_controller = ServoController(config)
    
    try:
        servo_controller.start_control_loop()
        
        print("Servo Controller Test")
        print("Commands:")
        print("  c - Center")
        print("  t - Test range of motion")
        print("  s - Test smooth movement")
        print("  arrow keys - Manual control")
        print("  q - Quit")
        
        servo_controller.move_to_center()
        time.sleep(1)
        
        while True:
            command = input("Enter command: ").strip().lower()
            
            if command == 'q':
                break
            elif command == 'c':
                servo_controller.move_to_center()
            elif command == 't':
                tester = ServoTester(servo_controller)
                tester.test_range_of_motion()
            elif command == 's':
                tester = ServoTester(servo_controller)
                tester.test_smooth_movement()
            elif command in ['w', 'up']:
                servo_controller.set_position(servo_controller.current_pan, 
                                           min(servo_controller.current_tilt + 0.1, 1.0))
            elif command in ['s', 'down']:
                servo_controller.set_position(servo_controller.current_pan, 
                                           max(servo_controller.current_tilt - 0.1, -1.0))
            elif command in ['a', 'left']:
                servo_controller.set_position(max(servo_controller.current_pan - 0.1, -1.0), 
                                           servo_controller.current_tilt)
            elif command in ['d', 'right']:
                servo_controller.set_position(min(servo_controller.current_pan + 0.1, 1.0), 
                                           servo_controller.current_tilt)
    
    finally:
        servo_controller.cleanup()