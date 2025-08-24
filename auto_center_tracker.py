#!/usr/bin/env python3
"""
Auto-Center Pan/Tilt Tracker with Servo Power Management
Implements the full specification for target centering with power-off when idle.
"""
import cv2
import numpy as np
import time
import threading
import logging
import sys
import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tests'))

from camera_test import LibcameraCapture

# GPIO availability check
GPIO_AVAILABLE = False
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    print("✅ RPi.GPIO imported successfully")
except ImportError:
    print("❌ RPi.GPIO not available - simulation mode")

class TrackingState(Enum):
    """State machine for tracking system"""
    IDLE_OFF = "idle_off"           # No target, PWM detached, power cut
    TRACKING = "tracking"           # Target present, running control loop
    CENTERED_HOLD = "centered_hold" # Inside deadband, minimal updates

@dataclass
class TargetConfig:
    """Configuration for target detection and tracking"""
    conf_min: float = 0.5           # Minimum confidence threshold
    deadband_x: int = 12            # Horizontal deadband in pixels
    deadband_y: int = 12            # Vertical deadband in pixels
    center_hold_ms: int = 400       # Time to hold in deadband (ms)
    lost_ms: int = 800              # Time before target considered lost (ms)
    debounce_frames: int = 3        # Frames to confirm detection

@dataclass
class ControlConfig:
    """Configuration for control system"""
    # PID parameters
    kp_x: float = 0.015             # Pan proportional gain
    ki_x: float = 0.002             # Pan integral gain
    kd_x: float = 0.008             # Pan derivative gain
    kp_y: float = 0.012             # Tilt proportional gain  
    ki_y: float = 0.0015            # Tilt integral gain
    kd_y: float = 0.006             # Tilt derivative gain
    
    # Rate limiting
    max_step_deg: float = 5.0       # Maximum step per frame (degrees)
    max_slew_rate: float = 120.0    # Maximum slew rate (degrees/second)
    
    # Smoothing
    ema_alpha: float = 0.3          # Exponential moving average factor
    
    # Angle limits
    pan_min: float = 10.0           # Pan minimum angle (degrees)
    pan_max: float = 170.0          # Pan maximum angle (degrees)
    tilt_min: float = 20.0          # Tilt minimum angle (degrees)
    tilt_max: float = 140.0         # Tilt maximum angle (degrees)

@dataclass
class SafetyConfig:
    """Configuration for safety features"""
    thermal_limit_ms: int = 2000    # Time at mechanical limit before backing off (ms)
    backoff_deg: float = 3.0        # Degrees to back off from limit
    soft_start_duration: float = 0.3 # Soft start ramp duration (seconds)
    soft_start_max_rate: float = 60.0 # Max rate during soft start (deg/s)

class PIDController:
    """Enhanced PID controller with integral clamping"""
    
    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float = 10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.prev_derivative = 0.0  # For derivative filtering
        
    def update(self, error: float, dt: Optional[float] = None) -> float:
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 0.016  # Default to ~60Hz
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with clamping
        self.integral += error * dt
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)
        i_term = self.ki * self.integral
        
        # Derivative term with filtering
        derivative = (error - self.prev_error) / dt
        filtered_derivative = 0.7 * self.prev_derivative + 0.3 * derivative  # Low-pass filter
        d_term = self.kd * filtered_derivative
        
        output = p_term + i_term + d_term
        
        self.prev_error = error
        self.prev_derivative = filtered_derivative
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset PID controller state"""
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_derivative = 0.0
        self.last_time = time.time()

class AdvancedTargetDetector:
    """Enhanced target detection with confidence thresholding"""
    
    def __init__(self, config: TargetConfig):
        self.config = config
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        
        # Detection history for debouncing
        self.detection_history = []
        
        logging.info("Advanced target detector initialized")
    
    def detect_targets(self, frame: np.ndarray) -> List[Dict]:
        """Detect targets with confidence scoring and debouncing"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        targets = []
        
        # Face detection with confidence
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in faces:
            center_x = x + w // 2
            center_y = y + h // 2
            area = w * h
            
            # Simple confidence based on size and detection quality
            confidence = min(1.0, area / 10000.0)  # Normalize by expected face size
            
            if confidence >= self.config.conf_min:
                targets.append({
                    'center': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'area': area,
                    'confidence': confidence,
                    'type': 'face'
                })
        
        # Body detection if no high-confidence faces
        if not targets or max(t['confidence'] for t in targets) < 0.7:
            bodies = self.body_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(60, 80)
            )
            
            for (x, y, w, h) in bodies:
                center_x = x + w // 2
                center_y = y + h // 2
                area = w * h
                confidence = min(0.8, area / 20000.0)  # Bodies get lower max confidence
                
                if confidence >= self.config.conf_min:
                    targets.append({
                        'center': (center_x, center_y),
                        'bbox': (x, y, w, h),
                        'area': area,
                        'confidence': confidence,
                        'type': 'body'
                    })
        
        # Sort by confidence, then by area
        targets.sort(key=lambda t: (t['confidence'], t['area']), reverse=True)
        
        # Apply debouncing
        self._update_detection_history(len(targets) > 0)
        
        return targets if self._is_detection_stable() else []
    
    def _update_detection_history(self, has_detection: bool):
        """Update detection history for debouncing"""
        self.detection_history.append(has_detection)
        if len(self.detection_history) > self.config.debounce_frames:
            self.detection_history.pop(0)
    
    def _is_detection_stable(self) -> bool:
        """Check if detection is stable (debounced)"""
        if len(self.detection_history) < self.config.debounce_frames:
            return False
        return all(self.detection_history)

class PowerManagedServoController:
    """Servo controller with power management and state machine"""
    
    def __init__(self, control_config: ControlConfig, safety_config: SafetyConfig,
                 pan_pin: int = 12, tilt_pin: int = 13, power_pin: Optional[int] = None):
        self.control_config = control_config
        self.safety_config = safety_config
        self.pan_pin = pan_pin
        self.tilt_pin = tilt_pin
        self.power_pin = power_pin  # Optional GPIO pin to control servo power
        self.simulation_mode = not GPIO_AVAILABLE
        
        # Current angles in degrees
        self.current_pan = 90.0   # Center position
        self.current_tilt = 80.0  # Slightly up
        self.target_pan = self.current_pan
        self.target_tilt = self.current_tilt
        
        # Smoothed setpoints
        self.smooth_pan = self.current_pan
        self.smooth_tilt = self.current_tilt
        
        # PWM objects
        self.pan_pwm = None
        self.tilt_pwm = None
        
        # Power management
        self.servos_powered = False
        self.pwm_attached = False
        
        # Rate limiting
        self.last_update_time = time.time()
        
        # Safety tracking
        self.limit_start_time = {}  # Track time at limits
        
        # Soft start state
        self.soft_start_active = False
        self.soft_start_start_time = 0.0
        self.soft_start_start_angles = (0.0, 0.0)
        
        # Initialize GPIO
        self._initialize_gpio()
        
        logging.info(f"Power-managed servo controller initialized (Power pin: {power_pin})")
    
    def _initialize_gpio(self):
        """Initialize GPIO for servo control"""
        if self.simulation_mode:
            logging.info("Servo controller in simulation mode")
            return
        
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pan_pin, GPIO.OUT)
            GPIO.setup(self.tilt_pin, GPIO.OUT)
            
            if self.power_pin:
                GPIO.setup(self.power_pin, GPIO.OUT)
                GPIO.output(self.power_pin, GPIO.LOW)  # Start with power off
            
            logging.info(f"GPIO initialized: Pan={self.pan_pin}, Tilt={self.tilt_pin}, Power={self.power_pin}")
            
        except Exception as e:
            logging.error(f"Failed to initialize GPIO: {e}")
            self.simulation_mode = True
    
    def _angle_to_duty_cycle(self, angle: float) -> float:
        """Convert angle (0-180°) to PWM duty cycle (2.5-12.5%)"""
        # Standard servo: 0.5ms-2.5ms pulse width at 50Hz
        # 0.5ms = 2.5% duty, 2.5ms = 12.5% duty
        duty = 2.5 + (angle / 180.0) * 10.0
        return max(2.5, min(12.5, duty))
    
    def enable_servo_power(self):
        """Enable servo power and attach PWM"""
        if self.simulation_mode:
            self.servos_powered = True
            self.pwm_attached = True
            logging.info("Servo power enabled (simulation)")
            return
        
        try:
            # Enable power if power pin available
            if self.power_pin:
                GPIO.output(self.power_pin, GPIO.HIGH)
                time.sleep(0.1)  # Wait for power stabilization
            
            # Create and start PWM
            if not self.pwm_attached:
                self.pan_pwm = GPIO.PWM(self.pan_pin, 50)  # 50Hz
                self.tilt_pwm = GPIO.PWM(self.tilt_pin, 50)
                
                # Start at current positions
                pan_duty = self._angle_to_duty_cycle(self.current_pan)
                tilt_duty = self._angle_to_duty_cycle(self.current_tilt)
                
                self.pan_pwm.start(pan_duty)
                self.tilt_pwm.start(tilt_duty)
                
                self.pwm_attached = True
            
            self.servos_powered = True
            logging.info("Servo power enabled and PWM attached")
            
        except Exception as e:
            logging.error(f"Failed to enable servo power: {e}")
    
    def disable_servo_power(self):
        """Disable servo power and detach PWM"""
        if self.simulation_mode:
            self.servos_powered = False
            self.pwm_attached = False
            logging.info("Servo power disabled (simulation)")
            return
        
        try:
            # Stop PWM
            if self.pwm_attached and self.pan_pwm and self.tilt_pwm:
                self.pan_pwm.stop()
                self.tilt_pwm.stop()
                self.pwm_attached = False
            
            # Cut power if power pin available
            if self.power_pin:
                GPIO.output(self.power_pin, GPIO.LOW)
            
            self.servos_powered = False
            logging.info("Servo power disabled and PWM detached")
            
        except Exception as e:
            logging.error(f"Failed to disable servo power: {e}")
    
    def start_soft_start(self, target_pan: float, target_tilt: float):
        """Start soft-start ramp to target angles"""
        self.soft_start_active = True
        self.soft_start_start_time = time.time()
        self.soft_start_start_angles = (self.current_pan, self.current_tilt)
        self.target_pan = target_pan
        self.target_tilt = target_tilt
        
        logging.info(f"Soft start initiated to Pan={target_pan:.1f}°, Tilt={target_tilt:.1f}°")
    
    def update_positions(self, target_pan: float, target_tilt: float, dt: float):
        """Update servo positions with rate limiting and smoothing"""
        # Apply angle limits
        target_pan = max(min(target_pan, self.control_config.pan_max), self.control_config.pan_min)
        target_tilt = max(min(target_tilt, self.control_config.tilt_max), self.control_config.tilt_min)
        
        # Handle soft start
        if self.soft_start_active:
            elapsed = time.time() - self.soft_start_start_time
            if elapsed < self.safety_config.soft_start_duration:
                # Ramp to target
                progress = elapsed / self.safety_config.soft_start_duration
                start_pan, start_tilt = self.soft_start_start_angles
                
                # Apply rate limit during soft start
                max_change = self.safety_config.soft_start_max_rate * elapsed
                
                target_pan = start_pan + (target_pan - start_pan) * progress
                target_tilt = start_tilt + (target_tilt - start_tilt) * progress
                
                # Ensure we don't exceed soft start rate
                pan_change = abs(target_pan - start_pan)
                tilt_change = abs(target_tilt - start_tilt)
                if pan_change > max_change or tilt_change > max_change:
                    scale = max_change / max(pan_change, tilt_change, 0.1)
                    target_pan = start_pan + (target_pan - start_pan) * scale
                    target_tilt = start_tilt + (target_tilt - start_tilt) * scale
            else:
                self.soft_start_active = False
                logging.info("Soft start completed")
        
        # Apply rate limiting
        max_change_per_frame = self.control_config.max_step_deg
        max_change_per_time = self.control_config.max_slew_rate * dt
        max_change = min(max_change_per_frame, max_change_per_time)
        
        # Limit pan change
        pan_diff = target_pan - self.target_pan
        if abs(pan_diff) > max_change:
            target_pan = self.target_pan + np.sign(pan_diff) * max_change
        
        # Limit tilt change
        tilt_diff = target_tilt - self.target_tilt
        if abs(tilt_diff) > max_change:
            target_tilt = self.target_tilt + np.sign(tilt_diff) * max_change
        
        self.target_pan = target_pan
        self.target_tilt = target_tilt
        
        # Apply exponential moving average smoothing
        alpha = self.control_config.ema_alpha
        self.smooth_pan = alpha * self.target_pan + (1 - alpha) * self.smooth_pan
        self.smooth_tilt = alpha * self.target_tilt + (1 - alpha) * self.smooth_tilt
        
        # Update current positions (for next iteration)
        self.current_pan = self.smooth_pan
        self.current_tilt = self.smooth_tilt
        
        # Apply safety checks
        self._check_safety_limits()
        
        # Update physical servos
        self._update_physical_servos()
    
    def _check_safety_limits(self):
        """Check for thermal/over-current conditions at limits"""
        current_time = time.time()
        
        # Check pan limits
        if (abs(self.current_pan - self.control_config.pan_min) < 1.0 or 
            abs(self.current_pan - self.control_config.pan_max) < 1.0):
            
            if 'pan' not in self.limit_start_time:
                self.limit_start_time['pan'] = current_time
            elif current_time - self.limit_start_time['pan'] > self.safety_config.thermal_limit_ms / 1000.0:
                # Back off from limit
                if self.current_pan < 90:  # Near minimum
                    self.current_pan += self.safety_config.backoff_deg
                else:  # Near maximum
                    self.current_pan -= self.safety_config.backoff_deg
                logging.warning(f"Pan thermal protection: backed off to {self.current_pan:.1f}°")
                del self.limit_start_time['pan']
        else:
            if 'pan' in self.limit_start_time:
                del self.limit_start_time['pan']
        
        # Check tilt limits  
        if (abs(self.current_tilt - self.control_config.tilt_min) < 1.0 or 
            abs(self.current_tilt - self.control_config.tilt_max) < 1.0):
            
            if 'tilt' not in self.limit_start_time:
                self.limit_start_time['tilt'] = current_time
            elif current_time - self.limit_start_time['tilt'] > self.safety_config.thermal_limit_ms / 1000.0:
                # Back off from limit
                if self.current_tilt < 80:  # Near minimum
                    self.current_tilt += self.safety_config.backoff_deg
                else:  # Near maximum
                    self.current_tilt -= self.safety_config.backoff_deg
                logging.warning(f"Tilt thermal protection: backed off to {self.current_tilt:.1f}°")
                del self.limit_start_time['tilt']
        else:
            if 'tilt' in self.limit_start_time:
                del self.limit_start_time['tilt']
    
    def _update_physical_servos(self):
        """Update physical servo positions"""
        if not self.servos_powered or not self.pwm_attached:
            return
        
        if self.simulation_mode:
            return
        
        try:
            pan_duty = self._angle_to_duty_cycle(self.current_pan)
            tilt_duty = self._angle_to_duty_cycle(self.current_tilt)
            
            if self.pan_pwm and self.tilt_pwm:
                self.pan_pwm.ChangeDutyCycle(pan_duty)
                self.tilt_pwm.ChangeDutyCycle(tilt_duty)
                
        except Exception as e:
            logging.error(f"Failed to update servo positions: {e}")
    
    def get_position_degrees(self) -> Tuple[float, float]:
        """Get current servo positions in degrees"""
        return self.current_pan, self.current_tilt
    
    def cleanup(self):
        """Clean up servo controller"""
        self.disable_servo_power()
        
        if not self.simulation_mode:
            try:
                GPIO.cleanup()
                logging.info("GPIO cleaned up")
            except:
                pass

class AutoCenterTracker:
    """Main auto-center tracking system with state machine"""
    
    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        # Configuration
        self.target_config = TargetConfig()
        self.control_config = ControlConfig()
        self.safety_config = SafetyConfig()
        
        # Frame parameters
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2
        
        # Components
        self.camera = LibcameraCapture(frame_width, frame_height)
        self.detector = AdvancedTargetDetector(self.target_config)
        self.servo_controller = PowerManagedServoController(
            self.control_config, self.safety_config, 
            pan_pin=12, tilt_pin=13, power_pin=None  # Set power_pin if available
        )
        
        # PID controllers
        self.pan_pid = PIDController(
            self.control_config.kp_x, 
            self.control_config.ki_x, 
            self.control_config.kd_x
        )
        self.tilt_pid = PIDController(
            self.control_config.kp_y, 
            self.control_config.ki_y, 
            self.control_config.kd_y
        )
        
        # State machine
        self.state = TrackingState.IDLE_OFF
        self.last_target_time = 0.0
        self.centered_start_time = 0.0
        
        # Control variables
        self.running = True
        self.last_frame_time = time.time()
        
        logging.basicConfig(level=logging.INFO)
        logging.info("Auto-center tracker initialized")
    
    def select_best_target(self, targets: List[Dict]) -> Optional[Dict]:
        """Select the best target based on confidence and size"""
        if not targets:
            return None
        
        # Already sorted by confidence and area in detector
        best_target = targets[0]
        
        # Additional filtering: prefer faces over bodies if confidence is similar
        for target in targets:
            if (target['type'] == 'face' and 
                target['confidence'] > best_target['confidence'] - 0.1):
                best_target = target
                break
        
        return best_target
    
    def compute_error(self, target_center: Tuple[int, int]) -> Tuple[float, float]:
        """Compute pixel error from image center with deadband"""
        target_x, target_y = target_center
        
        error_x = target_x - self.center_x  # Positive = right
        error_y = self.center_y - target_y  # Positive = up (inverted Y)
        
        return error_x, error_y
    
    def is_in_deadband(self, error_x: float, error_y: float) -> bool:
        """Check if target is within deadband"""
        return (abs(error_x) <= self.target_config.deadband_x and 
                abs(error_y) <= self.target_config.deadband_y)
    
    def compute_control_output(self, error_x: float, error_y: float, dt: float) -> Tuple[float, float]:
        """Compute control output using PID"""
        # Normalize errors by image dimensions
        normalized_error_x = error_x / self.frame_width
        normalized_error_y = error_y / self.frame_height
        
        # PID control
        pan_output = self.pan_pid.update(normalized_error_x, dt)
        tilt_output = self.tilt_pid.update(normalized_error_y, dt)
        
        # Convert to angle deltas (degrees)
        pan_delta = pan_output * 180.0  # Scale to reasonable servo range
        tilt_delta = tilt_output * 180.0
        
        return pan_delta, tilt_delta
    
    def update_state_machine(self, has_target: bool, in_deadband: bool):
        """Update the tracking state machine"""
        current_time = time.time()
        
        if self.state == TrackingState.IDLE_OFF:
            if has_target:
                # Wake on detection
                logging.info("Target detected - enabling servos and starting tracking")
                self.servo_controller.enable_servo_power()
                
                # Start soft-start to current target
                current_pan, current_tilt = self.servo_controller.get_position_degrees()
                self.servo_controller.start_soft_start(current_pan, current_tilt)
                
                self.state = TrackingState.TRACKING
                self.last_target_time = current_time
                
                # Reset PID controllers
                self.pan_pid.reset()
                self.tilt_pid.reset()
        
        elif self.state == TrackingState.TRACKING:
            if not has_target:
                # Check if target lost for too long
                if current_time - self.last_target_time > self.target_config.lost_ms / 1000.0:
                    logging.info("Target lost - disabling servos")
                    self.servo_controller.disable_servo_power()
                    self.state = TrackingState.IDLE_OFF
            
            elif in_deadband:
                # Target is centered
                self.centered_start_time = current_time
                self.state = TrackingState.CENTERED_HOLD
                logging.debug("Target centered - entering hold state")
            
            if has_target:
                self.last_target_time = current_time
        
        elif self.state == TrackingState.CENTERED_HOLD:
            if not has_target:
                # Check if target lost
                if current_time - self.last_target_time > self.target_config.lost_ms / 1000.0:
                    logging.info("Target lost from centered state - disabling servos")
                    self.servo_controller.disable_servo_power()
                    self.state = TrackingState.IDLE_OFF
            
            elif not in_deadband:
                # Target moved out of deadband
                self.state = TrackingState.TRACKING
                logging.debug("Target left deadband - resuming tracking")
            
            if has_target:
                self.last_target_time = current_time
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame"""
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Detect targets
        targets = self.detector.detect_targets(frame)
        best_target = self.select_best_target(targets)
        
        has_target = best_target is not None
        in_deadband = False
        
        if has_target:
            # Compute error
            error_x, error_y = self.compute_error(best_target['center'])
            in_deadband = self.is_in_deadband(error_x, error_y)
            
            # Only compute control if not in deadband and tracking
            if not in_deadband and self.state == TrackingState.TRACKING:
                pan_delta, tilt_delta = self.compute_control_output(error_x, error_y, dt)
                
                # Apply control to servos
                current_pan, current_tilt = self.servo_controller.get_position_degrees()
                new_pan = current_pan + pan_delta
                new_tilt = current_tilt - tilt_delta  # Invert tilt for correct direction
                
                self.servo_controller.update_positions(new_pan, new_tilt, dt)
                
                logging.debug(f"Control: error=({error_x:.1f}, {error_y:.1f}), "
                            f"delta=({pan_delta:.2f}, {tilt_delta:.2f}), "
                            f"pos=({new_pan:.1f}, {new_tilt:.1f})")
        
        # Update state machine
        self.update_state_machine(has_target, in_deadband)
        
        # Draw visualization
        display_frame = self._draw_visualization(frame, targets, best_target, 
                                               has_target, in_deadband)
        
        return display_frame
    
    def _draw_visualization(self, frame: np.ndarray, targets: List[Dict], 
                          best_target: Optional[Dict], has_target: bool, 
                          in_deadband: bool) -> np.ndarray:
        """Draw tracking visualization"""
        display = frame.copy()
        
        # Draw center crosshairs
        cv2.line(display, (self.center_x - 40, self.center_y), 
                (self.center_x + 40, self.center_y), (0, 255, 255), 2)
        cv2.line(display, (self.center_x, self.center_y - 40), 
                (self.center_x, self.center_y + 40), (0, 255, 255), 2)
        
        # Draw deadband rectangle
        deadband_color = (0, 255, 0) if in_deadband else (128, 128, 128)
        cv2.rectangle(display, 
                     (self.center_x - self.target_config.deadband_x, 
                      self.center_y - self.target_config.deadband_y),
                     (self.center_x + self.target_config.deadband_x, 
                      self.center_y + self.target_config.deadband_y),
                     deadband_color, 2)
        
        # Draw targets
        for i, target in enumerate(targets):
            center = target['center']
            bbox = target['bbox']
            x, y, w, h = bbox
            
            # Color coding
            if target == best_target:
                color = (0, 255, 0)  # Green for selected target
                thickness = 3
            else:
                color = (0, 255, 255)  # Yellow for other targets
                thickness = 2
            
            # Bounding box
            cv2.rectangle(display, (x, y), (x + w, y + h), color, thickness)
            
            # Center point
            cv2.circle(display, center, 8, color, -1)
            
            # Confidence and type
            label = f"{target['type']} {target['confidence']:.2f}"
            cv2.putText(display, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Tracking line for best target
            if target == best_target and has_target:
                cv2.line(display, center, (self.center_x, self.center_y), color, 2)
        
        # Status panel
        self._draw_status_panel(display, has_target, in_deadband)
        
        return display
    
    def _draw_status_panel(self, frame: np.ndarray, has_target: bool, in_deadband: bool):
        """Draw status information panel"""
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Title
        cv2.putText(frame, "AUTO-CENTER PAN/TILT TRACKER", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # State
        state_colors = {
            TrackingState.IDLE_OFF: (128, 128, 128),
            TrackingState.TRACKING: (0, 255, 0),
            TrackingState.CENTERED_HOLD: (0, 255, 255)
        }
        state_color = state_colors[self.state]
        cv2.putText(frame, f"State: {self.state.value.upper()}", (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        # Target status
        target_status = "TARGET ACQUIRED" if has_target else "NO TARGET"
        target_color = (0, 255, 0) if has_target else (128, 128, 128)
        cv2.putText(frame, target_status, (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, target_color, 1)
        
        # Deadband status
        if has_target:
            deadband_status = "CENTERED" if in_deadband else "ADJUSTING"
            deadband_color = (0, 255, 0) if in_deadband else (255, 255, 0)
            cv2.putText(frame, deadband_status, (200, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, deadband_color, 1)
        
        # Servo positions
        pan, tilt = self.servo_controller.get_position_degrees()
        cv2.putText(frame, f"Pan: {pan:.1f}° | Tilt: {tilt:.1f}°", (20, 115), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Servo power status
        power_status = "POWERED" if self.servo_controller.servos_powered else "POWER OFF"
        power_color = (0, 255, 0) if self.servo_controller.servos_powered else (0, 0, 255)
        cv2.putText(frame, f"Servos: {power_status}", (20, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, power_color, 1)
        
        # Controls
        cv2.putText(frame, "Controls: Q=Quit | C=Center | R=Reset", (20, 165), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Performance
        cv2.putText(frame, f"Deadband: ±{self.target_config.deadband_x}px", (20, 185), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    def run(self):
        """Main tracking loop"""
        print("\n" + "="*80)
        print("🎯 AUTO-CENTER PAN/TILT TRACKER WITH POWER MANAGEMENT")
        print("="*80)
        print("✨ Features:")
        print("  - Target selection with confidence thresholding")
        print("  - Precise deadband control with centering hold")
        print("  - PID control with rate limiting and smoothing")
        print("  - Servo power management (auto power-off when idle)")
        print("  - State machine: IDLE_OFF → TRACKING → CENTERED_HOLD")
        print("  - Safety features: thermal protection, soft-start")
        print("\n🎮 Controls:")
        print("  Q - Quit")
        print("  C - Manual center")
        print("  R - Reset PID controllers")
        print("-"*80)
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    logging.warning("Camera read failed")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Process frame
                display_frame = self.process_frame(frame)
                
                # Display
                cv2.imshow("Auto-Center Tracker", display_frame)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    print("\n⏹️ Quit requested")
                    break
                elif key == ord('c'):
                    print("🎯 Manual center command")
                    self.servo_controller.enable_servo_power()
                    self.servo_controller.update_positions(90.0, 80.0, 0.1)
                elif key == ord('r'):
                    print("🔄 Resetting PID controllers")
                    self.pan_pid.reset()
                    self.tilt_pid.reset()
                
                # Performance logging
                if frame_count % 300 == 0:  # Every 300 frames (~10s at 30fps)
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    logging.info(f"Frame {frame_count}, FPS: {fps:.1f}, State: {self.state.value}")
                
                # Frame rate limiting
                time.sleep(0.033)  # ~30 FPS
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        
        finally:
            print("\n🛑 Shutting down safely...")
            
            # Safe shutdown sequence
            if self.servo_controller.servos_powered:
                print("🎯 Centering servos for safe shutdown...")
                self.servo_controller.update_positions(90.0, 80.0, 0.1)
                time.sleep(1.0)
            
            # Cleanup
            self.servo_controller.cleanup()
            self.camera.release()
            cv2.destroyAllWindows()
            
            # Final report
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
                print(f"\n📊 Session Report:")
                print(f"  Duration: {elapsed:.1f}s")
                print(f"  Frames: {frame_count}")
                print(f"  Average FPS: {fps:.1f}")
                print(f"  Servo mode: {'Hardware' if not self.servo_controller.simulation_mode else 'Simulation'}")
            
            print("\n✅ Auto-center tracker stopped safely")

def main():
    """Main entry point"""
    print("🎯 Initializing Auto-Center Pan/Tilt Tracker...")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run tracker
    tracker = AutoCenterTracker(frame_width=1280, frame_height=720)
    tracker.run()

if __name__ == "__main__":
    main()