#!/usr/bin/env python3
"""
Configuration file for Auto-Center Pan/Tilt Tracker
Easily adjust all parameters without modifying main code.
"""
from dataclasses import dataclass

@dataclass
class TrackerConfiguration:
    """Complete tracker configuration"""
    
    # Camera settings
    frame_width: int = 1280
    frame_height: int = 720
    
    # GPIO pin assignments
    pan_servo_pin: int = 12      # GPIO12 (Physical pin 32)
    tilt_servo_pin: int = 13     # GPIO13 (Physical pin 33)
    power_relay_pin: int = None  # Optional: GPIO for servo power relay/MOSFET
    
    # Target detection parameters
    confidence_min: float = 0.5         # Minimum detection confidence (0.0-1.0)
    deadband_x_pixels: int = 15         # Horizontal deadband tolerance (pixels)
    deadband_y_pixels: int = 15         # Vertical deadband tolerance (pixels)
    center_hold_ms: int = 500           # Hold time in deadband before considering centered (ms)
    target_lost_ms: int = 1000          # Time before target considered lost (ms)
    debounce_frames: int = 3            # Frames to confirm stable detection
    
    # PID control parameters
    # Pan axis (horizontal movement)
    pan_kp: float = 0.020               # Proportional gain
    pan_ki: float = 0.003               # Integral gain  
    pan_kd: float = 0.008               # Derivative gain
    
    # Tilt axis (vertical movement)
    tilt_kp: float = 0.015              # Proportional gain
    tilt_ki: float = 0.002              # Integral gain
    tilt_kd: float = 0.006              # Derivative gain
    
    # Movement limits and smoothing
    max_step_per_frame: float = 4.0     # Maximum angle change per frame (degrees)
    max_slew_rate: float = 100.0        # Maximum movement rate (degrees/second)
    smoothing_factor: float = 0.25      # Exponential smoothing (0.0-1.0, higher = more responsive)
    
    # Servo mechanical limits (degrees, standard 0-180° servo range)
    pan_min_angle: float = 15.0         # Pan minimum (left limit)
    pan_max_angle: float = 165.0        # Pan maximum (right limit)
    tilt_min_angle: float = 30.0        # Tilt minimum (down limit)
    tilt_max_angle: float = 130.0       # Tilt maximum (up limit)
    
    # Initial servo positions (degrees)
    initial_pan_angle: float = 90.0     # Center pan position
    initial_tilt_angle: float = 75.0    # Slightly tilted up position
    
    # Safety parameters
    thermal_protection_ms: int = 2500   # Time at limit before thermal protection (ms)
    thermal_backoff_degrees: float = 5.0 # Degrees to back off from limit
    
    # Soft start parameters (gradual movement when powering on)
    soft_start_duration: float = 0.4    # Ramp-up time when servos first power on (seconds)
    soft_start_max_rate: float = 80.0   # Maximum rate during soft start (deg/s)
    
    # Performance parameters
    target_fps: float = 25.0            # Target frame rate
    status_update_frames: int = 150     # Frames between status log updates
    
    # Detection tuning
    face_scale_factor: float = 1.1      # Haar cascade scale factor
    face_min_neighbors: int = 4         # Minimum neighbors for face detection
    face_min_size: tuple = (50, 50)     # Minimum face size (width, height)
    
    body_scale_factor: float = 1.2      # Body detection scale factor  
    body_min_neighbors: int = 3         # Minimum neighbors for body detection
    body_min_size: tuple = (80, 100)    # Minimum body size (width, height)
    
    def validate(self):
        """Validate configuration parameters"""
        errors = []
        
        # Check angle limits
        if self.pan_min_angle >= self.pan_max_angle:
            errors.append("pan_min_angle must be less than pan_max_angle")
        if self.tilt_min_angle >= self.tilt_max_angle:
            errors.append("tilt_min_angle must be less than tilt_max_angle")
        
        # Check initial positions are within limits
        if not (self.pan_min_angle <= self.initial_pan_angle <= self.pan_max_angle):
            errors.append("initial_pan_angle must be within pan limits")
        if not (self.tilt_min_angle <= self.initial_tilt_angle <= self.tilt_max_angle):
            errors.append("initial_tilt_angle must be within tilt limits")
        
        # Check PID gains are positive
        if any(gain < 0 for gain in [self.pan_kp, self.pan_ki, self.pan_kd, 
                                    self.tilt_kp, self.tilt_ki, self.tilt_kd]):
            errors.append("All PID gains must be positive")
        
        # Check timing parameters
        if self.center_hold_ms <= 0 or self.target_lost_ms <= 0:
            errors.append("Timing parameters must be positive")
        
        # Check smoothing factor
        if not (0.0 <= self.smoothing_factor <= 1.0):
            errors.append("smoothing_factor must be between 0.0 and 1.0")
        
        # Check confidence threshold
        if not (0.0 <= self.confidence_min <= 1.0):
            errors.append("confidence_min must be between 0.0 and 1.0")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {err}" for err in errors))
        
        return True
    
    def get_tuning_suggestions(self):
        """Get tuning suggestions for different scenarios"""
        return """
🎯 TUNING GUIDE FOR AUTO-CENTER TRACKER

📍 For more responsive tracking (faster but less stable):
   - Increase PID proportional gains (pan_kp, tilt_kp): 0.025-0.035
   - Increase smoothing_factor: 0.4-0.6
   - Decrease debounce_frames: 2
   - Increase max_slew_rate: 150-200

🎯 For more stable tracking (slower but smoother):
   - Decrease PID proportional gains: 0.010-0.015
   - Decrease smoothing_factor: 0.1-0.2
   - Increase debounce_frames: 4-5
   - Decrease max_slew_rate: 60-80

🔧 For small/fast targets (birds, cats):
   - Increase confidence_min: 0.6-0.7
   - Decrease deadband: 8-12 pixels
   - Increase PID derivative gains (pan_kd, tilt_kd): 0.010-0.015
   - Decrease target_lost_ms: 600-800

🔧 For large/slow targets (people):
   - Decrease confidence_min: 0.3-0.4
   - Increase deadband: 20-30 pixels
   - Decrease PID derivative gains: 0.003-0.006
   - Increase target_lost_ms: 1200-1500

⚡ For power-critical applications:
   - Increase target_lost_ms: 1500+ (longer before power-off)
   - Increase center_hold_ms: 800+ (longer centered hold)
   - Consider using power_relay_pin for complete power cut

🛡️ For high-precision applications:
   - Decrease deadband: 5-8 pixels
   - Increase debounce_frames: 5+
   - Add integral gain (pan_ki, tilt_ki): 0.005+
   - Decrease max_step_per_frame: 2-3 degrees
"""

# Preset configurations for different use cases
class PresetConfigs:
    """Preset configurations for common scenarios"""
    
    @staticmethod
    def get_responsive_config():
        """Configuration optimized for fast, responsive tracking"""
        config = TrackerConfiguration()
        config.pan_kp = 0.030
        config.tilt_kp = 0.025
        config.smoothing_factor = 0.5
        config.max_slew_rate = 180.0
        config.debounce_frames = 2
        config.deadband_x_pixels = 10
        config.deadband_y_pixels = 10
        return config
    
    @staticmethod
    def get_stable_config():
        """Configuration optimized for stable, smooth tracking"""
        config = TrackerConfiguration()
        config.pan_kp = 0.012
        config.tilt_kp = 0.010
        config.pan_ki = 0.001
        config.tilt_ki = 0.001
        config.smoothing_factor = 0.15
        config.max_slew_rate = 60.0
        config.debounce_frames = 5
        config.deadband_x_pixels = 25
        config.deadband_y_pixels = 25
        return config
    
    @staticmethod
    def get_power_efficient_config():
        """Configuration optimized for power efficiency"""
        config = TrackerConfiguration()
        config.target_lost_ms = 2000
        config.center_hold_ms = 1000
        config.thermal_protection_ms = 1500
        config.max_slew_rate = 80.0
        config.confidence_min = 0.6  # Higher confidence to avoid false triggers
        return config
    
    @staticmethod
    def get_precision_config():
        """Configuration optimized for high precision"""
        config = TrackerConfiguration()
        config.deadband_x_pixels = 6
        config.deadband_y_pixels = 6
        config.pan_ki = 0.005
        config.tilt_ki = 0.004
        config.debounce_frames = 6
        config.max_step_per_frame = 2.0
        config.smoothing_factor = 0.2
        return config

# Default configuration instance
DEFAULT_CONFIG = TrackerConfiguration()

def load_config(preset: str = None) -> TrackerConfiguration:
    """Load configuration, optionally from preset"""
    if preset == "responsive":
        return PresetConfigs.get_responsive_config()
    elif preset == "stable":
        return PresetConfigs.get_stable_config()
    elif preset == "power_efficient":
        return PresetConfigs.get_power_efficient_config()
    elif preset == "precision":
        return PresetConfigs.get_precision_config()
    else:
        return DEFAULT_CONFIG

if __name__ == "__main__":
    # Test configuration validation
    print("🔧 Testing tracker configuration...")
    
    config = DEFAULT_CONFIG
    try:
        config.validate()
        print("✅ Default configuration is valid")
    except ValueError as e:
        print(f"❌ Configuration validation failed: {e}")
    
    print("\n" + config.get_tuning_suggestions())
    
    # Show all preset configs
    print("\n📋 AVAILABLE PRESET CONFIGURATIONS:")
    presets = ["responsive", "stable", "power_efficient", "precision"]
    for preset in presets:
        config = load_config(preset)
        print(f"\n{preset.upper()}:")
        print(f"  PID: Pan({config.pan_kp:.3f}, {config.pan_ki:.3f}, {config.pan_kd:.3f})")
        print(f"       Tilt({config.tilt_kp:.3f}, {config.tilt_ki:.3f}, {config.tilt_kd:.3f})")
        print(f"  Deadband: {config.deadband_x_pixels}×{config.deadband_y_pixels}px")
        print(f"  Max slew: {config.max_slew_rate:.0f}°/s")
        print(f"  Smoothing: {config.smoothing_factor:.2f}")