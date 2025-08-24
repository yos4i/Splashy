#!/usr/bin/env python3
"""
Power-Managed Servo Controller
Turns off servo power when not needed to save power and reduce wear
"""
import time
import threading
import logging
from typing import Optional
from dataclasses import dataclass

try:
    from gpiozero import Servo, Device
    from gpiozero.pins.native import NativeFactory
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logging.warning("GPIO libraries not available - running in simulation mode")

@dataclass
class PowerManagedServoConfig:
    # GPIO pins
    pan_pin: int = 18   # GPIO18 - Physical pin 12
    tilt_pin: int = 23  # GPIO23 - Physical pin 16
    
    # Servo parameters
    min_pulse_width: float = 0.5e-3   # 0.5ms
    max_pulse_width: float = 2.5e-3   # 2.5ms
    frame_width: float = 20e-3        # 20ms (50Hz)
    
    # Power management
    auto_disable_delay: float = 3.0   # Turn off servos after 3 seconds of no movement
    movement_settle_time: float = 0.5 # Wait 500ms after movement before allowing disable

class PowerManagedServoController:
    """Servo controller with automatic power management"""
    
    def __init__(self, config: PowerManagedServoConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Use native pin factory (Pi 5 compatible, no pigpio needed)
        if GPIO_AVAILABLE:
            try:
                Device.pin_factory = NativeFactory()
                self.logger.info("Using native pin factory (Pi 5 compatible)")
            except Exception as e:
                self.logger.warning(f"Native factory setup warning: {e}")
        
        # Servo objects
        self.pan_servo = None
        self.tilt_servo = None
        self.servos_enabled = False
        
        # Current positions
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.target_pan = 0.0
        self.target_tilt = 0.0
        
        # Power management
        self.last_movement_time = 0
        self.auto_disable_timer = None
        self.power_lock = threading.Lock()
        
        # Movement tracking
        self.is_moving = False
        
        print("✅ Power-managed servo controller initialized")
        print(f"📍 Pan: GPIO{self.config.pan_pin}, Tilt: GPIO{self.config.tilt_pin}")
        print(f"⏰ Auto-disable delay: {self.config.auto_disable_delay}s")
    
    def enable_servos(self):
        """Enable servo power and create servo objects"""
        with self.power_lock:
            if self.servos_enabled:
                return  # Already enabled
            
            if not GPIO_AVAILABLE:
                self.logger.info("Servo power enabled (simulation)")
                self.servos_enabled = True
                return
            
            try:
                self.logger.info("🔌 Enabling servo power...")
                
                # Create servo objects
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
                
                # Set to current positions
                self.pan_servo.value = self.current_pan
                self.tilt_servo.value = self.current_tilt
                
                self.servos_enabled = True
                self.logger.info("✅ Servo power enabled")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to enable servos: {e}")
                self.servos_enabled = False
    
    def disable_servos(self):
        """Disable servo power to save energy"""
        with self.power_lock:
            if not self.servos_enabled:
                return  # Already disabled
            
            if not GPIO_AVAILABLE:
                self.logger.info("Servo power disabled (simulation)")
                self.servos_enabled = False
                return
            
            try:
                self.logger.info("🔌 Disabling servo power...")
                
                if self.pan_servo:
                    self.pan_servo.close()
                    self.pan_servo = None
                
                if self.tilt_servo:
                    self.tilt_servo.close()
                    self.tilt_servo = None
                
                self.servos_enabled = False
                self.logger.info("✅ Servo power disabled - saving energy")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to disable servos: {e}")
    
    def cancel_auto_disable_timer(self):
        """Cancel the automatic disable timer"""
        if self.auto_disable_timer and self.auto_disable_timer.is_alive():
            self.auto_disable_timer.cancel()
            self.auto_disable_timer = None
    
    def start_auto_disable_timer(self):
        """Start timer to automatically disable servos after idle time"""
        self.cancel_auto_disable_timer()
        
        def auto_disable_callback():
            current_time = time.time()
            time_since_movement = current_time - self.last_movement_time
            
            if time_since_movement >= self.config.auto_disable_delay:
                self.logger.info(f"⏰ Auto-disabling servos after {time_since_movement:.1f}s idle")
                self.disable_servos()
        
        self.auto_disable_timer = threading.Timer(self.config.auto_disable_delay, auto_disable_callback)
        self.auto_disable_timer.start()
    
    def set_position(self, pan: float, tilt: float):
        """Set servo positions with automatic power management"""
        # Clamp values
        pan = max(-1, min(1, pan))
        tilt = max(-1, min(1, tilt))
        
        # Check if position actually changed
        pan_changed = abs(pan - self.current_pan) > 0.001
        tilt_changed = abs(tilt - self.current_tilt) > 0.001
        
        if not pan_changed and not tilt_changed:
            return  # No movement needed
        
        self.target_pan = pan
        self.target_tilt = tilt
        self.last_movement_time = time.time()
        self.is_moving = True
        
        # Cancel any pending auto-disable
        self.cancel_auto_disable_timer()
        
        # Enable servos if needed
        self.enable_servos()
        
        # Move servos
        if self.servos_enabled and GPIO_AVAILABLE:
            if self.pan_servo and pan_changed:
                self.pan_servo.value = pan
                self.logger.info(f"🎯 Pan: {self.current_pan:.3f} → {pan:.3f}")
            
            if self.tilt_servo and tilt_changed:
                self.tilt_servo.value = tilt
                self.logger.info(f"🎯 Tilt: {self.current_tilt:.3f} → {tilt:.3f}")
        else:
            self.logger.info(f"🎯 SIMULATED: Pan={pan:.3f}, Tilt={tilt:.3f}")
        
        # Update current positions
        self.current_pan = pan
        self.current_tilt = tilt
        
        # Schedule auto-disable after settle time
        def schedule_auto_disable():
            time.sleep(self.config.movement_settle_time)
            self.is_moving = False
            self.start_auto_disable_timer()
        
        timer_thread = threading.Thread(target=schedule_auto_disable, daemon=True)
        timer_thread.start()
    
    def move_to_center(self):
        """Move both servos to center position"""
        self.logger.info("🎯 Moving to center")
        self.set_position(0, 0)
    
    def force_enable_servos(self):
        """Manually enable servos (cancels auto-disable)"""
        self.cancel_auto_disable_timer()
        self.enable_servos()
        self.logger.info("🔌 Servos manually enabled")
    
    def force_disable_servos(self):
        """Manually disable servos immediately"""
        self.cancel_auto_disable_timer()
        self.disable_servos()
        self.logger.info("🔌 Servos manually disabled")
    
    def get_position_degrees(self):
        """Get current positions in degrees"""
        pan_degrees = self.current_pan * 90.0
        tilt_degrees = (self.current_tilt + 1.0) / 2.0 * 90.0 - 30.0  # -30° to +60°
        return pan_degrees, tilt_degrees
    
    def is_powered(self):
        """Check if servos are currently powered"""
        return self.servos_enabled
    
    def cleanup(self):
        """Clean shutdown"""
        self.logger.info("🛑 Cleaning up power-managed servo controller...")
        
        # Cancel timer
        self.cancel_auto_disable_timer()
        
        # Center and disable servos
        self.move_to_center()
        time.sleep(1)
        self.disable_servos()
        
        self.logger.info("✅ Power-managed servo controller cleanup complete")

# Test the power-managed servo controller
def test_power_management():
    """Test power management features"""
    print("\n🔋 POWER MANAGEMENT TEST")
    print("="*50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create config
    config = PowerManagedServoConfig(auto_disable_delay=2.0)  # 2 second delay for testing
    
    # Create controller
    servo = PowerManagedServoController(config)
    
    try:
        print("\n1. Testing initial state...")
        print(f"   Servos powered: {servo.is_powered()}")
        time.sleep(1)
        
        print("\n2. Moving to position (should auto-enable)...")
        servo.set_position(0.5, 0)
        print(f"   Servos powered: {servo.is_powered()}")
        time.sleep(3)  # Wait for movement to settle
        
        print("\n3. Waiting for auto-disable (2 seconds)...")
        time.sleep(3)  # Should auto-disable after 2 seconds
        print(f"   Servos powered: {servo.is_powered()}")
        
        print("\n4. Another movement (should re-enable)...")
        servo.set_position(-0.5, 0.3)
        print(f"   Servos powered: {servo.is_powered()}")
        time.sleep(1)
        
        print("\n5. Manual disable test...")
        servo.force_disable_servos()
        print(f"   Servos powered: {servo.is_powered()}")
        
        print("\n6. Manual enable test...")
        servo.force_enable_servos()
        print(f"   Servos powered: {servo.is_powered()}")
        
        print("\n✅ Power management test complete!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted")
    
    finally:
        servo.cleanup()

def main():
    print("🔋 POWER-MANAGED SERVO CONTROLLER")
    print("Automatically saves power when servos not in use")
    print("")
    print("Features:")
    print("- Auto-disable after idle time")
    print("- Auto-enable when movement needed")
    print("- Manual enable/disable control")
    print("- Native GPIO (Pi 5 compatible)")
    print("")
    
    try:
        test_power_management()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()