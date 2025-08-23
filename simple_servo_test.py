#!/usr/bin/env python3
"""
Simple Direct Servo Test - No threading, no queues, just direct control
This bypasses the complex servo controller to test if the jitter is from threading
"""
import time
import sys

# Try direct GPIO control first
try:
    from gpiozero import Servo, Device
    from gpiozero.pins.pigpio import PiGPIOFactory
    GPIO_AVAILABLE = True
    print("✅ GPIO libraries available")
except ImportError:
    GPIO_AVAILABLE = False
    print("❌ GPIO libraries not available - simulation mode")

class DirectServoTest:
    """Ultra-simple direct servo control - no threading"""
    
    def __init__(self):
        print("\n🔧 DIRECT SERVO TEST")
        print("="*40)
        print("Direct servo control - no threading, no queues")
        
        # Servo configuration
        self.pan_pin = 18    # GPIO18
        self.tilt_pin = 23   # GPIO23
        
        self.pan_servo = None
        self.tilt_servo = None
        
        if GPIO_AVAILABLE:
            self.setup_servos()
        else:
            print("⚠️ Running in simulation mode")
        
        print("="*40)
    
    def setup_servos(self):
        """Direct servo setup with optimized parameters"""
        try:
            # Use pigpio for best performance
            Device.pin_factory = PiGPIOFactory()
            print("✅ Using pigpio for servo control")
            
            # Create servos with precise timing
            self.pan_servo = Servo(
                self.pan_pin,
                min_pulse_width=0.5e-3,   # 0.5ms
                max_pulse_width=2.5e-3,   # 2.5ms  
                frame_width=20e-3         # 20ms (50Hz)
            )
            
            self.tilt_servo = Servo(
                self.tilt_pin,
                min_pulse_width=0.5e-3,   # 0.5ms
                max_pulse_width=2.5e-3,   # 2.5ms
                frame_width=20e-3         # 20ms (50Hz)
            )
            
            # Start at center
            self.pan_servo.value = 0
            self.tilt_servo.value = 0
            
            print(f"✅ Pan servo: GPIO{self.pan_pin}")
            print(f"✅ Tilt servo: GPIO{self.tilt_pin}")
            
        except Exception as e:
            print(f"❌ Servo setup failed: {e}")
            self.pan_servo = None
            self.tilt_servo = None
    
    def move_direct(self, pan_value, tilt_value):
        """Direct servo movement - no smoothing, no threading"""
        if not GPIO_AVAILABLE or not self.pan_servo or not self.tilt_servo:
            print(f"🎯 SIMULATED: Pan={pan_value:.3f}, Tilt={tilt_value:.3f}")
            return
        
        # Clamp values
        pan_value = max(-1, min(1, pan_value))
        tilt_value = max(-1, min(1, tilt_value))
        
        print(f"🎯 DIRECT MOVE: Pan={pan_value:.3f}, Tilt={tilt_value:.3f}")
        
        # Direct assignment - no queues, no threads
        self.pan_servo.value = pan_value
        self.tilt_servo.value = tilt_value
    
    def test_step_movements(self):
        """Test with step movements to see if jitter occurs"""
        print("\n📊 STEP MOVEMENT TEST")
        print("Each position held for 3 seconds")
        print("Watch for any jittering or unwanted movement")
        print("-" * 30)
        
        positions = [
            (0, 0, "Center"),
            (0.5, 0, "Right"),
            (1.0, 0, "Far Right"),
            (0, 0, "Center"),
            (-0.5, 0, "Left"), 
            (-1.0, 0, "Far Left"),
            (0, 0, "Center"),
            (0, 0.5, "Up"),
            (0, 1.0, "Far Up"),
            (0, 0, "Center"),
            (0, -0.5, "Down"),
            (0, -1.0, "Far Down"),
            (0, 0, "Center")
        ]
        
        for i, (pan, tilt, desc) in enumerate(positions):
            print(f"\n{i+1:2d}. {desc:10s} - Pan={pan:5.1f}, Tilt={tilt:5.1f}")
            self.move_direct(pan, tilt)
            
            # Hold position and watch for jitter
            print("    Holding position for 3 seconds... (watch for jitter)")
            time.sleep(3)
    
    def test_smooth_sweep(self):
        """Test smooth continuous movement"""
        print("\n🌊 SMOOTH SWEEP TEST")
        print("Continuous smooth movement for 30 seconds")
        print("Should be completely smooth with no jitter")
        print("-" * 30)
        
        import math
        
        start_time = time.time()
        duration = 30  # 30 seconds
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            
            # Slow sine wave for pan
            pan = 0.8 * math.sin(elapsed * 0.3)  # 0.3 Hz, slow movement
            
            # Even slower cosine for tilt
            tilt = 0.4 * math.cos(elapsed * 0.2)  # 0.2 Hz, very slow
            
            # Move directly
            self.move_direct(pan, tilt)
            
            # Status every 5 seconds
            if int(elapsed) % 5 == 0 and elapsed > 0 and int(elapsed * 10) % 50 < 2:
                print(f"🌊 {elapsed:5.1f}s: Pan={pan:6.3f}, Tilt={tilt:6.3f}")
            
            # Small delay - should be very smooth
            time.sleep(0.02)  # 50Hz update rate
        
        # Return to center
        print("🎯 Returning to center...")
        self.move_direct(0, 0)
    
    def test_micro_movements(self):
        """Test very small movements that might cause jitter"""
        print("\n🔍 MICRO-MOVEMENT TEST")
        print("Testing very small position changes")
        print("This often reveals jitter issues")
        print("-" * 30)
        
        # Start at center
        self.move_direct(0, 0)
        time.sleep(2)
        
        # Very small incremental movements
        increments = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
        
        for increment in increments:
            print(f"\n🔍 Testing increment: {increment:.3f}")
            
            position = 0
            for step in range(10):  # 10 micro steps
                position += increment
                position = min(0.5, position)  # Don't go too far
                
                print(f"   Step {step+1}: {position:.3f}")
                self.move_direct(position, 0)
                time.sleep(0.5)  # Hold each micro position
            
            # Reset to center
            self.move_direct(0, 0)
            time.sleep(1)
    
    def cleanup(self):
        """Clean shutdown"""
        print("\n🛑 Cleaning up...")
        if GPIO_AVAILABLE and self.pan_servo and self.tilt_servo:
            self.pan_servo.value = 0
            self.tilt_servo.value = 0
            time.sleep(0.5)
            
            # Close servo connections
            self.pan_servo.close()
            self.tilt_servo.close()
        
        print("✅ Direct servo test complete!")

def main():
    print("🔧 SIMPLE DIRECT SERVO TEST")
    print("This bypasses all complex threading and queuing")
    print("Direct servo control to isolate jitter source")
    print("")
    print("Tests available:")
    print("1. Step movements (hold positions)")
    print("2. Smooth sweep (continuous movement)")  
    print("3. Micro movements (tiny increments)")
    print("4. All tests")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
        
        servo_test = DirectServoTest()
        
        if choice == "1":
            servo_test.test_step_movements()
        elif choice == "2":
            servo_test.test_smooth_sweep()
        elif choice == "3":
            servo_test.test_micro_movements()
        elif choice == "4":
            servo_test.test_step_movements()
            input("\nPress Enter for smooth sweep test...")
            servo_test.test_smooth_sweep()
            input("\nPress Enter for micro movement test...")
            servo_test.test_micro_movements()
        else:
            print("Invalid choice, running all tests...")
            servo_test.test_step_movements()
            servo_test.test_smooth_sweep()
            servo_test.test_micro_movements()
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted")
    
    finally:
        if 'servo_test' in locals():
            servo_test.cleanup()

if __name__ == "__main__":
    main()