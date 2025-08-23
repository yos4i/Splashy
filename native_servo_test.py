#!/usr/bin/env python3
"""
Native Servo Test - Use gpiozero with native pins (no pigpio)
This should work on Pi 5 without pigpio daemon issues
"""
import time
import sys

try:
    from gpiozero import Servo, Device
    from gpiozero.pins.native import NativeFactory
    GPIO_AVAILABLE = True
    print("✅ gpiozero available")
except ImportError:
    GPIO_AVAILABLE = False
    print("❌ gpiozero not available")

class NativeServoTest:
    """Test servos using gpiozero native pins (no pigpio)"""
    
    def __init__(self):
        print("\n🔧 NATIVE SERVO TEST (Pi 5 Compatible)")
        print("="*50)
        print("Using gpiozero with native pin factory")
        print("No pigpio daemon required!")
        
        # Force native pin factory (Pi 5 compatible)
        if GPIO_AVAILABLE:
            try:
                Device.pin_factory = NativeFactory()
                print("✅ Using Native pin factory")
            except Exception as e:
                print(f"⚠️ Native factory setup warning: {e}")
        
        # Servo configuration matching your setup
        self.pan_pin = 18    # GPIO18 - Physical pin 12
        self.tilt_pin = 23   # GPIO23 - Physical pin 16
        
        self.pan_servo = None
        self.tilt_servo = None
        
        if GPIO_AVAILABLE:
            self.setup_native_servos()
        else:
            print("⚠️ Running in simulation mode")
        
        print("="*50)
    
    def setup_native_servos(self):
        """Setup servos with native pin factory"""
        try:
            print(f"🎯 Setting up Pan servo on GPIO{self.pan_pin}...")
            self.pan_servo = Servo(
                self.pan_pin,
                min_pulse_width=0.5e-3,   # 0.5ms
                max_pulse_width=2.5e-3,   # 2.5ms
                frame_width=20e-3         # 20ms (50Hz)
            )
            print("✅ Pan servo created")
            
            print(f"🎯 Setting up Tilt servo on GPIO{self.tilt_pin}...")
            self.tilt_servo = Servo(
                self.tilt_pin,
                min_pulse_width=0.5e-3,   # 0.5ms
                max_pulse_width=2.5e-3,   # 2.5ms
                frame_width=20e-3         # 20ms (50Hz)
            )
            print("✅ Tilt servo created")
            
            # Start both at center
            print("🎯 Moving both servos to center...")
            self.pan_servo.value = 0
            self.tilt_servo.value = 0
            
            print("✅ Native servos initialized successfully!")
            print(f"📍 Pan: GPIO{self.pan_pin} (Physical pin 12)")
            print(f"📍 Tilt: GPIO{self.tilt_pin} (Physical pin 16)")
            
        except Exception as e:
            print(f"❌ Native servo setup failed: {e}")
            print("💡 This might be a hardware wiring issue")
            self.pan_servo = None
            self.tilt_servo = None
    
    def move_servos(self, pan_value, tilt_value, description=""):
        """Move both servos to specified positions"""
        if not GPIO_AVAILABLE or not self.pan_servo or not self.tilt_servo:
            print(f"🎯 SIMULATED: {description} - Pan={pan_value:.3f}, Tilt={tilt_value:.3f}")
            return
        
        # Clamp values to safe range
        pan_value = max(-1, min(1, pan_value))
        tilt_value = max(-1, min(1, tilt_value))
        
        print(f"🎯 {description} - Pan={pan_value:.3f}, Tilt={tilt_value:.3f}")
        
        try:
            # Move both servos
            self.pan_servo.value = pan_value
            self.tilt_servo.value = tilt_value
        except Exception as e:
            print(f"❌ Servo movement error: {e}")
    
    def test_basic_positions(self):
        """Test basic servo positions"""
        print("\n📐 BASIC POSITION TEST")
        print("Testing fundamental servo positions")
        print("Each position held for 2 seconds")
        print("-" * 30)
        
        positions = [
            (0, 0, "CENTER"),
            (0.5, 0, "PAN RIGHT"),
            (-0.5, 0, "PAN LEFT"),
            (0, 0, "CENTER"),
            (0, 0.5, "TILT UP"),
            (0, -0.5, "TILT DOWN"),
            (0, 0, "CENTER"),
            (0.5, 0.5, "RIGHT + UP"),
            (-0.5, -0.5, "LEFT + DOWN"),
            (0, 0, "FINAL CENTER")
        ]
        
        for i, (pan, tilt, desc) in enumerate(positions):
            print(f"\n{i+1:2d}. {desc}")
            self.move_servos(pan, tilt, desc)
            print("    Holding position for 2 seconds...")
            time.sleep(2)
            
            if GPIO_AVAILABLE and self.pan_servo:
                # Check if servos are actually at expected positions
                actual_pan = getattr(self.pan_servo, 'value', 'unknown')
                actual_tilt = getattr(self.tilt_servo, 'value', 'unknown')
                print(f"    Actual positions: Pan={actual_pan}, Tilt={actual_tilt}")
    
    def test_smooth_movement(self):
        """Test smooth continuous movement"""
        print("\n🌊 SMOOTH MOVEMENT TEST")
        print("Sine wave movement for 20 seconds")
        print("Should be smooth with no jerking")
        print("-" * 30)
        
        import math
        
        start_time = time.time()
        duration = 20
        
        print("Starting smooth movement...")
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            
            # Slow, smooth sine wave
            pan = 0.6 * math.sin(elapsed * 0.4)    # Slow pan movement
            tilt = 0.3 * math.cos(elapsed * 0.3)   # Even slower tilt
            
            self.move_servos(pan, tilt, f"Wave t={elapsed:.1f}s")
            
            # Show progress every 5 seconds
            if int(elapsed) % 5 == 0 and int(elapsed * 10) % 50 < 2:
                print(f"🌊 {elapsed:4.1f}s: Pan={pan:6.3f}, Tilt={tilt:6.3f}")
            
            time.sleep(0.1)  # 10Hz update rate
        
        # Return to center
        print("🎯 Returning to center...")
        self.move_servos(0, 0, "CENTER")
        time.sleep(1)
    
    def test_step_by_step(self):
        """Detailed step-by-step test with user confirmation"""
        print("\n👥 INTERACTIVE STEP TEST")
        print("Manual step-by-step servo testing")
        print("Press Enter after each step to continue")
        print("-" * 30)
        
        steps = [
            (0, 0, "CENTER - Both servos should be in middle position"),
            (1, 0, "PAN FULL RIGHT - Pan servo should move fully right"),
            (-1, 0, "PAN FULL LEFT - Pan servo should move fully left"),
            (0, 0, "PAN CENTER - Pan servo should return to center"),
            (0, 1, "TILT FULL UP - Tilt servo should move up"),
            (0, -1, "TILT FULL DOWN - Tilt servo should move down"),
            (0, 0, "ALL CENTER - Both servos back to center")
        ]
        
        for i, (pan, tilt, instruction) in enumerate(steps):
            print(f"\n--- STEP {i+1} ---")
            print(f"INSTRUCTION: {instruction}")
            print(f"COMMAND: Pan={pan:.1f}, Tilt={tilt:.1f}")
            
            input("Press Enter to execute this step...")
            
            self.move_servos(pan, tilt, f"Step {i+1}")
            
            response = input("Did the servo(s) move as expected? (y/n): ").lower().strip()
            if response.startswith('n'):
                print("⚠️ Servo did not move as expected!")
                print("💡 Check wiring and power supply")
            else:
                print("✅ Step completed successfully")
    
    def cleanup(self):
        """Clean shutdown"""
        print("\n🛑 Cleaning up...")
        
        if GPIO_AVAILABLE and self.pan_servo and self.tilt_servo:
            try:
                # Center both servos
                self.pan_servo.value = 0
                self.tilt_servo.value = 0
                time.sleep(1)
                
                # Close connections
                self.pan_servo.close()
                self.tilt_servo.close()
                print("✅ Servos centered and closed")
            except Exception as e:
                print(f"⚠️ Cleanup warning: {e}")
        
        print("✅ Native servo test complete!")

def main():
    print("🔧 NATIVE SERVO TEST (Pi 5 Compatible)")
    print("No pigpio daemon required!")
    print("")
    print("Available tests:")
    print("1. Basic positions (recommended first test)")
    print("2. Smooth movement") 
    print("3. Interactive step-by-step")
    print("4. All tests")
    
    try:
        choice = input("\nChoose test (1-4): ").strip()
        
        servo_test = NativeServoTest()
        
        if choice == "1":
            servo_test.test_basic_positions()
        elif choice == "2":
            servo_test.test_smooth_movement()
        elif choice == "3":
            servo_test.test_step_by_step()
        elif choice == "4":
            print("\n🚀 Running all tests...")
            servo_test.test_basic_positions()
            input("\nPress Enter for smooth movement test...")
            servo_test.test_smooth_movement()
            input("\nPress Enter for interactive test...")
            servo_test.test_step_by_step()
        else:
            print("Invalid choice, running basic positions test...")
            servo_test.test_basic_positions()
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted")
    
    finally:
        if 'servo_test' in locals():
            servo_test.cleanup()

if __name__ == "__main__":
    main()