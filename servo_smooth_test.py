#!/usr/bin/env python3
"""
Smooth Servo Test - Simple side-to-side movement to test servo stability
No face detection, no camera - just pure servo control testing
"""
import time
import sys
import os
import logging
import math

# Import the servo controller
sys.path.append('core')
from servo_controller import ServoController, ServoConfig

class SmoothServoTest:
    """Simple servo test with smooth, predictable movements"""
    
    def __init__(self):
        print("\n🔧 SMOOTH SERVO TEST")
        print("="*40)
        print("Testing servo controller with smooth movements")
        print("No cameras, no detection - just pure servo control")
        print("")
        
        # Initialize servo controller
        print("🎛️ Initializing servo controller...")
        logging.basicConfig(level=logging.INFO)
        self.servo_config = ServoConfig()
        self.servo = ServoController(self.servo_config)
        self.servo.start_control_loop()
        
        # Movement parameters
        self.movement_speed = 0.02  # How fast to move (smaller = slower)
        self.pause_time = 0.1       # Pause between movements
        self.pan_range = 0.8        # How far to move pan (-0.8 to +0.8)
        self.tilt_range = 0.4       # How far to move tilt (-0.4 to +0.4)
        
        print(f"✅ Servo controller ready!")
        print(f"🎯 Pan GPIO: {self.servo_config.pan_pin}")
        print(f"🎯 Tilt GPIO: {self.servo_config.tilt_pin}")
        print(f"📐 Movement speed: {self.movement_speed}")
        print(f"📏 Pan range: ±{self.pan_range}")
        print(f"📏 Tilt range: ±{self.tilt_range}")
        print("="*40)
    
    def smooth_move_to(self, target_pan, target_tilt, steps=20):
        """Move smoothly to target position in small steps"""
        current_pan = self.servo.current_pan
        current_tilt = self.servo.current_tilt
        
        print(f"🎯 Smooth move: Pan {current_pan:.3f}→{target_pan:.3f} | Tilt {current_tilt:.3f}→{target_tilt:.3f}")
        
        # Calculate step sizes
        pan_step = (target_pan - current_pan) / steps
        tilt_step = (target_tilt - current_tilt) / steps
        
        # Move in small increments
        for i in range(steps):
            new_pan = current_pan + (pan_step * (i + 1))
            new_tilt = current_tilt + (tilt_step * (i + 1))
            
            # Clamp to limits
            new_pan = max(-1, min(1, new_pan))
            new_tilt = max(-1, min(1, new_tilt))
            
            # Set position
            self.servo.set_position(new_pan, new_tilt)
            
            # Small pause for smooth movement
            time.sleep(0.05)  # 50ms between micro-steps
        
        # Final position
        self.servo.set_position(target_pan, target_tilt)
        print(f"✅ Reached target: Pan {target_pan:.3f} | Tilt {target_tilt:.3f}")
    
    def test_simple_movements(self):
        """Test basic servo movements"""
        print("\n📐 BASIC MOVEMENT TEST")
        print("-" * 30)
        
        # Center first
        print("🎯 Moving to center...")
        self.servo.move_to_center()
        time.sleep(2)
        
        # Simple positions
        positions = [
            (0.5, 0),     # Right
            (0, 0.3),     # Up  
            (-0.5, 0),    # Left
            (0, -0.3),    # Down
            (0, 0)        # Center
        ]
        
        for i, (pan, tilt) in enumerate(positions):
            print(f"\n🎯 Position {i+1}: Pan={pan}, Tilt={tilt}")
            self.smooth_move_to(pan, tilt)
            time.sleep(1.5)  # Pause at each position
    
    def test_sine_wave(self, duration=30):
        """Test smooth sine wave movement"""
        print(f"\n🌊 SINE WAVE TEST ({duration} seconds)")
        print("-" * 30)
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            
            # Sine wave for pan (side to side)
            pan_position = self.pan_range * math.sin(elapsed * 0.5)  # 0.5 Hz frequency
            
            # Cosine wave for tilt (up and down) - slower
            tilt_position = self.tilt_range * math.cos(elapsed * 0.3) * 0.5  # 0.3 Hz, smaller amplitude
            
            # Set position
            self.servo.set_position(pan_position, tilt_position)
            
            # Show current position every 2 seconds
            if int(elapsed) % 2 == 0 and int(elapsed * 10) % 20 == 0:
                print(f"🌊 Wave: Pan={pan_position:.3f}, Tilt={tilt_position:.3f} | Time: {elapsed:.1f}s")
            
            time.sleep(0.1)  # 10 Hz update rate
        
        # Return to center
        print("🎯 Returning to center...")
        self.smooth_move_to(0, 0)
    
    def test_step_response(self):
        """Test how servo responds to step changes"""
        print("\n📊 STEP RESPONSE TEST")
        print("-" * 30)
        
        positions = [
            (0, 0),       # Center
            (0.8, 0),     # Large pan step
            (0.8, 0.4),   # Add tilt
            (-0.8, 0.4),  # Large pan change
            (-0.8, -0.4), # Add tilt change
            (0, 0)        # Back to center
        ]
        
        for i, (pan, tilt) in enumerate(positions):
            print(f"\n📊 Step {i+1}: Pan={pan}, Tilt={tilt}")
            
            # Immediate position change (no smoothing)
            self.servo.set_position(pan, tilt)
            
            # Monitor for jitter/oscillation for 3 seconds
            start_monitor = time.time()
            last_position = (self.servo.current_pan, self.servo.current_tilt)
            jitter_count = 0
            
            while time.time() - start_monitor < 3.0:
                current_position = (self.servo.current_pan, self.servo.current_tilt)
                
                # Check for position changes (potential jitter)
                if abs(current_position[0] - last_position[0]) > 0.001 or abs(current_position[1] - last_position[1]) > 0.001:
                    jitter_count += 1
                    if jitter_count <= 5:  # Show first few jitter events
                        print(f"⚠️ Position change detected: {last_position} → {current_position}")
                
                last_position = current_position
                time.sleep(0.1)
            
            if jitter_count == 0:
                print("✅ Position stable - no jitter detected")
            else:
                print(f"⚠️ Detected {jitter_count} position changes (potential jitter)")
    
    def run_all_tests(self):
        """Run complete servo test suite"""
        print("\n🚀 STARTING COMPLETE SERVO TEST SUITE")
        print("="*50)
        
        try:
            # Test 1: Basic movements
            self.test_simple_movements()
            
            print("\n" + "="*50)
            input("Press Enter to continue to sine wave test (or Ctrl+C to exit)...")
            
            # Test 2: Sine wave
            self.test_sine_wave(duration=20)
            
            print("\n" + "="*50)
            input("Press Enter to continue to step response test (or Ctrl+C to exit)...")
            
            # Test 3: Step response
            self.test_step_response()
            
        except KeyboardInterrupt:
            print("\n⏹️ Test interrupted by user")
        
        finally:
            print("\n🛑 Cleaning up...")
            self.servo.move_to_center()
            time.sleep(1)
            self.servo.cleanup()
            print("✅ Servo test complete!")
    
    def run_continuous_side_to_side(self):
        """Simple continuous side-to-side movement"""
        print("\n🔄 CONTINUOUS SIDE-TO-SIDE TEST")
        print("=" * 40)
        print("Press Ctrl+C to stop")
        print("")
        
        # Center first
        self.servo.move_to_center()
        time.sleep(1)
        
        try:
            while True:
                # Move to right
                print("👉 Moving right...")
                self.smooth_move_to(self.pan_range, 0)
                time.sleep(1)
                
                # Move to left
                print("👈 Moving left...")
                self.smooth_move_to(-self.pan_range, 0)
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        
        finally:
            print("🎯 Returning to center...")
            self.servo.move_to_center()
            time.sleep(1)
            self.servo.cleanup()
            print("✅ Test complete!")

def main():
    print("🔧 SERVO SMOOTH TEST UTILITY")
    print("This will test servo movements without any camera/detection")
    print("")
    print("Available tests:")
    print("1. Complete test suite (recommended)")
    print("2. Continuous side-to-side only")
    print("3. Basic movements only")
    print("4. Sine wave only")
    print("5. Step response only")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
        
        servo_test = SmoothServoTest()
        
        if choice == "1":
            servo_test.run_all_tests()
        elif choice == "2":
            servo_test.run_continuous_side_to_side()
        elif choice == "3":
            servo_test.test_simple_movements()
        elif choice == "4":
            servo_test.test_sine_wave()
        elif choice == "5":
            servo_test.test_step_response()
        else:
            print("Invalid choice. Running complete test suite...")
            servo_test.run_all_tests()
            
    except KeyboardInterrupt:
        print("\n⏹️ Exiting...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()