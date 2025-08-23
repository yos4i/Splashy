#!/usr/bin/env python3
"""
GPIO Debug Tool - Test basic GPIO functionality and servo connections
"""
import time
import sys

# Test GPIO availability
print("🔍 GPIO DEBUG TOOL")
print("="*40)

try:
    import RPi.GPIO as GPIO
    RPI_GPIO_AVAILABLE = True
    print("✅ RPi.GPIO available")
except ImportError:
    RPI_GPIO_AVAILABLE = False
    print("❌ RPi.GPIO not available")

try:
    from gpiozero import Servo, LED, Device
    from gpiozero.pins.pigpio import PiGPIOFactory
    GPIOZERO_AVAILABLE = True
    print("✅ gpiozero available")
except ImportError:
    GPIOZERO_AVAILABLE = False
    print("❌ gpiozero not available")

try:
    import pigpio
    PIGPIO_AVAILABLE = True
    print("✅ pigpio available")
except ImportError:
    PIGPIO_AVAILABLE = False
    print("❌ pigpio not available")

print("="*40)

def test_gpio_basic():
    """Test basic GPIO functionality"""
    print("\n🧪 BASIC GPIO TEST")
    print("-" * 20)
    
    if not RPI_GPIO_AVAILABLE:
        print("❌ RPi.GPIO not available - skipping")
        return
    
    # Test GPIO setup
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Test pins we're using for servos
        test_pins = [18, 23]  # Pan and tilt pins
        
        for pin in test_pins:
            print(f"🔍 Testing GPIO{pin}...")
            
            # Set as output
            GPIO.setup(pin, GPIO.OUT)
            
            # Test high/low
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(pin, GPIO.LOW) 
            time.sleep(0.5)
            
            print(f"✅ GPIO{pin} basic test passed")
        
        GPIO.cleanup()
        print("✅ Basic GPIO test completed")
        
    except Exception as e:
        print(f"❌ Basic GPIO test failed: {e}")
        try:
            GPIO.cleanup()
        except:
            pass

def test_pigpio_daemon():
    """Test if pigpio daemon is running"""
    print("\n🔧 PIGPIO DAEMON TEST")
    print("-" * 20)
    
    if not PIGPIO_AVAILABLE:
        print("❌ pigpio not available")
        return
    
    try:
        pi = pigpio.pi()
        
        if not pi.connected:
            print("❌ pigpio daemon not running")
            print("💡 Try: sudo systemctl start pigpiod")
            print("💡 Or: sudo pigpiod")
            return
        
        print("✅ pigpio daemon is running")
        
        # Test servo pins
        test_pins = [18, 23]
        
        for pin in test_pins:
            print(f"🔍 Testing GPIO{pin} with pigpio...")
            
            # Test basic PWM
            pi.set_servo_pulsewidth(pin, 1500)  # Center position
            time.sleep(1)
            pi.set_servo_pulsewidth(pin, 1000)  # One extreme
            time.sleep(1) 
            pi.set_servo_pulsewidth(pin, 2000)  # Other extreme
            time.sleep(1)
            pi.set_servo_pulsewidth(pin, 1500)  # Back to center
            time.sleep(1)
            pi.set_servo_pulsewidth(pin, 0)     # Turn off
            
            print(f"✅ GPIO{pin} pigpio test completed")
        
        pi.stop()
        print("✅ pigpio test completed")
        
    except Exception as e:
        print(f"❌ pigpio test failed: {e}")

def test_gpiozero_servo():
    """Test gpiozero servo with different configurations"""
    print("\n🎛️ GPIOZERO SERVO TEST")
    print("-" * 20)
    
    if not GPIOZERO_AVAILABLE:
        print("❌ gpiozero not available")
        return
    
    # Test with different pin factories
    factories = [
        ("default", None),
        ("pigpio", PiGPIOFactory)
    ]
    
    for factory_name, factory_class in factories:
        print(f"\n🔧 Testing with {factory_name} factory...")
        
        try:
            if factory_class:
                Device.pin_factory = factory_class()
            
            # Test servo on pin 18 (pan)
            print("🎯 Creating servo on GPIO18...")
            servo = Servo(18, min_pulse_width=0.5e-3, max_pulse_width=2.5e-3)
            
            print("🔄 Testing servo positions...")
            positions = [0, 1, -1, 0]  # Center, Max, Min, Center
            
            for i, pos in enumerate(positions):
                print(f"   Position {i+1}: {pos}")
                servo.value = pos
                time.sleep(2)  # Wait 2 seconds at each position
            
            servo.close()
            print(f"✅ {factory_name} factory test completed")
            
        except Exception as e:
            print(f"❌ {factory_name} factory test failed: {e}")

def test_servo_power():
    """Test servo power requirements"""
    print("\n⚡ SERVO POWER TEST")
    print("-" * 20)
    print("🔍 Checking power-related issues...")
    print("")
    print("SERVO POWER CHECKLIST:")
    print("□ External 5V power supply connected?")
    print("□ Servos connected to external power, not Pi 5V rail?")
    print("□ Common ground between Pi and servo power supply?")
    print("□ Power supply adequate (5V, 3A+ for servos)?")
    print("□ Servo signal wires connected to correct GPIO pins?")
    print("")
    print("WIRING CHECK:")
    print("Pan Servo:")
    print("  - Red (VCC) → External 5V")
    print("  - Black (GND) → Common ground")
    print("  - Signal → GPIO18 (Pin 12)")
    print("")
    print("Tilt Servo:")
    print("  - Red (VCC) → External 5V")  
    print("  - Black (GND) → Common ground")
    print("  - Signal → GPIO23 (Pin 16)")

def run_comprehensive_test():
    """Run all diagnostic tests"""
    print("🚀 COMPREHENSIVE GPIO/SERVO DIAGNOSTIC")
    print("="*50)
    
    test_gpio_basic()
    input("\nPress Enter to continue to pigpio test...")
    
    test_pigpio_daemon() 
    input("\nPress Enter to continue to gpiozero test...")
    
    test_gpiozero_servo()
    print("\n" + "="*50)
    
    test_servo_power()
    
    print("\n" + "="*50)
    print("🔍 DIAGNOSTIC COMPLETE")
    print("")
    print("NEXT STEPS:")
    print("1. If pigpio daemon not running: sudo systemctl start pigpiod")
    print("2. Check all wiring per the power checklist above")
    print("3. Verify external power supply is connected and adequate")
    print("4. Test servos with a simple servo tester if available")

def main():
    print("Select test:")
    print("1. Basic GPIO test")
    print("2. Pigpio daemon test")  
    print("3. Gpiozero servo test")
    print("4. Power/wiring checklist")
    print("5. Comprehensive test (all)")
    
    try:
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == "1":
            test_gpio_basic()
        elif choice == "2":
            test_pigpio_daemon()
        elif choice == "3":
            test_gpiozero_servo()
        elif choice == "4":
            test_servo_power()
        elif choice == "5":
            run_comprehensive_test()
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted")
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    main()