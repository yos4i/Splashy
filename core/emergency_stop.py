#!/usr/bin/env python3
"""Emergency stop script for water gun system"""

import time
from servo_controller import ServoController, ServoConfig

def emergency_stop():
    """Stop all servo movement immediately"""
    print("🛑 EMERGENCY STOP - Stopping all servos...")
    
    try:
        config = ServoConfig()
        servo = ServoController(config)
        servo.start_control_loop()
        
        # Center servos
        servo.move_to_center()
        time.sleep(1)
        
        # Clean shutdown
        servo.cleanup()
        print("✅ All servos stopped and centered")
        
    except Exception as e:
        print(f"Error during emergency stop: {e}")
        print("Servos should still be stopped")

if __name__ == "__main__":
    emergency_stop()