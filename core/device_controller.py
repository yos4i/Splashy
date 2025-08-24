#!/usr/bin/env python3
import time
import threading
import logging
import queue
from typing import Optional
from dataclasses import dataclass
from enum import Enum

try:
    from gpiozero import LED, Button, Device
    from gpiozero.pins.pigpio import PiGPIOFactory
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logging.warning("GPIO libraries not available - running in simulation mode")

class DeviceState(Enum):
    OFF = 0
    ON = 1
    ERROR = 2

@dataclass
class DeviceConfig:
    # GPIO pins based on your specifications
    pump_pin: int = 17      # GPIO17 - Physical pin 11
    ir_led_pin: int = 27    # GPIO27 - Physical pin 13  
    pir_sensor_pin: int = 22 # GPIO22 - Physical pin 15
    
    # Pump configuration
    pump_max_duration: float = 5.0  # Maximum continuous run time (seconds)
    pump_cooldown: float = 2.0      # Minimum time between pump activations
    
    # IR LED configuration
    ir_auto_enable: bool = True     # Automatically enable IR in low light
    ir_threshold: float = 0.3       # Light threshold for auto IR (0-1)
    
    # PIR sensor configuration
    pir_sensitivity_delay: float = 0.5  # Debounce delay for PIR sensor
    pir_motion_timeout: float = 10.0    # Time to wait after motion before sleep

class DeviceController:
    def __init__(self, config: DeviceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize GPIO factory
        if GPIO_AVAILABLE:
            try:
                Device.pin_factory = PiGPIOFactory()
                self.logger.info("Using pigpio for device control")
            except Exception as e:
                self.logger.warning(f"Failed to initialize pigpio: {e}")
        
        # Device states
        self.pump_state = DeviceState.OFF
        self.ir_state = DeviceState.OFF
        self.motion_detected = False
        self.last_motion_time = 0
        self.last_pump_time = 0
        
        # Initialize devices
        self.pump = None
        self.ir_led = None
        self.pir_sensor = None
        self.initialize_devices()
        
        # Control threads
        self.command_queue = queue.Queue()
        self.control_thread = None
        self.running = False
        
        # Statistics
        self.pump_activations = 0
        self.total_pump_runtime = 0
        self.motion_detections = 0
    
    def initialize_devices(self):
        """Initialize GPIO devices"""
        if not GPIO_AVAILABLE:
            self.logger.info("Device controller running in simulation mode")
            return
        
        try:
            # Initialize pump control (via MOSFET/Relay)
            self.pump = LED(self.config.pump_pin)
            self.pump.off()
            self.logger.info(f"Pump initialized on GPIO{self.config.pump_pin}")
            
            # Initialize IR LED
            self.ir_led = LED(self.config.ir_led_pin)
            self.ir_led.off()
            self.logger.info(f"IR LED initialized on GPIO{self.config.ir_led_pin}")
            
            # Initialize PIR sensor
            self.pir_sensor = Button(self.config.pir_sensor_pin, pull_up=False)
            self.pir_sensor.when_pressed = self._on_motion_detected
            self.pir_sensor.when_released = self._on_motion_stopped
            self.logger.info(f"PIR sensor initialized on GPIO{self.config.pir_sensor_pin}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize devices: {e}")
            self.pump = None
            self.ir_led = None
            self.pir_sensor = None
    
    def _on_motion_detected(self):
        """Callback for PIR sensor motion detection"""
        current_time = time.time()
        if current_time - self.last_motion_time > self.config.pir_sensitivity_delay:
            self.motion_detected = True
            self.last_motion_time = current_time
            self.motion_detections += 1
            self.logger.info("Motion detected!")
            
            # Queue motion detection event
            command = {
                'action': 'motion_detected',
                'timestamp': current_time
            }
            
            try:
                self.command_queue.put_nowait(command)
            except queue.Full:
                pass
    
    def _on_motion_stopped(self):
        """Callback for PIR sensor motion stopped"""
        self.logger.debug("PIR sensor signal ended")
    
    def fire_pump(self, duration: float = 1.0) -> bool:
        """Activate water pump for specified duration"""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_pump_time < self.config.pump_cooldown:
            self.logger.warning(f"Pump activation denied - cooldown period active")
            return False
        
        # Check maximum duration
        if duration > self.config.pump_max_duration:
            duration = self.config.pump_max_duration
            self.logger.warning(f"Pump duration limited to {duration} seconds")
        
        command = {
            'action': 'pump_fire',
            'duration': duration,
            'timestamp': current_time
        }
        
        try:
            self.command_queue.put_nowait(command)
            return True
        except queue.Full:
            self.logger.error("Command queue full - pump activation failed")
            return False
    
    def _execute_pump_fire(self, duration: float):
        """Execute pump firing sequence"""
        if self.pump_state != DeviceState.OFF:
            self.logger.warning("Pump already active")
            return
        
        try:
            self.pump_state = DeviceState.ON
            self.last_pump_time = time.time()
            self.pump_activations += 1
            
            if GPIO_AVAILABLE and self.pump:
                self.pump.on()
                self.logger.info(f"Pump ON for {duration:.1f} seconds")
            else:
                self.logger.info(f"[SIMULATION] Pump ON for {duration:.1f} seconds")
            
            # Wait for duration
            time.sleep(duration)
            
            # Turn off pump
            if GPIO_AVAILABLE and self.pump:
                self.pump.off()
            
            self.pump_state = DeviceState.OFF
            self.total_pump_runtime += duration
            self.logger.info("Pump OFF")
            
        except Exception as e:
            self.logger.error(f"Error during pump operation: {e}")
            self.pump_state = DeviceState.ERROR
            if GPIO_AVAILABLE and self.pump:
                self.pump.off()
    
    def set_ir_led(self, enabled: bool):
        """Control IR LED state"""
        command = {
            'action': 'ir_control',
            'enabled': enabled,
            'timestamp': time.time()
        }
        
        try:
            self.command_queue.put_nowait(command)
        except queue.Full:
            pass
    
    def _execute_ir_control(self, enabled: bool):
        """Execute IR LED control"""
        try:
            if enabled:
                if GPIO_AVAILABLE and self.ir_led:
                    self.ir_led.on()
                else:
                    self.logger.info("[SIMULATION] IR LED ON")
                self.ir_state = DeviceState.ON
                self.logger.info("IR illuminator enabled")
            else:
                if GPIO_AVAILABLE and self.ir_led:
                    self.ir_led.off()
                else:
                    self.logger.info("[SIMULATION] IR LED OFF")
                self.ir_state = DeviceState.OFF
                self.logger.info("IR illuminator disabled")
                
        except Exception as e:
            self.logger.error(f"Error controlling IR LED: {e}")
            self.ir_state = DeviceState.ERROR
    
    def toggle_ir_led(self):
        """Toggle IR LED state"""
        enabled = self.ir_state != DeviceState.ON
        self.set_ir_led(enabled)
    
    def get_motion_status(self) -> dict:
        """Get current motion detection status"""
        return {
            'motion_detected': self.motion_detected,
            'last_motion_time': self.last_motion_time,
            'time_since_motion': time.time() - self.last_motion_time,
            'total_detections': self.motion_detections
        }
    
    def get_pump_status(self) -> dict:
        """Get current pump status"""
        current_time = time.time()
        return {
            'state': self.pump_state.name,
            'last_activation': self.last_pump_time,
            'time_since_last': current_time - self.last_pump_time,
            'cooldown_remaining': max(0, self.config.pump_cooldown - (current_time - self.last_pump_time)),
            'can_activate': (current_time - self.last_pump_time) >= self.config.pump_cooldown and self.pump_state == DeviceState.OFF,
            'total_activations': self.pump_activations,
            'total_runtime': self.total_pump_runtime
        }
    
    def get_ir_status(self) -> dict:
        """Get current IR LED status"""
        return {
            'state': self.ir_state.name,
            'enabled': self.ir_state == DeviceState.ON
        }
    
    def get_device_status(self) -> dict:
        """Get comprehensive device status"""
        return {
            'pump': self.get_pump_status(),
            'ir_led': self.get_ir_status(),
            'motion': self.get_motion_status(),
            'timestamp': time.time()
        }
    
    def emergency_stop(self):
        """Emergency stop all devices"""
        command = {
            'action': 'emergency_stop',
            'timestamp': time.time()
        }
        
        try:
            self.command_queue.put_nowait(command)
        except queue.Full:
            # Force immediate stop if queue is full
            self._execute_emergency_stop()
    
    def _execute_emergency_stop(self):
        """Execute emergency stop procedure"""
        self.logger.warning("EMERGENCY STOP activated")
        
        try:
            # Stop pump immediately
            if GPIO_AVAILABLE and self.pump:
                self.pump.off()
            self.pump_state = DeviceState.OFF
            
            # Turn off IR LED
            if GPIO_AVAILABLE and self.ir_led:
                self.ir_led.off()
            self.ir_state = DeviceState.OFF
            
            self.logger.info("All devices stopped")
            
        except Exception as e:
            self.logger.error(f"Error during emergency stop: {e}")
    
    def start_control_loop(self):
        """Start the device control loop"""
        if self.running:
            return
        
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop_worker)
        self.control_thread.start()
        self.logger.info("Device control loop started")
    
    def _control_loop_worker(self):
        """Main device control loop"""
        while self.running:
            try:
                # Process commands from queue
                try:
                    command = self.command_queue.get(timeout=0.1)
                    self._process_command(command)
                except queue.Empty:
                    continue
                
                # Auto IR control based on time (simple day/night detection)
                if self.config.ir_auto_enable:
                    self._auto_ir_control()
                
                # Motion timeout handling
                self._handle_motion_timeout()
                
            except Exception as e:
                self.logger.error(f"Error in device control loop: {e}")
                time.sleep(0.1)
    
    def _process_command(self, command: dict):
        """Process a device command"""
        action = command.get('action')
        
        if action == 'pump_fire':
            duration = command.get('duration', 1.0)
            self._execute_pump_fire(duration)
        
        elif action == 'ir_control':
            enabled = command.get('enabled', False)
            self._execute_ir_control(enabled)
        
        elif action == 'emergency_stop':
            self._execute_emergency_stop()
        
        elif action == 'motion_detected':
            # Handle motion detection events (could trigger actions)
            pass
    
    def _auto_ir_control(self):
        """Automatic IR LED control based on ambient light"""
        current_hour = time.localtime().tm_hour
        
        # Simple day/night detection (enable IR during night hours)
        night_time = current_hour < 6 or current_hour > 20
        
        if night_time and self.ir_state == DeviceState.OFF:
            self.set_ir_led(True)
        elif not night_time and self.ir_state == DeviceState.ON:
            self.set_ir_led(False)
    
    def _handle_motion_timeout(self):
        """Handle motion detection timeout"""
        if self.motion_detected:
            current_time = time.time()
            if current_time - self.last_motion_time > self.config.pir_motion_timeout:
                self.motion_detected = False
                self.logger.debug("Motion detection timeout")
    
    def stop_control_loop(self):
        """Stop the device control loop"""
        self.running = False
        
        if self.control_thread:
            self.control_thread.join(timeout=2.0)
        
        self.logger.info("Device control loop stopped")
    
    def cleanup(self):
        """Clean up device controller"""
        self.stop_control_loop()
        
        # Turn off all devices
        try:
            if GPIO_AVAILABLE:
                if self.pump:
                    self.pump.off()
                    self.pump.close()
                if self.ir_led:
                    self.ir_led.off()
                    self.ir_led.close()
                if self.pir_sensor:
                    self.pir_sensor.close()
        except Exception as e:
            self.logger.error(f"Error during device cleanup: {e}")

class DeviceTester:
    """Test utility for device controller"""
    
    def __init__(self, device_controller: DeviceController):
        self.device = device_controller
    
    def test_pump(self):
        """Test pump functionality"""
        print("Testing pump...")
        status = self.device.get_pump_status()
        
        if status['can_activate']:
            print("Activating pump for 1 second...")
            success = self.device.fire_pump(1.0)
            if success:
                time.sleep(2)  # Wait for completion + cooldown
                print("Pump test completed")
            else:
                print("Pump activation failed")
        else:
            print(f"Pump cannot be activated (cooldown: {status['cooldown_remaining']:.1f}s)")
    
    def test_ir_led(self):
        """Test IR LED functionality"""
        print("Testing IR LED...")
        
        print("Turning IR LED ON...")
        self.device.set_ir_led(True)
        time.sleep(2)
        
        print("Turning IR LED OFF...")
        self.device.set_ir_led(False)
        time.sleep(1)
        
        print("IR LED test completed")
    
    def test_motion_sensor(self):
        """Test motion sensor"""
        print("Testing motion sensor...")
        print("Wave your hand in front of the PIR sensor...")
        
        start_time = time.time()
        while time.time() - start_time < 10:
            motion_status = self.device.get_motion_status()
            if motion_status['motion_detected']:
                print(f"Motion detected! Total detections: {motion_status['total_detections']}")
            time.sleep(0.5)
        
        print("Motion sensor test completed")
    
    def print_status(self):
        """Print comprehensive device status"""
        status = self.device.get_device_status()
        
        print("\n--- Device Status ---")
        print(f"Pump: {status['pump']['state']} | Activations: {status['pump']['total_activations']} | Runtime: {status['pump']['total_runtime']:.1f}s")
        print(f"IR LED: {status['ir_led']['state']}")
        print(f"Motion: {'DETECTED' if status['motion']['motion_detected'] else 'NONE'} | Total: {status['motion']['total_detections']}")
        print(f"Time since last motion: {status['motion']['time_since_motion']:.1f}s")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = DeviceConfig()
    device_controller = DeviceController(config)
    
    try:
        device_controller.start_control_loop()
        tester = DeviceTester(device_controller)
        
        print("Device Controller Test")
        print("Commands:")
        print("  p - Test pump")
        print("  i - Test IR LED")
        print("  m - Test motion sensor")
        print("  f - Fire pump (1s)")
        print("  t - Toggle IR LED")
        print("  s - Print status")
        print("  e - Emergency stop")
        print("  q - Quit")
        
        while True:
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'q':
                break
            elif command == 'p':
                tester.test_pump()
            elif command == 'i':
                tester.test_ir_led()
            elif command == 'm':
                tester.test_motion_sensor()
            elif command == 'f':
                success = device_controller.fire_pump(1.0)
                print(f"Pump fire: {'SUCCESS' if success else 'FAILED'}")
            elif command == 't':
                device_controller.toggle_ir_led()
            elif command == 's':
                tester.print_status()
            elif command == 'e':
                device_controller.emergency_stop()
                print("EMERGENCY STOP executed")
    
    finally:
        device_controller.cleanup()