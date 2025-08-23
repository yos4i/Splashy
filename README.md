# 💦 Splashy - Smart Water Gun Turret System

Splashy is a comprehensive automated water gun turret system with computer vision tracking, web-based remote control, and precise servo-driven aiming for Raspberry Pi 5 running Debian.

## 🌟 Features

- **🎥 Computer Vision Detection**: Face, body, and color detection using OpenCV + libcamera
- **🎯 PID Tracking**: Smooth servo control with PID loops for accurate targeting  
- **🎮 Web Remote Control**: Modern web interface with real-time video streaming
- **⚡ Hardware Control**: Servo pan/tilt, water pump, IR illuminator, PIR motion sensor
- **🔧 Always-On Service**: Systemd integration for automatic startup and monitoring
- **📱 Mobile Friendly**: Responsive web interface works on phones and tablets
- **🛡️ Safety Features**: Emergency stops, cooldown periods, and security restrictions
- **📹 Ultra HD Camera**: Up to 4608x2592 (12MP) maximum resolution support

## 🔧 Hardware Requirements

### Components
- **Raspberry Pi 5** (4GB+ recommended) with Debian OS
- **Pi Camera Module v3** (12MP, autofocus) - recommended for best performance
- **2x Servo Motors** (MG996R or similar) for pan/tilt mechanism
- **Water Pump** (12V DC submersible pump)
- **PIR Motion Sensor** (HC-SR501 or similar)
- **IR LED Array** (850nm for night vision, optional)
- **Power Supply** (5V/5A for Pi + separate 5V/3A for servos + 12V for pump)
- **MOSFET/Relay Module** for pump control

### Wiring (Raspberry Pi 5 - 40-pin header)

| Device | Signal | GPIO | Physical Pin | Notes |
|--------|--------|------|--------------|-------|
| Pan Servo | PWM | GPIO12 (PWM0) | Pin 32 | Stable hardware PWM |
| Tilt Servo | PWM | GPIO13 (PWM1) | Pin 33 | Stable hardware PWM |  
| Pump Control | Digital | GPIO17 | Pin 11 | HIGH = ON (via MOSFET/Relay) |
| IR LED | Digital | GPIO27 | Pin 13 | HIGH = IR LEDs ON |
| PIR Sensor | Input | GPIO22 | Pin 15 | HIGH when motion detected |
| 5V Servos | VCC | — | Pin 2/4 | Prefer external 5V supply |
| Common GND | GND | — | Pin 6/9/14/20/25/30/34/39 | Mandatory shared ground |

⚠️ **Important**: Do not power servos/pump directly from Pi's 5V rail. Use external supplies with common ground.

## 📁 Project Structure

```
splashy/
├── tracker.py               # 🚀 Ultra HD face tracker (main application)
├── final_tracker.py         # Alternative advanced tracker
├── core/                    # Core system files
│   ├── splashy_system.py    # Main system controller
│   ├── servo_controller.py  # Servo control logic
│   ├── device_controller.py # Pump, IR, PIR control
│   └── emergency_stop.py    # Safety systems
├── tests/                   # Testing and calibration
│   ├── camera_test.py       # Basic camera test
│   └── improved_camera.py   # High-performance streaming
├── web/                     # Web interface
│   ├── web_interface.py     # Flask web server
│   ├── web_camera_viewer.py # Camera streaming
│   └── index.html          # Web dashboard
├── requirements.txt         # Python dependencies
├── install.sh              # Installation script
├── splashy.service          # Systemd service
└── venv/                   # Python virtual environment
```

## 🚀 Quick Installation (Raspberry Pi Debian)

### Prerequisites
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install python3-pip python3-venv libcamera-apps -y
```

### Installation Steps

1. **Navigate to project directory**:
   ```bash
   cd /home/yossi/Downloads/splashy
   ```

2. **Setup Python virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Test camera connection**:
   ```bash
   # Quick test
   libcamera-hello --list-cameras
   
   # Basic camera test
   python3 tests/camera_test.py
   ```

4. **Run live camera test**:
   ```bash
   # 10-second test
   python3 tests/quick_camera_test.py
   
   # 60-second test with green detection
   python3 tests/live_camera_test.py
   ```

## 🎮 Usage

### Camera Testing Commands

```bash
# Activate virtual environment first
source venv/bin/activate

# Quick 10-second test
python3 tests/quick_camera_test.py

# Live camera with object detection (60 seconds)
python3 tests/live_camera_test.py

# Test different camera backends
python3 tests/opencv_camera_test.py

# Performance testing
python3 tests/improved_camera.py
```

### Web Interface
```bash
# Start web interface
python3 web/web_interface.py

# Access dashboard at: http://[PI_IP_ADDRESS]:5000
```

### Main System
```bash
# Run ultra HD face tracker (recommended)
python3 tracker.py

# Or run main splashy system
python3 core/splashy_system.py
```

## 📹 Camera Configuration

The system now supports high-definition camera capture:

- **Default Resolution**: 1280x720 (HD)
- **Supported Resolutions**: 640x480, 1280x720, 1920x1080
- **Camera Interface**: libcamera (optimized for Pi 5)
- **Detection**: OpenCV-based green object detection
- **Performance**: ~1.3 FPS with libcamera-still (can be optimized to 30+ FPS with libcamera-vid)

### Camera Troubleshooting

**Camera not detected**:
```bash
# Check camera connection
libcamera-hello --list-cameras

# Test basic capture
libcamera-still --output test.jpg --timeout 1000
```

**Camera connection issues**:
- Ensure camera ribbon cable is properly connected to Pi 5
- Check that camera is enabled in raspi-config
- Verify camera module is compatible with Pi 5

**Performance optimization**:
- For real-time video: Replace libcamera-still with libcamera-vid
- For better FPS: Use GStreamer pipeline
- For lower latency: Reduce resolution or processing complexity

## ⚙️ Configuration

### Camera Settings
- **Resolution**: Configurable in camera test files
- **FPS Target**: 30+ FPS with optimized streaming
- **Detection**: Green object tracking with adjustable HSV ranges
- **Autofocus**: Hardware autofocus supported on Pi Camera v3

### System Configuration
The main configuration is in `core/splashy_system.py`:
```python
# Default camera configuration
frame_width: int = 1280    # HD resolution
frame_height: int = 720
fps_target: int = 30

# PID control parameters
kp_pan: float = 0.8
ki_pan: float = 0.1
kd_pan: float = 0.2
```

## 🧪 Testing Components

### Camera Tests
```bash
# Basic camera functionality
python3 tests/camera_test.py

# Live detection test (10 seconds)
python3 tests/quick_camera_test.py

# Extended live test (60 seconds)
python3 tests/live_camera_test.py

# Performance benchmarking
python3 tests/improved_camera.py
```

### System Tests
```bash
# Test servo control
python3 core/servo_controller.py

# Test devices (pump, IR, PIR)
python3 core/device_controller.py

# Full system test
python3 core/splashy_system.py
```

## 🛠️ Troubleshooting

### Camera Issues

**No camera detected**:
- Check physical connection of camera ribbon cable
- Ensure camera is properly seated in Pi 5 connector
- Run `libcamera-hello --list-cameras`

**Autofocus errors (dw9807 device)**:
- Usually indicates loose camera connection
- Disconnect and reconnect camera cable
- Ensure cable is inserted fully and locked

**Low FPS performance**:
- Current implementation uses libcamera-still (photo mode)
- For video streaming: implement libcamera-vid integration
- For OpenCV: use GStreamer backend with hardware acceleration

### Virtual Environment
```bash
# If venv activation fails
python3 -m venv venv --clear
source venv/bin/activate
pip install -r requirements.txt
```

### Permissions
```bash
# Camera access permissions
sudo usermod -a -G video $USER
sudo reboot
```

## 🎯 Performance Optimizations

### Current Performance
- **Resolution**: 1280x720 (HD) - 2.25x improvement over original 640x480
- **FPS**: ~1.3 FPS with current libcamera-still implementation
- **Detection**: Real-time green object detection and tracking

### Future Optimizations
1. **libcamera-vid Integration**: Stream video instead of individual photos
2. **GStreamer Pipeline**: Hardware-accelerated video processing
3. **Threading**: Separate capture and processing threads
4. **Resolution Scaling**: Dynamic resolution based on detection needs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Legal Disclaimer

This system is designed for entertainment and educational purposes. Users are responsible for:
- Complying with local laws and regulations
- Ensuring safe operation and appropriate use
- Proper installation and maintenance
- Respecting privacy and property rights

**Use responsibly and have fun! 🎯💦**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`) 
5. Open a Pull Request

---

**Built with ❤️ for Raspberry Pi 5 + Debian enthusiasts and makers worldwide! 🎯**