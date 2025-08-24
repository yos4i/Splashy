#!/bin/bash

# Water Gun System Installation Script for Raspberry Pi 5
# Run with: sudo bash install.sh

set -e

echo "🎯 Water Gun System Installation Script"
echo "========================================"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "❌ This script must be run as root (use sudo)"
   exit 1
fi

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo; then
    echo "⚠️  This script is designed for Raspberry Pi. Continue anyway? (y/N)"
    read -n 1 -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "📦 Updating system packages..."
apt update && apt upgrade -y

echo "📦 Installing system dependencies..."
apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libtiff5-dev \
    libpng-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libfontconfig1-dev \
    libcairo2-dev \
    libgdk-pixbuf2.0-dev \
    libpango1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    pigpio \
    pigpiod \
    git \
    curl \
    nginx

echo "🔧 Setting up pigpio daemon..."
systemctl enable pigpiod
systemctl start pigpiod

echo "📁 Creating installation directory..."
mkdir -p /opt/water-gun-system
chown pi:pi /opt/water-gun-system

echo "📋 Copying system files..."
cp water_gun_system.py /opt/water-gun-system/
cp servo_controller.py /opt/water-gun-system/
cp device_controller.py /opt/water-gun-system/
cp web_interface.py /opt/water-gun-system/
cp requirements.txt /opt/water-gun-system/
cp -r templates /opt/water-gun-system/

# Set permissions
chown -R pi:pi /opt/water-gun-system
chmod +x /opt/water-gun-system/*.py

echo "🐍 Setting up Python virtual environment..."
sudo -u pi python3 -m venv /opt/water-gun-system/venv
sudo -u pi /opt/water-gun-system/venv/bin/pip install --upgrade pip

echo "📦 Installing Python packages..."
sudo -u pi /opt/water-gun-system/venv/bin/pip install -r /opt/water-gun-system/requirements.txt

echo "🔧 Installing systemd service..."
cp water_gun_system.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable water_gun_system

echo "🌐 Setting up nginx reverse proxy..."
cat > /etc/nginx/sites-available/water-gun-system << EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        proxy_buffering off;
        proxy_redirect off;
    }

    location /socket.io/ {
        proxy_pass http://127.0.0.1:5000/socket.io/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Remove default nginx site and enable water gun system
rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/water-gun-system /etc/nginx/sites-enabled/
systemctl enable nginx
systemctl restart nginx

echo "🎥 Enabling camera interface..."
if ! grep -q "^camera_auto_detect=1" /boot/firmware/config.txt; then
    echo "camera_auto_detect=1" >> /boot/firmware/config.txt
fi

if ! grep -q "^dtoverlay=vc4-kms-v3d" /boot/firmware/config.txt; then
    echo "dtoverlay=vc4-kms-v3d" >> /boot/firmware/config.txt
fi

# Add user to video group
usermod -a -G video pi

echo "⚡ Setting up GPIO permissions..."
usermod -a -G gpio pi
usermod -a -G dialout pi

# Create udev rules for GPIO access
cat > /etc/udev/rules.d/99-gpio.rules << EOF
SUBSYSTEM=="gpio", GROUP="gpio", MODE="0664"
SUBSYSTEM=="gpio*", PROGRAM="/bin/sh -c 'find -L /sys/class/gpio/ -maxdepth 2 -exec chown root:gpio {} \; -exec chmod 664 {} \; || true'"
EOF

echo "🔧 Creating configuration files..."
sudo -u pi mkdir -p /opt/water-gun-system/config

cat > /opt/water-gun-system/config/system_config.json << EOF
{
    "tracking": {
        "frame_width": 640,
        "frame_height": 480,
        "deadzone_radius": 30,
        "kp_pan": 0.8,
        "ki_pan": 0.1,
        "kd_pan": 0.2,
        "kp_tilt": 0.8,
        "ki_tilt": 0.1,
        "kd_tilt": 0.2
    },
    "servos": {
        "pan_pin": 12,
        "tilt_pin": 13,
        "pan_min": -90,
        "pan_max": 90,
        "tilt_min": -30,
        "tilt_max": 60
    },
    "devices": {
        "pump_pin": 17,
        "ir_pin": 27,
        "pir_pin": 22,
        "pump_max_duration": 5.0,
        "pump_cooldown": 2.0
    },
    "web": {
        "host": "0.0.0.0",
        "port": 5000,
        "debug": false
    }
}
EOF

chown pi:pi /opt/water-gun-system/config/system_config.json

echo "📝 Creating log directory..."
mkdir -p /var/log/water-gun-system
chown pi:pi /var/log/water-gun-system

echo "🔧 Setting up log rotation..."
cat > /etc/logrotate.d/water-gun-system << EOF
/var/log/water-gun-system/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    su pi pi
}
EOF

echo "🎯 Creating convenience scripts..."

# Create start script
cat > /opt/water-gun-system/start.sh << 'EOF'
#!/bin/bash
cd /opt/water-gun-system
source venv/bin/activate
python3 web_interface.py
EOF

# Create stop script
cat > /opt/water-gun-system/stop.sh << 'EOF'
#!/bin/bash
sudo systemctl stop water_gun_system
EOF

# Create status script
cat > /opt/water-gun-system/status.sh << 'EOF'
#!/bin/bash
echo "Water Gun System Status"
echo "======================"
systemctl status water_gun_system --no-pager
echo ""
echo "Recent logs:"
journalctl -u water_gun_system --no-pager -n 10
EOF

chmod +x /opt/water-gun-system/*.sh
chown pi:pi /opt/water-gun-system/*.sh

echo "✅ Installation completed successfully!"
echo ""
echo "🎯 WATER GUN SYSTEM SETUP COMPLETE 🎯"
echo "======================================"
echo ""
echo "📋 NEXT STEPS:"
echo "1. Reboot the system: sudo reboot"
echo "2. After reboot, start the service: sudo systemctl start water_gun_system"
echo "3. Check status: sudo systemctl status water_gun_system"
echo "4. Access web interface: http://$(hostname -I | cut -d' ' -f1)"
echo ""
echo "🔧 USEFUL COMMANDS:"
echo "• Start service: sudo systemctl start water_gun_system"
echo "• Stop service: sudo systemctl stop water_gun_system"
echo "• Check status: sudo systemctl status water_gun_system"
echo "• View logs: journalctl -u water_gun_system -f"
echo "• Manual start: /opt/water-gun-system/start.sh"
echo ""
echo "📍 FILES INSTALLED:"
echo "• System files: /opt/water-gun-system/"
echo "• Service file: /etc/systemd/system/water_gun_system.service"
echo "• Config: /opt/water-gun-system/config/"
echo "• Logs: /var/log/water-gun-system/"
echo ""
echo "⚠️  HARDWARE SETUP REQUIRED:"
echo "• Connect servos to GPIO 12 (Pan) and GPIO 13 (Tilt)"
echo "• Connect pump control to GPIO 17"
echo "• Connect IR LED to GPIO 27"
echo "• Connect PIR sensor to GPIO 22"
echo "• Ensure proper power supply for servos and pump"
echo "• Use common ground for all components"
echo ""
echo "🔒 SECURITY NOTES:"
echo "• Service runs as user 'pi' with restricted permissions"
echo "• Nginx reverse proxy handles external connections"
echo "• System logs all activities"
echo ""
echo "A reboot is recommended to ensure all changes take effect."