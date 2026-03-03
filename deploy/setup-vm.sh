#!/usr/bin/env bash
# Idempotent setup script for paper-trader on GCP e2-micro (Debian 12)
set -euo pipefail

REPO_URL="https://github.com/youngkim0/invest-trading.git"
APP_DIR="/opt/paper-trader"
SWAP_SIZE="1G"

echo "=== Paper Trader VM Setup ==="

# --- 1. Swap (needed for pip install on 1GB RAM) ---
if [ ! -f /swapfile ]; then
    echo ">>> Creating ${SWAP_SIZE} swap..."
    fallocate -l ${SWAP_SIZE} /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
else
    echo ">>> Swap already exists, skipping."
fi

# --- 2. System packages ---
echo ">>> Installing system packages..."
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-dev build-essential git

# --- 3. Create paper user ---
if ! id -u paper &>/dev/null; then
    echo ">>> Creating paper user..."
    useradd --system --shell /usr/sbin/nologin --home-dir ${APP_DIR} paper
else
    echo ">>> User paper already exists, skipping."
fi

# --- 4. Clone or update repo ---
if [ ! -d "${APP_DIR}/.git" ]; then
    echo ">>> Cloning repo..."
    git clone ${REPO_URL} ${APP_DIR}
else
    echo ">>> Repo exists, pulling latest..."
    cd ${APP_DIR} && git pull
fi

# --- 5. Move .env if staged in /tmp ---
if [ -f /tmp/.env ]; then
    echo ">>> Moving .env into place..."
    mv /tmp/.env ${APP_DIR}/.env
    chmod 600 ${APP_DIR}/.env
fi

# --- 6. Create logs directory ---
mkdir -p ${APP_DIR}/logs

# --- 7. Python venv + deps ---
echo ">>> Setting up Python venv..."
if [ ! -d "${APP_DIR}/venv" ]; then
    python3 -m venv ${APP_DIR}/venv
fi
${APP_DIR}/venv/bin/pip install --upgrade pip -q
${APP_DIR}/venv/bin/pip install -r ${APP_DIR}/deploy/requirements-paper.txt -q
echo ">>> Dependencies installed."

# --- 8. Fix ownership ---
chown -R paper:paper ${APP_DIR}

# --- 9. Install and start systemd service ---
echo ">>> Installing systemd service..."
cp ${APP_DIR}/deploy/paper-trader.service /etc/systemd/system/paper-trader.service
systemctl daemon-reload
systemctl enable paper-trader
systemctl restart paper-trader

echo ""
echo "=== Setup complete ==="
echo "Check status:  sudo systemctl status paper-trader"
echo "View logs:     sudo journalctl -u paper-trader -f"
