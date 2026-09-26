#!/usr/bin/env bash
# One-time EC2 setup. Run from the repo root on the instance:
#     bash deployment/setup_ec2.sh
# Idempotent. Installs system packages (Ubuntu/Debian apt or Amazon Linux dnf),
# enables clock sync (Roostoo rejects requests >60s off server time), creates
# the virtualenv, creates .env from the template if missing, and installs the
# systemd service WITHOUT starting it: fill in .env and review first.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(whoami)}"

if command -v apt-get >/dev/null; then
    sudo apt-get update -y
    sudo apt-get install -y python3 python3-venv python3-pip git chrony
    sudo systemctl enable --now chrony
elif command -v dnf >/dev/null; then
    sudo dnf install -y python3 python3-pip git chrony
    sudo systemctl enable --now chronyd
else
    echo "Unsupported distro: install python3 (>=3.10), venv, git and chrony manually." >&2
    exit 1
fi

python3 -m venv "$REPO/.venv"
"$REPO/.venv/bin/pip" install --upgrade pip
"$REPO/.venv/bin/pip" install -r "$REPO/requirements.txt"

if [ ! -f "$REPO/.env" ]; then
    cp "$REPO/.env.example" "$REPO/.env"
    echo "Created $REPO/.env from the template - fill in ROOSTOO_API_KEY / ROOSTOO_API_SECRET."
fi
chmod 600 "$REPO/.env"

sed -e "s|__USER__|$RUN_USER|g" -e "s|__REPO__|$REPO|g" "$REPO/deployment/rustoo-bot.service" \
    | sudo tee /etc/systemd/system/rustoo-bot.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable rustoo-bot

"$REPO/.venv/bin/python" -m pytest -q "$REPO/tests"

cat <<MSG

Installed. Next steps:
  1. Edit $REPO/.env (API key/secret; APP_ENV=paper for a dry run first).
  2. Smoke test:        $REPO/.venv/bin/python -m src.main --once
  3. Start the service: sudo systemctl start rustoo-bot
  4. Watch it:          journalctl -u rustoo-bot -f     (JSON log: $REPO/logs/bot.jsonl)
  5. Go live: set APP_ENV=live and LIVE_TRADING=true in .env, then
                        sudo systemctl restart rustoo-bot
  Kill switch (stop sending orders, keep logging): touch $REPO/STOP
MSG
