#!/usr/bin/env bash
# Create .venv and install dependencies. Run from code/playing-atari-with-drl/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
VENV_DIR="$ROOT/.venv"

create_venv() {
  if "$PY" -m venv "$VENV_DIR" 2>/dev/null; then
    return 0
  fi
  if command -v virtualenv >/dev/null 2>&1; then
    echo "python3-venv not available; using virtualenv"
    virtualenv "$VENV_DIR"
    return 0
  fi
  echo "Could not create a virtual environment."
  echo ""
  echo "Option A (recommended on Ubuntu/Debian):"
  echo "  sudo apt install python3-venv python3-full"
  echo "  bash scripts/setup.sh"
  echo ""
  echo "Option B (one-time, without apt):"
  echo "  pip install --break-system-packages virtualenv"
  echo "  bash scripts/setup.sh"
  echo ""
  echo "Do not use system-wide pip install -r requirements.txt (PEP 668)."
  exit 1
}

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at .venv"
  create_venv
fi

PIP="$VENV_DIR/bin/pip"
PY_VENV="$VENV_DIR/bin/python"

"$PIP" install -U pip
"$PIP" install -r requirements.txt

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "Installing PyTorch (CUDA 12.4 wheels)..."
  "$PIP" install torch --index-url https://download.pytorch.org/whl/cu124
else
  echo "Installing PyTorch (CPU)..."
  "$PIP" install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "Setup complete."
echo "  source .venv/bin/activate"
echo "  python -m cartpole.play"
