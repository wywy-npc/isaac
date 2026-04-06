#!/usr/bin/env bash
set -euo pipefail

# ISAAC Install / Update Script
# Run: curl -s https://raw.githubusercontent.com/.../install.sh | bash
# Or:  cd ~/ISAAC && bash install.sh

ISAAC_DIR="${ISAAC_DIR:-$HOME/ISAAC}"
VENV_DIR="$ISAAC_DIR/.venv"
PYTHON=""

echo "╔══════════════════════════════════════╗"
echo "║  ISAAC — Install / Update            ║"
echo "╚══════════════════════════════════════╝"
echo ""

# --- Find Python 3.12+ ---
for candidate in python3.13 python3.12 python3; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 12 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

# Check common non-PATH locations
if [ -z "$PYTHON" ]; then
    for path in /usr/local/bin/python3.13 /usr/local/bin/python3.12 /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12; do
        if [ -x "$path" ]; then
            PYTHON="$path"
            break
        fi
    done
fi

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.12+ required but not found."
    echo ""
    echo "Install options:"
    echo "  macOS:  brew install python@3.13"
    echo "  Ubuntu: sudo apt install python3.13 python3.13-venv"
    echo "  Any:    curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.13"
    exit 1
fi

echo "Using Python: $PYTHON ($($PYTHON --version))"

# --- Create or update venv ---
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "Virtual environment exists, updating..."
fi

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip --quiet

# --- Install ISAAC ---
echo "Installing ISAAC..."
cd "$ISAAC_DIR"
pip install -e ".[dev]" --quiet

# --- Verify ---
echo ""
echo "Verifying installation..."
isaac --help > /dev/null 2>&1 && echo "  ✓ isaac CLI works" || echo "  ✗ isaac CLI failed"
python -c "from isaac.core.orchestrator import Orchestrator; print('  ✓ core imports work')" 2>/dev/null || echo "  ✗ core imports failed"
python -c "from isaac.sandbox.fly import FlySandbox; print('  ✓ sandbox imports work')" 2>/dev/null || echo "  ✗ sandbox imports failed"

# --- Run tests ---
echo ""
echo "Running tests..."
python -m pytest tests/test_core.py -q 2>&1 | tail -3

# --- Initialize if first run ---
if [ ! -f "$HOME/.isaac/agents.yaml" ]; then
    echo ""
    echo "First install — initializing ~/.isaac..."
    isaac init
fi

# --- Shell integration ---
SHELL_RC=""
if [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ -f "$HOME/.bashrc" ]; then
    SHELL_RC="$HOME/.bashrc"
fi

if [ -n "$SHELL_RC" ]; then
    # Add venv activation to shell if not already there
    if ! grep -q "ISAAC" "$SHELL_RC" 2>/dev/null; then
        echo "" >> "$SHELL_RC"
        echo "# ISAAC" >> "$SHELL_RC"
        echo "export PATH=\"$VENV_DIR/bin:\$PATH\"" >> "$SHELL_RC"
        echo "  ✓ Added isaac to PATH in $SHELL_RC"
    else
        echo "  ✓ isaac already in PATH"
    fi
fi

echo ""
echo "══════════════════════════════════════"
echo "  ISAAC installed successfully"
echo ""
echo "  Quick start:"
echo "    isaac chat              # start default agent"
echo "    isaac start             # all agents in tmux"
echo "    isaac heartbeat --once  # run one heartbeat"
echo ""
echo "  Required env var:"
echo "    export ANTHROPIC_API_KEY=sk-ant-..."
echo ""
echo "  Optional:"
echo "    export FLY_API_TOKEN=...         # VM sandbox"
echo "    export FLY_APP=isaac-sandbox     # VM sandbox"
echo "    export BRAVE_API_KEY=...         # web search"
echo "    export TELEGRAM_BOT_TOKEN=...    # telegram gateway"
echo "══════════════════════════════════════"
