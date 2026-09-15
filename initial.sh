#!/usr/bin/env bash
# Set up the CyberQA environment: dependencies, .env, Qdrant, and the initial
# benchmark download + ingestion.
#
# Run with: ./initial.sh
# Any arguments are forwarded to rag.load_documents (e.g. --force-download).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# shellcheck disable=SC1091
source scripts/qdrant.sh

echo "==> Checking for curl..."
ensure_curl || exit 1

just_installed_uv=0
echo "==> Checking for uv..."
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed."
    if [ -t 0 ]; then
        read -r -p "Install it now via the official installer (curl -LsSf https://astral.sh/uv/install.sh | sh)? [Y/n] " reply
        case "$reply" in
            [nN]*)
                echo "Skipping uv install. Install it yourself, then re-run this script." >&2
                exit 1
                ;;
        esac
    fi
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    [ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
    export PATH="$HOME/.local/bin:$PATH"

    if ! command -v uv >/dev/null 2>&1; then
        echo "uv installed, but isn't on PATH yet. Open a new shell and re-run this script." >&2
        exit 1
    fi
    echo "uv installed: $(uv --version)"
    just_installed_uv=1
else
    echo "uv is already installed: $(uv --version)"
fi

echo "==> Checking Docker..."
ensure_docker_running || exit 1
echo "Docker is running."

echo "==> Installing Python 3.12 and project dependencies (uv sync)..."
uv sync

if [ ! -f .env ]; then
    echo "==> Creating .env from .env.example"
    cp .env.example .env
    echo "    Edit .env and set LLM_API_KEY (and LLM_MODEL) before running the agent."
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

ensure_qdrant_running

echo "==> Downloading the benchmark and ingesting documents into Qdrant..."
uv run python -m rag.load_documents "$@"

echo
if [ "$just_installed_uv" -eq 1 ]; then
    echo "uv was just installed in this script -- that PATH change doesn't carry over to"
    echo "your shell once the script exits, so 'uv' won't be found here yet. Either:"
    echo "  - open a new terminal, or"
    echo "  - run: source \$HOME/.local/bin/env"
    echo "before using uv commands directly (this script already used it fine internally)."
    echo
fi
echo "Setup complete. Set your API key and start chatting with:"
echo "  uv run python main.py"
