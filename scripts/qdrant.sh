# Ensure a Qdrant instance is reachable at $QDRANT_URL, starting the local
# Docker container if needed. Docker itself must already be installed and
# running -- this doesn't install or start it (auto-installing/starting
# Docker reliably across OSes and WSL setups proved too fragile), it just
# uses `docker` directly and surfaces docker's own error if that fails.
# Meant to be sourced, not executed.
#
# Usage: source scripts/qdrant.sh && ensure_qdrant_running

ensure_curl() {
    if command -v curl >/dev/null 2>&1; then
        return 0
    fi
    echo "==> curl is not installed; installing it now..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -y && sudo apt-get install -y curl
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y curl
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y curl
    elif command -v apk >/dev/null 2>&1; then
        sudo apk add --no-cache curl
    else
        echo "Could not detect a package manager to install curl. Install it yourself, then re-run this script." >&2
        return 1
    fi
    command -v curl >/dev/null 2>&1
}

# Prints a plain red warning and returns 1 if Docker isn't installed and
# running -- no suggested fix, just the fact. Colored only when stderr is a
# terminal.
ensure_docker_running() {
    if command -v docker >/dev/null 2>&1 && docker ps >/dev/null 2>&1; then
        return 0
    fi
    if [ -t 2 ]; then
        printf '\033[0;31mDocker is not running.\033[0m\n' >&2
    else
        echo "Docker is not running." >&2
    fi
    return 1
}

ensure_qdrant_running() {
    local QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
    ensure_curl || return 1

    if curl -sf "$QDRANT_URL/" >/dev/null 2>&1; then
        echo "Qdrant is already running at $QDRANT_URL."
        return 0
    fi

    case "$QDRANT_URL" in
        http://localhost:6333 | http://127.0.0.1:6333)
            ensure_docker_running || return 1

            local QDRANT_CONTAINER=cyberqa-qdrant
            if docker ps --format '{{.Names}}' | grep -qx "$QDRANT_CONTAINER"; then
                echo "Qdrant container '$QDRANT_CONTAINER' is already running."
            elif docker ps -a --format '{{.Names}}' | grep -qx "$QDRANT_CONTAINER"; then
                echo "Starting existing Qdrant container '$QDRANT_CONTAINER'..."
                docker start "$QDRANT_CONTAINER" >/dev/null
            else
                echo "Creating Qdrant container '$QDRANT_CONTAINER'..."
                docker run -d --name "$QDRANT_CONTAINER" -p 127.0.0.1:6333:6333 \
                    -v cyberqa_qdrant_storage:/qdrant/storage qdrant/qdrant >/dev/null
            fi
            ;;
        *)
            echo "==> QDRANT_URL is set to $QDRANT_URL; skipping local Qdrant container management."
            ;;
    esac

    echo "==> Waiting for Qdrant at $QDRANT_URL..."
    local ready=0
    for _ in $(seq 1 30); do
        if curl -sf "$QDRANT_URL/" >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 1
    done
    if [ "$ready" -ne 1 ]; then
        echo "Qdrant did not become reachable at $QDRANT_URL in time." >&2
        return 1
    fi
    echo "Qdrant is ready."
}
