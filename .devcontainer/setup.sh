#!/usr/bin/env bash
# .devcontainer/setup.sh
# Run once after container creation. Installs uv and syncs the project.
set -euo pipefail

echo "==> Installing uv"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "==> Ensuring Python packaging prerequisites"
if ! python3 -c "import distutils" >/dev/null 2>&1; then
	echo "distutils is missing; installing python3-distutils"
	sudo apt-get update
	sudo apt-get install -y python3-distutils
fi

echo "==> Recreating project virtual environment"
if [ -d /workspace/.venv ]; then
	if ! rm -rf /workspace/.venv 2>/dev/null; then
		echo "Existing .venv is not writable; removing with sudo"
		sudo rm -rf /workspace/.venv
	fi
fi
uv venv /workspace/.venv

echo "==> Syncing project dependencies with dev tools"
cd /workspace
if [ -f pyproject.toml ]; then
	uv sync --group dev

	echo "==> Verifying GPU (informational — failure is non-fatal)"
	uv run python - <<'EOF'
import torch
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.version.cuda}")
print(f"GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'}")
EOF
else
	echo "  No pyproject.toml found — skipping uv sync (run 'uv sync --group dev' once the project is initialised)"
fi

echo "==> Setting up SSH"

# ── SSH helpers ──────────────────────────────────────────────────────────────

setup_ssh_agent() {
    # Copy SSH keys from the host-mounted read-only directory (/host-ssh-keys)
    # into the container's ~/.ssh so ssh-agent can load them.
    if [ -d "/host-ssh-keys" ]; then
        mkdir -p "$HOME/.ssh"
        chmod 700 "$HOME/.ssh"
        for key in id_ed25519 id_rsa id_ecdsa; do
            if [ -f "/host-ssh-keys/$key" ] && [ ! -f "$HOME/.ssh/$key" ]; then
                cp "/host-ssh-keys/$key" "$HOME/.ssh/$key"
                chmod 600 "$HOME/.ssh/$key"
                echo "  Copied $key from host"
            fi
        done
    fi

    # Start a persistent agent on a fixed socket path.
    rm -f "$HOME/.ssh/agent.sock"
    ssh-agent -a "$HOME/.ssh/agent.sock" > /dev/null 2>&1
    export SSH_AUTH_SOCK="$HOME/.ssh/agent.sock"

    # Load keys that have no passphrase silently; warn about the rest.
    local loaded=0
    for key in id_ed25519 id_rsa id_ecdsa; do
        if [ -f "$HOME/.ssh/$key" ]; then
            if SSH_ASKPASS='' ssh-add "$HOME/.ssh/$key" > /dev/null 2>&1; then
                echo "  Loaded $key into local agent"
                loaded=$((loaded + 1))
            else
                echo "  $key requires a passphrase — run: ssh-add ~/.ssh/$key"
            fi
        fi
    done
    [ $loaded -eq 0 ] && echo "  No keys loaded automatically. Run: ssh-add ~/.ssh/id_ed25519"
    echo "  SSH agent started at $HOME/.ssh/agent.sock"
}

setup_ssh_config() {
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"
    cat > "$HOME/.ssh/config" <<'EOF'
Host *
    PreferredAuthentications publickey
    PasswordAuthentication no
    AddKeysToAgent yes

Host github.com
    HostName github.com
    User git

Host ssh.dev.azure.com
    HostName ssh.dev.azure.com
    User git
EOF
    chmod 600 "$HOME/.ssh/config"
    echo "  SSH config written"
}

# Re-attach to (or start) the agent in every new shell session.
if ! grep -q "DEVCONTAINER_SSH_SOCK_GUARD" "$HOME/.bashrc" 2>/dev/null; then
    cat >> "$HOME/.bashrc" <<'EOF'

# DEVCONTAINER_SSH_SOCK_GUARD — reconnect to the container-local SSH agent.
if [ ! -S "$HOME/.ssh/agent.sock" ]; then
    ssh-agent -a "$HOME/.ssh/agent.sock" >/dev/null 2>&1 || true
fi
if [ -S "$HOME/.ssh/agent.sock" ]; then
    export SSH_AUTH_SOCK="$HOME/.ssh/agent.sock"
fi
EOF
fi

setup_ssh_agent
setup_ssh_config

echo "==> Setup complete."
