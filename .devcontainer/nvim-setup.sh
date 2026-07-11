#!/usr/bin/env bash
# .devcontainer/nvim-setup.sh
#
# "Neovim inside the container" runtime setup. Called at the end of setup.sh.
# Full rationale: ~/.config/nvim/DEVCONTAINER_GUIDE.md ("Neovim inside the
# container"). The host ~/.config/nvim is bind-mounted read-write and
# ~/.local/share/nvim is a container-local named volume (see devcontainer.json);
# Neovim itself is installed by the neovim devcontainer feature. This script
# provides the remaining runtime tools that config needs and warms the plugins.
#
# Deliberately NOT `set -e`: every step is guarded with `|| true`. A network
# hiccup warming plugins, or a missing optional tool, must never fail the whole
# container create.

echo "==> Neovim-in-container setup"

# --- (1) Own ~/.local FIRST -------------------------------------------------
# Docker creates the named volume's PARENT dirs (~/.local, ~/.local/share) as
# ROOT to host the mount, which also blocks ~/.local/state/nvim (nvim's log dir).
# chown ALL of ~/.local (not just the volume) BEFORE anything else, as its own
# statement, so a later failure below can't skip it (the #1 cause of the
# lazy.nvim "could not create leading directories ... Permission denied" error).
sudo chown -R "$(id -u):$(id -g)" "$HOME/.local" 2>/dev/null || true

# --- (2) Runtime deps for Treesitter / Mason / Telescope --------------------
# A C toolchain (Treesitter parser compilation), git/curl/unzip (Mason
# downloads), and ripgrep + fd (Telescope).
echo "  Installing runtime deps (build-essential, ripgrep, fd, unzip)..."
sudo apt-get update -qq || true
sudo apt-get install -y --no-install-recommends \
    build-essential git curl unzip ripgrep fd-find 2>/dev/null || true

# Debian/Ubuntu ship `fd` as `fdfind`; Telescope expects `fd` on PATH.
if command -v fdfind >/dev/null 2>&1 && ! command -v fd >/dev/null 2>&1; then
    mkdir -p "$HOME/.local/bin"
    ln -sf "$(command -v fdfind)" "$HOME/.local/bin/fd"
fi

# --- (3) tree-sitter CLI ----------------------------------------------------
# nvim-treesitter's `main` branch (which this config tracks) compiles parsers by
# invoking the `tree-sitter` CLI — a C compiler alone is NOT enough. Provided by
# the Node feature's npm.
#
# No version pin: the trixie base image (glibc 2.41) runs current tree-sitter-cli
# (0.26+ needs GLIBC_2.39). Only pin to tree-sitter-cli@0.25.x if you ever move
# back to a glibc<2.39 base such as *-bookworm (2.36), where 0.26+ dies with
# "GLIBC_2.39 not found".
if ! command -v tree-sitter >/dev/null 2>&1; then
    echo "  Installing tree-sitter CLI (nvim-treesitter main branch needs it)..."
    npm install -g tree-sitter-cli 2>/dev/null \
        || sudo npm install -g tree-sitter-cli 2>/dev/null \
        || echo "  ⚠️  tree-sitter CLI install hiccuped — run 'npm i -g tree-sitter-cli' manually."
fi

# --- (4) Own the nvim data volume, then warm plugins ------------------------
# Named volumes mount root-owned; hand it to the remote user before nvim writes
# plugins / Mason servers / Treesitter parsers into it.
[ -d "$HOME/.local/share/nvim" ] && \
    sudo chown -R "$(id -u):$(id -g)" "$HOME/.local/share/nvim" 2>/dev/null || true

if command -v nvim >/dev/null 2>&1; then
    echo "  Warming Neovim plugins (headless lazy.nvim sync)..."
    # GIT_SSH_COMMAND='ssh -o BatchMode=yes': if any git config rewrites github
    # HTTPS clones to SSH with no key loaded, a headless sync hangs forever on an
    # auth prompt — BatchMode makes it fail fast. </dev/null turns any first-run
    # prompt into EOF; `timeout` is a hard backstop. All of it is non-fatal:
    # Mason servers + Treesitter parsers finish on the first interactive launch,
    # and the data dir is a persistent volume so that only happens once.
    timeout 300 env GIT_SSH_COMMAND='ssh -o BatchMode=yes' \
        nvim --headless '+Lazy! sync' +qa </dev/null >/dev/null 2>&1 \
        || echo "  ⚠️  Headless plugin sync timed out/hiccuped — it'll finish on first 'nvim' launch."
    echo "==> Neovim ready. Open 'nvim' once to finish Mason + Treesitter (persisted in the volume)."
else
    echo "  ⚠️  nvim not found on PATH — is the neovim devcontainer feature enabled? Skipping warm."
fi
