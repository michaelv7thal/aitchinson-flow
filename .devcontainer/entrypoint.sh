#!/bin/bash
# entrypoint.sh — RenkuLab session startup script
set -euo pipefail

HOME="${HOME:-/home/user}"
RENKU_MOUNT_DIR="${RENKU_MOUNT_DIR:-${HOME}/work}"
REPO_DIR="${RENKU_MOUNT_DIR}/aitchinson-flow"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-300}"

mkdir -p "${RENKU_MOUNT_DIR}/.vscode/extensions" || true

# ── Trust the repo dir so git works regardless of UID mismatch ───────────────
git config --global --add safe.directory "${REPO_DIR}" 2>/dev/null || true

# ── Detect and recover a stale/incomplete checkout ───────────────────────────
# Symptom: .git exists (possibly from a previous session's persistent volume)
# but the working tree is empty. The Renku sidecar may have skipped the clone
# because the directory already existed.
recover_stale_checkout() {
    [ -d "${REPO_DIR}/.git" ] || return 0

    # Count files in the working tree (excluding .git)
    local file_count
    file_count="$(find "${REPO_DIR}" -maxdepth 1 ! -path "${REPO_DIR}" ! -name '.git' | wc -l)"
    [ "${file_count}" -gt 0 ] && return 0   # working tree has files, nothing to do

    echo "==> Stale/empty working tree detected — recovering checkout..."

    # Determine the remote default branch from the remote (doesn't need a
    # local checkout to work, just network access).
    local default_branch
    default_branch="$(git -C "${REPO_DIR}" ls-remote --symref origin HEAD 2>/dev/null \
        | awk '/^ref:/{sub("refs/heads/","", $2); print $2; exit}')"
    default_branch="${default_branch:-main}"
    echo "==> Remote default branch: '${default_branch}'"

    git -C "${REPO_DIR}" fetch --depth=1 origin "${default_branch}"
    git -C "${REPO_DIR}" checkout -B "${default_branch}" "origin/${default_branch}"
    echo "==> Recovery complete — $(git -C "${REPO_DIR}" rev-list --count HEAD) commits checked out."
}

recover_stale_checkout

# ── Wait for pyproject.toml (covers the normal fresh-clone path) ─────────────
echo "==> Waiting for repository checkout in ${RENKU_MOUNT_DIR}..."
elapsed=0
while [ ! -f "${REPO_DIR}/pyproject.toml" ]; do
    sleep 2
    elapsed=$((elapsed + 2))

    # Retry recovery every 30s in case the sidecar is racing us
    if (( elapsed % 30 == 0 )); then
        echo "--- Still waiting (${elapsed}s) ---"
        ls -la "${REPO_DIR}" 2>/dev/null || true
        recover_stale_checkout
    fi

    if [ "${elapsed}" -ge "${MAX_WAIT_SECONDS}" ]; then
        echo "ERROR: Timed out after ${MAX_WAIT_SECONDS}s waiting for ${REPO_DIR}/pyproject.toml"
        ls -la "${RENKU_MOUNT_DIR}" || true
        ls -la "${REPO_DIR}" || true
        exit 1
    fi
done

echo "==> Repository ready at ${REPO_DIR}"

# ── Sync dependencies ─────────────────────────────────────────────────────────
echo "==> Running uv sync (frozen) in ${REPO_DIR}"
cd "${REPO_DIR}"
# Point uv cache to a directory we own — the base image's /home/jovyan/.cache
# is owned by a different UID and causes a permission error at startup.
export UV_CACHE_DIR="${HOME}/.cache/uv"
mkdir -p "${UV_CACHE_DIR}"
uv sync --frozen --group dev

# ── Launch VSCodium ───────────────────────────────────────────────────────────
RENKU_WORKING_DIR="${RENKU_WORKING_DIR:-${REPO_DIR}}"
RENKU_BASE_URL_PATH="${RENKU_BASE_URL_PATH:-/}"
[[ "${RENKU_BASE_URL_PATH}" != */ ]] && RENKU_BASE_URL_PATH="${RENKU_BASE_URL_PATH}/"
RENKU_SESSION_IP="${RENKU_SESSION_IP:-0.0.0.0}"
RENKU_SESSION_PORT="${RENKU_SESSION_PORT:-8888}"

echo "==> Starting VSCodium on ${RENKU_SESSION_IP}:${RENKU_SESSION_PORT}"
exec /opt/vscodium/bin/codium-server \
    --server-base-path    "${RENKU_BASE_URL_PATH}" \
    --host                "${RENKU_SESSION_IP}" \
    --port                "${RENKU_SESSION_PORT}" \
    --extensions-dir      "${RENKU_MOUNT_DIR}/.vscode/extensions" \
    --server-data-dir     "${RENKU_MOUNT_DIR}/.vscode" \
    --without-connection-token \
    --telemetry-level     off \
    --default-folder      "${RENKU_WORKING_DIR}"