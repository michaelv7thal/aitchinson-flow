#!/bin/bash
# entrypoint.sh — RenkuLab session startup script
set -euo pipefail

HOME="${HOME:-/home/user}"
RENKU_MOUNT_DIR="${RENKU_MOUNT_DIR:-${HOME}/work}"
REPO_DIR="${RENKU_MOUNT_DIR}/aitchinson-flow"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-600}"

mkdir -p "${RENKU_MOUNT_DIR}/.vscode/extensions" || true

# ── Wait for the Renku git-clone sidecar to finish ───────────────────────────
# The sidecar clones the repo as a different UID. We cannot rely on git
# commands (ownership check blocks them) or directory listings (the dir is
# created empty first, then populated). We simply poll for pyproject.toml.
echo "==> Waiting for repository checkout in ${RENKU_MOUNT_DIR}..."
elapsed=0
while [ ! -f "${REPO_DIR}/pyproject.toml" ]; do
    sleep 2
    elapsed=$((elapsed + 2))

    if (( elapsed % 30 == 0 )); then
        echo "--- Still waiting (${elapsed}s) — contents of ${RENKU_MOUNT_DIR}: ---"
        ls -la "${RENKU_MOUNT_DIR}" 2>/dev/null || true
        echo "--- contents of ${REPO_DIR}: ---"
        ls -la "${REPO_DIR}" 2>/dev/null || echo "(unreadable or empty)"
    fi

    if [ "${elapsed}" -ge "${MAX_WAIT_SECONDS}" ]; then
        echo "ERROR: Timed out after ${MAX_WAIT_SECONDS}s waiting for ${REPO_DIR}/pyproject.toml"
        echo "Final state of ${RENKU_MOUNT_DIR}:"
        ls -la "${RENKU_MOUNT_DIR}" || true
        echo "Final state of ${REPO_DIR}:"
        ls -la "${REPO_DIR}" || true
        exit 1
    fi
done

echo "==> pyproject.toml found — repository ready at ${REPO_DIR}"

# ── Register safe.directory now that the clone is complete ───────────────────
git config --global --add safe.directory "${REPO_DIR}" 2>/dev/null || true

# ── Sync dependencies ─────────────────────────────────────────────────────────
echo "==> Running uv sync (frozen) in ${REPO_DIR}"
cd "${REPO_DIR}"
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