#!/bin/bash
set -e

HOME="${HOME:-/home/user}"
RENKU_MOUNT_DIR="${RENKU_MOUNT_DIR:-${HOME}/work}"
REPO_DIR="${RENKU_MOUNT_DIR}/aitchinson-flow"

mkdir -p "${RENKU_MOUNT_DIR}/.vscode/extensions"

# ── Wait for Renku's git-clone sidecar to finish ──────────────────────────────
# Renku initialises the .git directory before checking out files, so we can't
# rely on .git existing. Wait for pyproject.toml which only appears after the
# full checkout completes.
echo "==> Waiting for Renku to finish cloning the repository..."
until [ -f "${REPO_DIR}/pyproject.toml" ]; do
    sleep 2
done
echo "==> Repository ready at ${REPO_DIR}"

# ── Install the local package into the baked-in venv ─────────────────────────
# /opt/venv already has all transitive deps (installed at image build time with
# --no-install-project). This step only installs the editable aitchinson_flow
# package itself, which takes seconds.
echo "==> Running uv sync"
cd "${REPO_DIR}"
uv sync --group dev

# ── Start VSCodium server ─────────────────────────────────────────────────────
RENKU_WORKING_DIR="${RENKU_WORKING_DIR:-${REPO_DIR}}"

RENKU_BASE_URL_PATH="${RENKU_BASE_URL_PATH:-/}"
if [[ "${RENKU_BASE_URL_PATH}" != */ ]]; then
    RENKU_BASE_URL_PATH="${RENKU_BASE_URL_PATH}/"
fi

RENKU_SESSION_IP="${RENKU_SESSION_IP:-0.0.0.0}"
RENKU_SESSION_PORT="${RENKU_SESSION_PORT:-8888}"

exec /codium-server/bin/codium-server \
    --server-base-path "${RENKU_BASE_URL_PATH}" \
    --host "${RENKU_SESSION_IP}" \
    --port "${RENKU_SESSION_PORT}" \
    --extensions-dir "${RENKU_MOUNT_DIR}/.vscode/extensions" \
    --server-data-dir "${RENKU_MOUNT_DIR}/.vscode" \
    --without-connection-token \
    --accept-server-license-terms \
    --telemetry-level off \
    --default-folder "${RENKU_WORKING_DIR}"
