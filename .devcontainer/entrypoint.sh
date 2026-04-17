#!/bin/bash
set -e

HOME="${HOME:-/home/user}"
RENKU_MOUNT_DIR="${RENKU_MOUNT_DIR:-${HOME}/work}"
REPO_DIR="${RENKU_MOUNT_DIR}/aitchinson-flow"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-300}"

resolve_repo_dir() {
    # Prefer an explicitly provided path if it already looks like a checked-out repo.
    if [ -f "${REPO_DIR}/pyproject.toml" ]; then
        printf '%s\n' "${REPO_DIR}"
        return 0
    fi

    # Some Renku setups create an extra nesting layer (<repo>/<repo>).
    if [ -f "${REPO_DIR}/aitchinson-flow/pyproject.toml" ]; then
        printf '%s\n' "${REPO_DIR}/aitchinson-flow"
        return 0
    fi

    # Fallback: discover the first pyproject.toml within the mounted work dir.
    local discovered
    discovered="$(find "${RENKU_MOUNT_DIR}" -maxdepth 3 -type f -name pyproject.toml 2>/dev/null | head -n 1 || true)"
    if [ -n "${discovered}" ]; then
        dirname "${discovered}"
        return 0
    fi

    return 1
}

mkdir -p "${RENKU_MOUNT_DIR}/.vscode/extensions"

# ── Wait for Renku's git-clone sidecar to finish ──────────────────────────────
# Renku initialises the .git directory before checking out files, so we can't
# rely on .git existing. Wait for pyproject.toml which only appears after the
# full checkout completes.
echo "==> Waiting for Renku to finish cloning the repository..."
elapsed=0
until resolve_repo_dir >/dev/null 2>&1; do
    sleep 2
    elapsed=$((elapsed + 2))
    if [ "${elapsed}" -ge "${MAX_WAIT_SECONDS}" ]; then
        echo "ERROR: Timed out after ${MAX_WAIT_SECONDS}s waiting for repository checkout in ${RENKU_MOUNT_DIR}"
        echo "Directory snapshot for debugging:"
        ls -la "${RENKU_MOUNT_DIR}" || true
        exit 1
    fi
done
REPO_DIR="$(resolve_repo_dir)"
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
