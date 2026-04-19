# ── Use the Renku base image to ensure compatibility with RenkuLab proxy/sidecars ──
# This version uses Ubuntu 22.04 which matches your CUDA requirement.
FROM renku/renkulab-py:3.10-0.21.0
USER root

# ── Install CUDA 12.4 & cuDNN ────────────────────────────────────────────────
# We add the NVIDIA repositories to the Renku base image
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-12-4 \
        cuda-libraries-12-4 \
        cuda-cudart-12-4 \
        libcudnn9-cuda-12 \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f cuda-keyring_1.1-1_all.deb

# Set CUDA paths
ENV PATH="/usr/local/cuda-12.4/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"

# ── Install VSCodium Server ──────────────────────────────────────────────────
# Renku looks for a server on port 8888 for the "VSCode" session type.
# FIX #1: Canonical install path is /opt/vscodium — must match entrypoint exactly.
ARG VSCODIUM_VERSION=1.96.4.25026
RUN mkdir -p /opt/vscodium && \
    curl -fsSL "https://github.com/VSCodium/vscodium/releases/download/${VSCODIUM_VERSION}/vscodium-reh-web-linux-x64-${VSCODIUM_VERSION}.tar.gz" \
    | tar -xz -C /opt/vscodium --strip-components=1

# ── Install uv ───────────────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ── Environment & Python Setup ──────────────────────────────────────────────
# Pin interpreter: PyTorch wheels target specific ABIs (cp311–cp313); uv would
# otherwise pick the newest CPython allowed by requires-python (e.g. 3.14).
ENV UV_PYTHON=3.11
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV TORCH_ATTENTION_BACKEND=math
ENV TOKENIZERS_PARALLELISM=false

# ── Pre-install dependencies ────────────────────────────────────────────────
WORKDIR /tmp/build
COPY pyproject.toml uv.lock ./

# Repo commits cu130 (local Blackwell). For this CUDA 12.4 image: patch indices,
# regenerate uv.lock in-tree (does not change the host’s committed lockfile),
# then frozen sync matches the patched pyproject.
RUN grep -q 'pytorch-cu130\|whl/cu130' pyproject.toml || \
        { echo "ERROR: Expected cu130 index marker not found in pyproject.toml — nothing to patch."; exit 1; } && \
    sed -i 's|pytorch-cu130|pytorch-cu124|g; s|whl/cu130|whl/cu124|g' pyproject.toml && \
    uv lock && \
    echo "==> Patched pyproject.toml to cu124 and wrote image-local uv.lock" && \
    grep -E 'pytorch-cu|whl/cu' pyproject.toml || true

# FIX #3: Lockfile above matches patched pyproject; --frozen skips a second solve.
# Create the venv in /opt/venv so it's persistent and separate from /home/renku.
RUN uv venv /opt/venv && \
    uv sync --frozen --group dev --no-install-project

# Clean up the temporary build directory
RUN rm -rf /tmp/build

# ── Permissions Fix ─────────────────────────────────────────────────────────
# FIX #4: Use numeric UID/GID (1000:1000) rather than the symbolic name
# "renku:renku" so this is robust even if nsswitch/passwd resolution differs
# between build and runtime environments.
RUN chown -R 1000:1000 /opt/venv /opt/vscodium

# NOTE: We do NOT attempt `uv pip install -e .` here because the project
# source is not present in the image — it is mounted at runtime by Renku into
# ${RENKU_MOUNT_DIR}. The editable install is performed by the entrypoint
# script after the repository checkout is confirmed ready.

# ── Entrypoint (COPY/chmod run as root; USER renku follows) ───────────────────
COPY --chown=1000:1000 .devcontainer/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8888
WORKDIR /home/renku
USER renku
ENTRYPOINT ["/entrypoint.sh"]