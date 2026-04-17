# ─────────────────────────────────────────────────────────────────────────────
# aitchison-flow — GPU training image (Renku VSCode session)
# Target hardware : NVIDIA A100 (Compute Capability 8.0)
# CUDA version    : 12.4.1 + cuDNN
# Python          : 3.11
# Frontend        : VSCodium server (port 8888)
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Suppress interactive apt prompts
ENV DEBIAN_FRONTEND=noninteractive

ARG VSCODIUM_VERSION=1.96.4.25026
ARG SESSION_USER=user

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3-pip \
        curl \
        git \
        git-lfs \
        ca-certificates \
        build-essential \
        tini \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# ── VSCodium server ───────────────────────────────────────────────────────────
RUN mkdir -p /codium-server \
    && curl -fsSL "https://github.com/VSCodium/vscodium/releases/download/${VSCODIUM_VERSION}/vscodium-reh-web-linux-x64-${VSCODIUM_VERSION}.tar.gz" \
       | tar -xz -C /codium-server

# ── uv ────────────────────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

# ── Environment variables ─────────────────────────────────────────────────────
# Required for second-order autograd through transformer attention
ENV TORCH_ATTENTION_BACKEND=math
# Suppress HuggingFace tokeniser parallelism warnings
ENV TOKENIZERS_PARALLELISM=false
# venv lives inside the image, not the mounted workspace
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
# Put venv binaries on PATH so python/pip resolve without activation
ENV PATH="/opt/venv/bin:$PATH"

# ── Python dependencies (baked in, source code excluded) ─────────────────────
WORKDIR /build
COPY pyproject.toml uv.lock ./

# Remap the PyTorch index from cu130 (Blackwell/local) to cu124 (A100/cloud).
# pyproject.toml is left unchanged so the local Blackwell setup is unaffected.
RUN sed -i 's|pytorch-cu130|pytorch-cu124|g; s|whl/cu130|whl/cu124|g' pyproject.toml

# Create venv and install all deps; --no-install-project skips the local
# package (source isn't here yet — added at runtime via git pull).
RUN uv venv /opt/venv \
    && uv sync --group dev --no-install-project

# ── Verify GPU visibility at build time (informational) ───────────────────────
# This will print "not available" during a CPU-only build — that's fine.
RUN python - <<'EOF'
import torch
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"GPU     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available (build-time)'}")
EOF

# ── Non-root user for Renku session runtime ───────────────────────────────────
RUN useradd -m -u 1000 -s /bin/bash ${SESSION_USER} \
    && chown -R ${SESSION_USER}:${SESSION_USER} /opt/venv \
    && chown -R ${SESSION_USER}:${SESSION_USER} /codium-server

COPY .devcontainer/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER ${SESSION_USER}
ENV HOME=/home/${SESSION_USER}
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

WORKDIR /home/${SESSION_USER}
EXPOSE 8888

ENTRYPOINT ["tini", "--", "/bin/bash", "/entrypoint.sh"]
