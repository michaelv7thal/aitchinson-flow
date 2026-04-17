# ── Base: CUDA 12.4 + cuDNN 9 ────────────────────────────────────────────────
# Matches the pytorch-cu124 wheel index used in pyproject.toml / uv.lock.
# Build and run directly on the Linux cluster:
#   docker build -t <image> .
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    WANDB_CONSOLE=off \
    # HuggingFace downloads (GPT-2, wikitext-2-raw-v1) land here so they
    # persist in the data volume and are not re-fetched on every run.
    HF_HOME=/workspace/data/hf_cache \
    TOKENIZERS_PARALLELISM=false

# ── System: Python 3.11 + pip + Quarto ───────────────────────────────────────
# wget fetches the Quarto .deb; gdebi-core installs it with dependency handling.
ARG QUARTO_VERSION=1.6.42
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        git \
        wget \
        gdebi-core \
    && wget -q "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-amd64.deb" \
    && gdebi --non-interactive "quarto-${QUARTO_VERSION}-linux-amd64.deb" \
    && rm "quarto-${QUARTO_VERSION}-linux-amd64.deb" \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# ── uv: fast, lock-file-based package manager ────────────────────────────────
RUN pip install --no-cache-dir uv

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only the lock files first so this heavy layer (~2 GB torch CUDA wheel)
# is cached by Docker and only rebuilt when deps actually change,
# not on every source-code edit.
WORKDIR /workspace
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Add the project venv to PATH so plain `python` / script shebangs work.
ENV PATH="/workspace/.venv/bin:$PATH"

# ── Source code ───────────────────────────────────────────────────────────────
COPY . .

# ── Runtime volumes ───────────────────────────────────────────────────────────
# Persist data (text8, wiki cache, HF model weights) and results across restarts.
# Mount these before running any script:
#
#   docker run --gpus all \
#     -v $(pwd)/data:/workspace/data \
#     -v $(pwd)/results:/workspace/results \
#     -e WANDB_API_KEY=$WANDB_API_KEY \
#     -e WANDB_PROJECT=aitchison-flow \
#     <image> bash scripts/run_dna.sh
#
# Available entry-point scripts:
#   scripts/run_dna.sh     — DNA promoter benchmark (4 models)
#   scripts/run_text8.sh   — text8 character-level benchmark (4 models)
#   scripts/run_wiki.sh    — WikiText-2 hallucination auditor (GPT-2 logits)
VOLUME ["/workspace/data", "/workspace/results"]

