# ---------------------------------------------------------------------------
# vLLM – CPU-only build, two-stage image
# Adapted from the official docs (June 2025) and tested on Debian/Ubuntu hosts
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stage 1 – “builder”  #
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# ---------------------------------------------------------------------------
# 0. Environment
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    VLLM_TARGET_DEVICE=cpu \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

# ---------------------------------------------------------------------------
# 1. Core tool-chain (gcc-12, rust, cmake, etc.)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential cmake ninja-build gcc-12 g++-12 \
        libopenblas-dev libtcmalloc-minimal4 python3-dev rust-all \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. NUMA headers (needed by vLLM’s C++ CPU backend)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 3. Python bits – latest pip + CPU PyTorch wheel
# ---------------------------------------------------------------------------
RUN pip install --upgrade pip \
 && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ---------------------------------------------------------------------------
# 4. Build vLLM from source
# ---------------------------------------------------------------------------
WORKDIR /opt
RUN git clone --depth 1 https://github.com/vllm-project/vllm.git
WORKDIR /opt/vllm

# Runtime-level deps (keeps `pip check` happy)
RUN pip install -r requirements/cpu.txt

# Strip out any pinned torch version & install build-time deps
RUN python use_existing_torch.py \
 && pip install -r requirements/build.txt

# Compile + install vLLM, re-using the torch wheel already present
RUN VLLM_TARGET_DEVICE=cpu \
    pip install --no-build-isolation -e .

########################
# Stage 2 – “runtime”  #
########################
FROM python:3.12-slim AS runtime

ENV VLLM_TARGET_DEVICE=cpu \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 \
    PYTHONUNBUFFERED=1

# Only what’s needed to *run* the server
RUN apt-get update
RUN apt-get install -y --no-install-recommends libgomp1 libnuma1 libtcmalloc-minimal4 g++
RUN rm -rf /var/lib/apt/lists/*

# Pull the fully-built Python environment and vLLM sources from the builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /opt/vllm /opt/vllm

WORKDIR /workspace
EXPOSE 8000

# OpenAI-compatible REST endpoint
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]