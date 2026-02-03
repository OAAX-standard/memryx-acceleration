# -----------------------------------------------------------------------------
# Dockerfile: ONNX -> MemryX toolchain container
# -----------------------------------------------------------------------------

FROM python:3.10.14

# Avoid interactive prompts during apt operations.
ARG DEBIAN_FRONTEND=noninteractive

# Use a predictable working directory.
WORKDIR /root

# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------
RUN set -eux; \
    echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections; \
    apt-get update -qq; \
    apt-get install -qq -y --no-install-recommends \
        gcc g++ make build-essential \
        curl wget git \
        apt-utils dos2unix \
        autoconf automake libtool \
        python3 unzip cmake \
        protobuf-compiler libprotoc-dev \
        xxd \
        ffmpeg libsm6 libxext6 \
    ; \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Python dependencies
# -----------------------------------------------------------------------------
# Copy dependency metadata first to maximize Docker layer caching.
COPY requirements.txt /root/requirements.txt
COPY setup.py /root/setup.py

# Install Python dependencies without caching wheels in the image.
RUN set -eux; \
    pip install --no-cache-dir -r /root/requirements.txt

# -----------------------------------------------------------------------------
# Application code
# -----------------------------------------------------------------------------
COPY conversion_block /root/conversion_block
COPY src/run-block.sh /usr/local/bin/conversion-run-block

# Install the local package so the console_scripts entrypoint is available.
RUN set -eux; \
    pip install --no-cache-dir .

# Install MemryX once, at build time
RUN set -eux; \
    pip install --no-cache-dir \
      --extra-index-url https://developer.memryx.com/pip \
      memryx

# Normalize script line endings and ensure executable permissions.
RUN set -eux; \
    dos2unix /usr/local/bin/conversion-run-block; \
    chmod +x /usr/local/bin/conversion-run-block

ENV PATH="/usr/local/bin:${PATH}"

ENTRYPOINT ["bash", "/usr/local/bin/conversion-run-block"]
