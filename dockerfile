# SEESR with SD Turbo Dockerfile
# Optimized for super-resolution with minimal dependencies

FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-pip \
    python3.9-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Set working directory
WORKDIR /src

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional dependencies for SEESR
RUN pip3 install --no-cache-dir \
    pywt \
    opencv-python-headless \
    timm \
    diffusers[torch] \
    accelerate

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    mkdir -p /root/.cache/huggingface/transformers && \
    mkdir -p preset/models/seesr && \
    mkdir -p preset/models/sd-turbo && \
    mkdir -p preset/models/ram

# Set Python path
ENV PYTHONPATH="/src:${PYTHONPATH}"

# Optimize for production
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
ENV FORCE_CUDA="1"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

# The actual command will be managed by Cog
CMD ["python3", "predict.py"]