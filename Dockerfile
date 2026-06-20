# Dockerfile for spark-auto-round on DGX Spark (GB10 / sm_121 / aarch64 / CUDA 13).
#
# Based on NVIDIA's NGC PyTorch image, which is a reliable source of a
# CUDA-enabled aarch64 torch for Blackwell GB10. It already provides nvcc,
# CUDA_HOME, the build toolchain, and a torch built against CUDA 13.
#
# The 25.11-py3 tag ships torch 2.10 + CUDA 13 for aarch64. Override at build
# time with: docker build --build-arg SAR_IMAGE=nvcr.io/nvidia/pytorch:26.xx-py3 .

ARG SAR_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3
FROM ${SAR_IMAGE}

# Already set in the NGC image; declared explicitly so the causal-conv1d CUDA
# extension build can always locate the toolkit.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}

WORKDIR /opt/spark-auto-round

# Copy the source. .dockerignore keeps .git, venvs, models and caches out.
COPY . .

# Install against the image's CUDA torch.
#   --no-build-isolation : builds causal-conv1d against the image's torch
#                          rather than a PyPI cpu wheel pulled into an isolated
#                          build env.
#   -e                   : editable install; a bind-mounted source tree (dev
#                          use) takes effect live while the image still runs
#                          standalone from this baked-in copy.
RUN pip install --no-build-isolation -e .

# Runtime defaults: work from /workspace where compose mounts the host output
# directory so quantised models survive the container exiting.
WORKDIR /workspace
ENTRYPOINT ["spark-auto-round"]
CMD ["--help"]
