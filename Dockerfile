# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Python 3.12 from deadsnakes + git for foundation library installs.
# Deliberately NOT installing apt's `python3-pip`: that installs pip
# for the system python3 (3.10 on Ubuntu 22.04), which imports
# `distutils` — removed from stdlib in Python 3.12 per PEP 632.
# We bootstrap a 3.12-specific pip below instead.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        git curl \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.12 the default `python` and `python3`
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Bootstrap pip for Python 3.12 via get-pip.py (distutils-free path).
RUN curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
 && python /tmp/get-pip.py \
 && rm /tmp/get-pip.py

# JAX with CUDA 12 + the CI/dev toolchain that every python_jax project needs.
# Project-specific deps still install at devcontainer postCreateCommand /
# CI install step from each repo's requirements.txt; these are the
# always-present ones so they live in the image and don't reinstall per run.
RUN python -m pip install --upgrade pip \
 && python -m pip install \
        "jax[cuda12]" \
        ruff \
        pyright \
        pytest \
        nbstripout \
        ipykernel

WORKDIR /workspace
