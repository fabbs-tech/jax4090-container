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
        libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# libatomic1 above: pyright downloads its own Node binary on first run
# (via pyright-python/nodeenv), and that binary links libatomic.so.1.
# The cuda-runtime base image doesn't ship it, so pyright crashed with
# "error while loading shared libraries: libatomic.so.1" until added.

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

# Non-root user for devcontainer + CI. Default UID/GID 1000 match the
# typical Linux developer account; VS Code's updateRemoteUserUID remaps
# them at container-start time to the actual host UID, so files written
# into the bind-mounted /workspace (especially /workspace/.git) keep
# host ownership. Without this, a commit done from inside the container
# leaves .git/ subtrees root-owned on the host — see
# tooling/dev_notes/log/python_jax_phase3_bringup.md trailing note.
#
# The USER directive is deliberately NOT set here: the image still
# defaults to root so ad-hoc `docker run` keeps working, and CI's
# `docker run --user 1000:1000` / devcontainer's `remoteUser` selects
# the dev user explicitly. /usr/local/{lib/python3.12,bin} are chowned
# so the dev user can `pip install` project deps at runtime.
ARG DEV_USER=dev
ARG DEV_UID=1000
ARG DEV_GID=1000
RUN groupadd --gid ${DEV_GID} ${DEV_USER} \
 && useradd --uid ${DEV_UID} --gid ${DEV_GID} --create-home --shell /bin/bash ${DEV_USER} \
 && chown -R ${DEV_USER}:${DEV_USER} /usr/local/lib/python3.12 /usr/local/bin

WORKDIR /workspace
