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
        git curl openssh-client \
        libatomic1 \
    && apt-get purge -y --auto-remove python3-pyparsing \
    && rm -rf /var/lib/apt/lists/*

# python3-pyparsing purge: software-properties-common pulls pyparsing
# 2.4.7 into /usr/lib/python3/dist-packages/ (root-owned, shared across
# all Python versions on Ubuntu). pip for 3.12 sees it via sys.path and
# tries to uninstall it whenever a downstream `pip install` needs
# pyparsing>=3 (e.g. matplotlib -> pyparsing>=3 in a consumer repo's
# requirements.txt). That uninstall fails with EACCES because the dev
# user doesn't own /usr/lib/python3/dist-packages. We don't need
# software-properties-common after add-apt-repository runs, and nothing
# else in this image depends on pyparsing, so --auto-remove cleans it up.

# openssh-client: needed so interactive `git push` works from inside the
# container against SCP-form remotes (git@github.com:org/repo.git). The
# build-time foundation-library install path uses the
# url."https://...".insteadOf "ssh://git@github.com/" rewrite with
# FOUNDATION_TOKEN, but that PAT is read-scoped and the rewrite doesn't
# match SCP-shorthand origins anyway. Push is expected to use the host's
# SSH agent, forwarded into the container via $SSH_AUTH_SOCK in the
# consumer repo's devcontainer.json. Distinct from the RunPod variant's
# openssh-server (see README §"RunPod variant").

# Accept-new policy for the well-known forges: openssh-client alone
# isn't enough to make `git push` work non-interactively, because a
# fresh container has no entries in ~/.ssh/known_hosts and SSH defaults
# to prompting on first connect ("The authenticity of host 'github.com'
# can't be established... Are you sure you want to continue?"). That
# prompt either hangs scripted pushes outright or forces every
# new-container user to type "yes" once.
#
# `accept-new` is TOFU: SSH silently records the host key into the
# user's ~/.ssh/known_hosts on first connect and pins it from then on,
# so a MitM after the first push is still caught. We deliberately do
# NOT ship a pre-populated /etc/ssh/ssh_known_hosts — that would couple
# this image to GitHub's specific host keys and require an image
# rebuild on every key rotation. The snippet is scoped to github.com /
# gitlab.com / bitbucket.org so we don't relax host-key policy
# globally.
RUN install -d /etc/ssh/ssh_config.d \
 && printf 'Host github.com gitlab.com bitbucket.org\n    StrictHostKeyChecking accept-new\n' \
        > /etc/ssh/ssh_config.d/10-accept-new.conf

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
        jaxtyping \
        beartype \
        ruff \
        pyright \
        pytest \
        nbstripout \
        ipykernel

# jaxtyping + beartype: shape/dtype annotations and the runtime
# typechecker we pair with them. Convention (see
# tooling/methods/python_jax/README.md §"Array typing"): annotate with
# jaxtyping everywhere in compute/, wrap public entry points with
# @jaxtyped(typechecker=beartype) for runtime enforcement. Baked into
# the image because they're required by every python_jax project, same
# reasoning as jax/ruff/pyright above.

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
# the dev user explicitly.
#
# Chowned to dev so `pip install` works at runtime:
#   /usr/local/lib/python3.12  — pip's site-packages target
#   /usr/local/bin             — entry-point scripts (ruff, pytest, ...)
#   /usr/local/share           — manpages, shell completions, locale
#                                data, etc. (matplotlib's fonttools dep
#                                writes ttx.1 under share/man/man1; a
#                                missing chown there aborts the whole
#                                pip transaction with EACCES even
#                                though the Python files would install
#                                fine)
#   /usr/lib/python3/dist-packages — Debian's cross-version dist-packages
#                                dir. pip for 3.12 sees it on sys.path
#                                and tries to uninstall older copies of
#                                anything a consumer's requirements.txt
#                                upgrades. We already purge the
#                                pyparsing chain above; chowning the
#                                dir closes the entire class of
#                                apt-installed-shadows-pip collisions
#                                preemptively. If apt later reinstalls
#                                a package here it runs as root and
#                                overrides ownership, so this chown is
#                                a one-way concession to dev, not a
#                                lock-in.
ARG DEV_USER=dev
ARG DEV_UID=1000
ARG DEV_GID=1000
RUN groupadd --gid ${DEV_GID} ${DEV_USER} \
 && useradd --uid ${DEV_UID} --gid ${DEV_GID} --create-home --shell /bin/bash ${DEV_USER} \
 && chown -R ${DEV_USER}:${DEV_USER} \
        /usr/local/lib/python3.12 \
        /usr/local/bin \
        /usr/local/share \
        /usr/lib/python3/dist-packages

WORKDIR /workspace
