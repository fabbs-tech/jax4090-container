# syntax=docker/dockerfile:1.7
#
# jax4090-container: the dev / CI image for the `python_jax` tooling
# discipline. It is the runtime image (Dockerfile.runtime: Python 3.12,
# jax[cuda12], jaxtyping, beartype, numpy, scipy, curl) plus the dev
# toolchain and the non-root dev user, so code tested here runs on the
# exact JAX / jaxlib / CUDA-wheel stack the cloud jobs get.
#
# BASE_IMAGE: CI passes the runtime image it just built (by digest);
# locally, build Dockerfile.runtime first or let this pull :latest.
ARG BASE_IMAGE=ghcr.io/fabbs-tech/jax4090-runtime:latest
FROM ${BASE_IMAGE}

# Overrides the runtime's title/description; nl.fabbs.jax.* and the
# source label are inherited, so `docker inspect` names the JAX version.
LABEL org.opencontainers.image.title="jax4090-container" \
      org.opencontainers.image.description="JAX dev/CI image: jax4090-runtime plus ruff, pyright, pytest, ipykernel, git and the dev user"

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# PYTHONDONTWRITEBYTECODE / PYTHONUNBUFFERED come from the runtime image.

# git for foundation library installs (pip install git+ssh://...).
#
# Python itself is no longer installed here. The runtime image is the
# official python:3.12-slim image, whose Python lives under /usr/local
# and ships pip for 3.12. That removes three old workarounds:
# - deadsnakes PPA + software-properties-common: not needed, so neither
#   is the `apt-get purge python3-pyparsing` that undid its side effect
#   (pyparsing 2.4.7 in /usr/lib/python3/dist-packages, which pip for
#   3.12 then tried and failed to uninstall as the dev user).
# - apt's `python3-pip`: it would install pip for Debian's system
#   python3, not 3.12 (and on Ubuntu 22.04 it imported `distutils`,
#   removed in 3.12 by PEP 632). We never install it; `pip` here is
#   3.12's own.
# - get-pip.py bootstrap: the base image already has pip for 3.12.
# There is no Debian python3 at all, so /usr/lib/python3/dist-packages
# is not on 3.12's sys.path and the apt-installed-shadows-pip collision
# class is gone (the old image chowned that dir to dev to defuse it).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git openssh-client \
        libatomic1 \
    && rm -rf /var/lib/apt/lists/*

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
# Slim base images don't ship it, so pyright crashed with
# "error while loading shared libraries: libatomic.so.1" until added.

# `python` and `python3` are 3.12 already (/usr/local/bin, first on
# PATH). The alternatives additionally make /usr/bin/python and
# /usr/bin/python3 point at 3.12, so `#!/usr/bin/python3` shebangs and
# tools that hard-code /usr/bin keep working as in the old image.
RUN update-alternatives --install /usr/bin/python  python  /usr/local/bin/python3.12 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1

# Non-root user for devcontainer + CI. Default UID/GID 1000 match the
# typical Linux developer account; VS Code's updateRemoteUserUID remaps
# them at container-start time to the actual host UID, so files written
# into the bind-mounted /workspace (especially /workspace/.git) keep
# host ownership. Without this, a commit done from inside the container
# leaves .git/ subtrees root-owned on the host — see
# tooling/dev_notes/log/python_jax_phase3_bringup.md trailing note.
#
# The USER directive is deliberately NOT left set: the image still
# defaults to root so ad-hoc `docker run` keeps working, and CI's
# `docker run --user 1000:1000` / devcontainer's `remoteUser` selects
# the dev user explicitly.
#
# Writable pip locations for dev, so `pip install -r requirements.txt`
# works at runtime as the dev user:
#   /usr/local/lib/python3.12  — pip's site-packages target
#   /usr/local/bin             — entry-point scripts (ruff, pytest, ...)
#   /usr/local/share           — manpages, shell completions, locale
#                                data, etc. (matplotlib's fonttools dep
#                                writes ttx.1 under share/man/man1; a
#                                missing permission there aborts the
#                                whole pip transaction with EACCES even
#                                though the Python files would install
#                                fine)
#   /usr/local/etc             — the OTHER wheel data-file destination:
#                                `widgetsnbextension` (pulled in by
#                                `ipywidgets`) ships
#                                etc/jupyter/nbconfig/notebook.d/*.json,
#                                so a consumer whose requirements.txt
#                                names ipywidgets died with
#                                "Permission denied: '/usr/local/etc/
#                                jupyter'" at the very end of an
#                                otherwise complete install. Created
#                                first: do not rely on the base image
#                                having it.
#   /usr/local/include         — headers, the third data-file
#                                destination (`<prefix>/include/
#                                python3.12/<pkg>`). Nothing in the
#                                current fleet installs one, but it is
#                                the last of the three sibling dirs a
#                                wheel can write to, and leaving it out
#                                only buys the next EACCES.
#
# How, without the old 6 GB layer: the old image ran `chown -R` over
# these trees. Changing a file's owner in a later layer copies the whole
# file into that layer, so the chown duplicated all of site-packages
# (JAX + CUDA wheels). Here only the DIRECTORIES are handed to dev
# (`find -type d`): a directory's metadata change stores just the
# directory entry, so the layer stays tiny. That is enough for pip,
# because pip never writes into an existing file: an upgrade or
# uninstall moves the old files away (rename: needs write on the
# directory) and a file being installed is unlinked first if present
# and then created (also only directory permission). The runtime's
# files stay root-owned and read-only for dev, which is fine for every
# pip operation and keeps the shared stack bit-identical to the runtime
# image. The dev tools below are then installed AS dev, so their files
# are dev-owned from the moment they are written. (Group-writable
# site-packages set in the runtime's install RUN would also avoid the
# copy, but would put dev-user policy into the public runtime image.)
ARG DEV_USER=dev
ARG DEV_UID=1000
ARG DEV_GID=1000
RUN groupadd --gid ${DEV_GID} ${DEV_USER} \
 && useradd --uid ${DEV_UID} --gid ${DEV_GID} --create-home --shell /bin/bash ${DEV_USER} \
 && install -d /usr/local/etc /usr/local/include /usr/local/share \
 && find /usr/local/lib/python3.12 \
         /usr/local/bin \
         /usr/local/share \
         /usr/local/etc \
         /usr/local/include \
         -type d -exec chown ${DEV_USER}:${DEV_USER} {} +

# The CI/dev toolchain every python_jax project needs. Project-specific
# deps still install at devcontainer postCreateCommand / CI install step
# from each repo's requirements.txt; these are the always-present ones,
# so they live in the image and don't reinstall per run. Repos pin the
# exact ruff/pyright versions in their requirements.txt.
#
# The constraints file is the runtime's `pip freeze`: a dev tool may not
# move jax, numpy, scipy or a CUDA wheel away from what the cloud jobs
# run (the install fails instead).
#
# jaxtyping + beartype (convention, tooling/methods/python_jax/README.md
# §"Array typing": annotate with jaxtyping everywhere in compute/, wrap
# public entry points with @jaxtyped(typechecker=beartype)) come from the
# runtime image, since the cloud jobs need them too.
USER ${DEV_USER}
RUN python -m pip freeze > /tmp/runtime-constraints.txt \
 && python -m pip install --no-cache-dir -c /tmp/runtime-constraints.txt \
        ruff \
        pyright \
        pytest \
        nbstripout \
        ipykernel \
 && rm /tmp/runtime-constraints.txt
USER root

WORKDIR /workspace
