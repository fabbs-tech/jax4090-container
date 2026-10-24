# jax4090-container

Container images for the `python_jax` tooling discipline. Two images,
one stack:

| Image | Dockerfile | For | Uncompressed | Pull (compressed) |
|---|---|---|---|---|
| `ghcr.io/fabbs-tech/jax4090-runtime` | `Dockerfile.runtime` | cloud GPU jobs (compute_broker), any `docker run --gpus all` | ~6.0 GB | ~3.5 GB |
| `ghcr.io/fabbs-tech/jax4090-container` | `Dockerfile` | devcontainers and CI of the python_jax repos | ~6.3 GB | ~3.6 GB |

(The previous single image, on `nvidia/cuda:12.6.3-cudnn-runtime`, was
18.8 GB uncompressed / 12.4 GB compressed; `docker images` showed
31.2 GB disk usage, which counts both. It carried CUDA twice, the
base image's and the pip wheels', and a 6 GB `chown -R` layer that
duplicated site-packages.)

**Runtime** (`jax4090-runtime`): Python 3.12 (official
`python:3.12-slim`, Debian trixie), `jax[cuda12]` (pinned; the version
is in the `nl.fabbs.jax.version` label), jaxtyping, beartype, numpy,
scipy, and what compute_broker's boot script needs inside a container:
`sh`, `curl`, `tar`, `gzip`, `timeout`, `ca-certificates`. Runs as
root. No secrets and no company code, so the package is public. The
job's own code and inputs come with each job, not in the image.

**Dev** (`jax4090-container`): `FROM` the runtime image, plus git,
openssh-client (accept-new for github/gitlab/bitbucket), libatomic1,
`/usr/bin/python{,3}` -> 3.12, ruff, pyright, pytest, nbstripout,
ipykernel, and the `dev` user (UID/GID 1000) who can `pip install`
into the system site-packages. Default user stays root.

Because the dev image is built on the runtime image (CI pins it by
digest), tests in a devcontainer run on the same Python, JAX, jaxlib
and CUDA wheels as the cloud jobs.

### Why no CUDA base image

`jax[cuda12]` installs the CUDA runtime, cuBLAS, cuDNN, cuFFT,
cuSOLVER, cuSPARSE, NCCL and ptxas as pip wheels (~4.5 GB of the
runtime image) and jaxlib loads them from site-packages. An
`nvidia/cuda` base would be a second copy. The driver (`libcuda.so`,
`nvidia-smi`) comes from the host: the NVIDIA container runtime mounts
it in at `docker run --gpus all` (the image sets
`NVIDIA_DRIVER_CAPABILITIES=compute,utility`). The host driver must
support CUDA 12.x.

### Layer sizes (uncompressed, local build 2026-10-24)

Runtime: Debian slim base 87.7 MB + Python 41.4 MB + 4.9 MB, curl
13.5 MB, pip stack 5.89 GB. Dev adds: apt (git, openssh-client,
libatomic1) 99.6 MB, dev user + directory ownership 5.6 MB, dev tools
180 MB.

## Tags

- `:latest` floats with `main` (both images).
- `:<tag>` on both images when a manual `workflow_dispatch` is run
  with an explicit tag (e.g. `jax0.12.0.1-cuda12`).

Consumer repos pin the specific tag in their
`.devcontainer/devcontainer.json` and CI workflows; compute_broker
names the runtime image per job. The floating `:latest` is fine for
local iteration; prefer pinned tags before committing to a system's
first release.

## Updating the images

1. Edit `Dockerfile.runtime` (JAX and the shared stack: bump the
   `*_VERSION` args together) or `Dockerfile` (dev tools) on a branch,
   open a PR, merge to `main`.
2. The `paths` filter on `build.yml` covers both Dockerfiles and the
   workflow, so merging rebuilds both images, runtime first, dev from
   it. Both land at `:latest`.
3. To also stamp a versioned tag: **Actions -> Build and Push JAX
   Containers -> Run workflow**, enter a tag like `jax0.12.0.1-cuda12`.

Local build:

```sh
docker build -f Dockerfile.runtime -t ghcr.io/fabbs-tech/jax4090-runtime:latest .
docker build -t ghcr.io/fabbs-tech/jax4090-container:latest .   # FROM the line above
```

### One-time step for a person: make the runtime package public

The first CI push creates the `jax4090-runtime` package in the
`fabbs-tech` org as **private**. A person with admin rights on the
package must set it to public once: GitHub -> fabbs-tech -> Packages ->
`jax4090-runtime` -> Package settings -> Change visibility -> Public.
Until then cloud providers cannot pull it without a registry credential.
(Check in the same settings page that this repository has Admin access
under "Manage Actions access", which the cleanup workflow needs; a
package created by this repo's workflow normally gets it.)

## Why ghcr.io and not Docker Hub

Single registry inside the `fabbs-tech` org, GitHub-native auth for
both push (workflow `GITHUB_TOKEN`) and pull (anonymous for public;
`gh auth` token for private). See
`tooling/dev_notes/decisions/foundation_install_source.md` for the
broader auth story across the stack.

## RunPod variant

Superseded for batch jobs: compute_broker runs `jax4090-runtime`
directly on RunPod / Vast (container: `/bin/sh -c <bootstrap>`) and
Lambda / Verda (VM: `docker run --gpus all <image> ...`). An
interactive RunPod image with `openssh-server` + root-pubkey config
(sketch in `tooling/dev_notes/plans/jax_container_runpod.md`) would
now layer on `jax4090-runtime` via a sibling `Dockerfile.runpod`.
