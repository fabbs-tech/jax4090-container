# jax4090-container

Container image for the `python_jax` tooling discipline. Provides
JAX + CUDA 12 + the standard Python CI toolchain (ruff, pyright,
pytest, nbstripout, ipykernel) on Ubuntu 22.04 with Python 3.12.

Built and pushed automatically by `.github/workflows/build.yml` on
every change to `Dockerfile`. Available at:

- `ghcr.io/fabbs-tech/jax4090-container:latest` — floats with `main`
- `ghcr.io/fabbs-tech/jax4090-container:<tag>` — when a manual
  `workflow_dispatch` is run with an explicit tag (e.g.
  `jax0.4.38-cuda12.6`)

Consumer repos pin the specific tag in their
`.devcontainer/devcontainer.json` and CI workflows. The floating
`:latest` is fine for local iteration; prefer pinned tags before
committing to a system's first release.

## Updating the image

1. Edit `Dockerfile` on a branch, open a PR, merge to `main`.
2. The `paths` filter on `build.yml` includes `Dockerfile` and the
   workflow file itself, so merging triggers a rebuild. Image
   lands at `:latest`.
3. To also stamp a versioned tag: **Actions → Build and Push JAX
   Container → Run workflow**, enter a tag like
   `jax0.4.38-cuda12.6`.

## Why ghcr.io and not Docker Hub

Single registry inside the `fabbs-tech` org, GitHub-native auth for
both push (workflow `GITHUB_TOKEN`) and pull (anonymous for public;
`gh auth` token for private). See
`tooling/dev_notes/decisions/foundation_install_source.md` for the
broader auth story across the stack.

## RunPod variant

Deferred. Sketch + trigger signals live in
`tooling/dev_notes/plans/jax_container_runpod.md`. The RunPod image
would layer `openssh-server` + root-pubkey config on top of this
one via a sibling `Dockerfile.runpod`; current deployment target
is local workstations (RTX 4090) only.
