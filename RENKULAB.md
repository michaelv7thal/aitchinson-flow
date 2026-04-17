# RenkuLab Quick Start

Use these settings when creating a new RenkuLab session for this project.

## Session Settings

- Container image: `mvonsiebenth/aitchinson-flow:latest`
- Default URL path: `/`
- Port: `8888`
- Mount directory: `/home/user/work`
- Working directory: `/home/user/work/aitchinson-flow`
- UID: `1000`
- GID: `1000`
- Command: *(leave empty)*
- Args: *(leave empty)*
- Strip session URL path prefix: `No`

## Why These Values

- The container entrypoint starts VSCodium server on port `8888`.
- `Mount directory` must be writable by `user` (uid `1000`).
- `Working directory` points VSCodium to this repository folder at startup.
- Renku's git service clones the repository automatically using your GitHub OAuth
  token. The entrypoint waits for the checkout to complete, then runs `uv sync`
  to install the editable `aitchinson_flow` package into the baked-in venv.

## Common Error

If you see:

- `PermissionError: [Errno 13] Permission denied: '/home/user/aitchinson-flow'`

Then set:

- Mount directory to `/home/user/work`
- Working directory to `/home/user/work/aitchinson-flow`

This avoids writing directly under `/home/user` where repo initialization may fail.
