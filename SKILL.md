# memo (C99 VectorDB) — Agent Skill Guide

This guide tells AI agents how to use the `memo` tool in this workspace.

## What this project is

- A C99 interactive CLI that stores/retrieves text memories using embedding similarity search.
- Binary name: `memo`.
- GPU backend: Vulkan compute shaders.
- Main commands inside CLI:
  - `memo <text>`
  - `recall <k> <text>`
  - `save <name>`
  - `load <name>`
  - `exit`

## Workspace assumptions

- Workspace root contains:
  - `Makefile`
  - `src/`
  - `shaders/`
  - `models/`
- `models/` must contain:
  - `stories110M.bin` (large; typically symlinked)
  - `tokenizer.bin` (small; copied locally)

## Build

From workspace root:

```bash
make
```

Expected outputs:

- `build/memo` (primary)
- `build/vdb` (compat symlink)
- `build/vdb_search.spv`
- `build/headless.spv`

## Install into PATH

Use a symlink so runtime path auto-detection resolves back to this workspace build directory:

```bash
mkdir -p ~/.local/bin
ln -sfn /workspace/c99-vectordb/build/memo ~/.local/bin/memo
```

## Run

You can run from any current directory after installation:

```bash
memo
```

For non-interactive execution:

```bash
printf 'memo my name is Bob\nrecall 1 bob\nexit\n' | memo
```

## Why global run works

- On startup, `memo` resolves the real executable path via `/proc/self/exe`.
- It derives workspace root from `<workspace>/build/memo`.
- It then loads runtime files relative to workspace root (`models/*`, `build/*.spv`).

## Dependency policy

If dependencies are outside this workspace:

- Copy small files into workspace.
- For large files, create symlinks under `models/`.
- Keep `models/` out of git (already in `.gitignore`).

## Quick verification flow

```bash
cd /workspace/c99-vectordb
make
printf 'memo my name is Bob\nmemo cake is for birthdays\nmemo carrots are orange\nrecall 2 party food\nexit\n' | ./build/memo
```

Expected recall top matches include:

- `cake is for birthdays`
- `carrots are orange`

## Common failure modes

- `Failed to open model file: models/stories110M.bin`
  - `models/` is missing or symlink target missing.
- `Failed to open shader file: ...`
  - Build artifacts not present; run `make`.
- Vulkan initialization errors
  - Missing Vulkan runtime/driver support on host.

## Agent operating checklist

1. Ensure `models/` exists and required files resolve.
2. Run `make` from workspace root.
3. Verify `build/memo` exists.
4. Run a short stdin script through `memo`.
5. If installing globally, ensure `~/.local/bin/memo` is a symlink to workspace `build/memo`.
