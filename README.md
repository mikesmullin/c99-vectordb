# memo (mac)

| version (linked) | description |
|---|---|
| [v2 (current)](https://github.com/mikesmullin/memo/tree/v2) | Current mac-focused FAISS reimplementation, optimized for macOS workflow and YAML-file-based save input. |
| [v1](https://github.com/mikesmullin/memo/tree/v1) | Previous C99/Vulkan implementation with in-process embedding pipeline and the original save/recall internals. |

FAISS-based reimplementation of `memo` for macOS.

## Goals

- Match existing `memo` CLI commands (`save`, `recall`, `clean`) and output style.
- Use FAISS for vector indexing/search.
- Replace old save input method with YAML file-only input.

## Setup

```bash
cd mac
uv sync
```

Or run directly with wrapper:

```bash
./memo --help
```

## CLI

```bash
memo [--help] [-v] [-f <file>]
memo save [-f <file>] [-v] <yaml_file>
memo recall [-f <file>] [-v] [-k <N>] [--filter <expr>] <query>
memo clean [-f <file>] [-v]
```

## Save YAML format

Single entry:

```yaml
metadata:
  source: chat
  tags: [ops, triage]
body: |
  A multi-line note body.
```

Batch entries in one file:

```yaml
---
metadata:
  source: chat
body: First note

---
metadata:
  source: user
body: Second note
```

Optional overwrite by id (feature parity with old `<id> <note>` path):

```yaml
---
id: 3
metadata:
  source: user
body: Updated note for id 3
```

## Storage files

Given default db basename `memo` in current working directory:

- `memo.memo` (FAISS index)
- `memo.txt` (binary text sidecar)
- `memo.meta` (binary metadata sidecar)
