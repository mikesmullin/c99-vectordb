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
memo analyze [-f <file>] [-v] --filter <expr> [--fields <list>] [--stats <key>] [--limit <N>] [--offset <N>]
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
- `memo.yaml` (human-readable records)

## Analyze metadata

Use `analyze` for metadata-only workflows (no semantic embedding/search path):

```bash
memo analyze --filter '{source: user}'
memo analyze --filter '{source: user}' --fields id,source,rule,ts
memo analyze --filter '{source: user}' --stats rule
memo analyze --filter '{source: user}' --limit 100 --offset 0
```

Notes:

- `--filter` is required.
- `--fields` supports `id`, `metadata`, and metadata keys (for example `source` or `metadata.source`).
- `--stats <key>` prints cardinality and range summary (numeric/date-like when parseable).

Runtime now uses only `.memo + .yaml`.
