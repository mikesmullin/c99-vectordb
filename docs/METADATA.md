# Metadata Filtering

Memo supports deterministic (non-semantic) metadata filtering on its vector
database. Each record can carry YAML Flow metadata that is stored in a
**separate sidecar file**, indexed for exact-match / prefix / range lookups,
and used to pre-filter candidates **before** the GPU KNN search.

---

## 1. Concepts

### What metadata is

```yaml
# YAML Flow syntax — one line per record, stored alongside the vector
source: chat, ts: 2026-02-21, tags: [personal, food], speaker: user
```

Outer `{ }` braces are optional (both for `-m` metadata and `--filter` filters).

- Attached to a record at `save` time via a `-m` flag.
- Persisted in its own sidecar (not inside the `.memo` binary blob).
- Searched deterministically — no embedding, no cosine similarity.
- Acts as a **pre-filter**: narrow candidates first, then run KNN on the
  survivors.

### File layout (per DB basename)

With the default basename `memo`, files are created in the current directory:

```
memo.memo        # binary vectors + ids
memo.txt         # line texts
memo.meta        # metadata sidecar
```

All three files share the same ID space (index 0..N-1).

---

## 2. CLI Usage

### `memo save`

```
memo save [-f <file>] [-m <yaml>] [<id>] <note>
```

The `-m` flag attaches YAML Flow metadata to the saved record. If omitted, the
record is stored with no metadata (it will not match any filter).

```bash
memo save -m "source: chat, tags: [personal]" I love sushi
memo save -m "ts: 2026-02-21" My meeting is at 3pm
memo save My cat is named Luna          # no metadata
```

### `memo recall`

```
memo recall [-f <file>] [-k <N>] [--filter <expr>] <query>
```

The `--filter` flag narrows results by metadata before running the semantic
search. Only records whose metadata passes the filter are sent to the GPU for
KNN scoring.

```bash
memo recall --filter 'source: chat' what food do I like
memo recall --filter 'tags: {$contains: personal}' tell me about myself
memo recall -k 5 --filter 'ts: {$gte: 2026-02-01}' recent events
```

### `memo clean`

Removes all three sidecar files (`.memo`, `.txt`, `.meta`).

---

## 3. Filter Expression Syntax

Filters use **YAML Flow** syntax with `$operator` keys — the same format as
the `-m` metadata input, so you only need to learn one syntax. The outer `{ }`
braces are optional; all examples below omit them, but
`--filter '{source: user}'` works identically to `--filter 'source: user'`.

### Operators

| Operator       | Meaning            | Example                      |
|----------------|--------------------|------------------------------|
| *(bare value)* | exact match        | `source: user`               |
| `$ne`          | not equal          | `source: {$ne: system}`      |
| `$gte`         | greater-or-equal   | `priority: {$gte: 2}`        |
| `$lte`         | less-or-equal      | `ts: {$lte: 2026-01-31}`     |
| `$prefix`      | prefix match       | `category: {$prefix: per}`   |
| `$contains`    | array contains     | `tags: {$contains: food}`    |

### AND / OR logic

Multiple top-level keys are implicitly ANDed:

```bash
memo recall --filter 'source: user, priority: {$gte: 2}' urgent user items
```

For multiple conditions on the **same key**, use explicit `$and`:

```bash
memo recall --filter '$and: [{ts: {$gte: 2026-01-01}}, {ts: {$lte: 2026-01-31}}]' january memories
```

Use `$or` when **any** condition matching is enough:

```bash
memo recall --filter '$or: [{source: user}, {source: chat}]' user or chat memories
```

### Examples — individual operators

```bash
# Exact match (bare value, string)
memo recall --filter 'source: chat' what did we talk about
#   → only records where source is exactly "chat"

# Exact match (integer)
memo recall --filter 'priority: 1' urgent items
#   → only records where priority is exactly 1

# Not equal
memo recall --filter 'source: {$ne: system}' what do I know
#   → all records except those from source "system"

# Greater-or-equal (numeric)
memo recall --filter 'priority: {$gte: 3}' important things
#   → records where priority is 3, 4, 5, ...

# Greater-or-equal (lexicographic / date string)
memo recall --filter 'ts: {$gte: 2026-02-01}' recent events
#   → records with ts "2026-02-01" or later (string comparison)

# Less-or-equal
memo recall --filter 'ts: {$lte: 2026-01-31}' older memories
#   → records with ts on or before "2026-01-31"

# Prefix match
memo recall --filter 'category: {$prefix: per}' personal stuff
#   → matches "personal", "performance", "permissions", etc.

# Array contains
memo recall --filter 'tags: {$contains: food}' what do I eat
#   → records whose tags array contains the element "food"
```

### Examples — combined filters (AND)

```bash
# Two exact matches
memo recall --filter 'source: user, category: pref' my preferences
#   → source is "user" AND category is "pref"

# Exact + range
memo recall --filter 'source: chat, ts: {$gte: 2026-02-01}' recent chats
#   → source is "chat" AND ts is on or after Feb 1

# Exact + array-contains
memo recall --filter 'source: user, tags: {$contains: health}' health facts about me
#   → source is "user" AND tags contains "health"

# Date window (same key — needs $and)
memo recall --filter '$and: [{ts: {$gte: 2026-01-01}}, {ts: {$lte: 2026-01-31}}]' january memories
#   → ts between Jan 1 and Jan 31 inclusive

# Prefix + not-equal
memo recall --filter 'category: {$prefix: per}, source: {$ne: system}' personal non-system
#   → category starts with "per" AND source is not "system"

# Three-way combo
memo recall --filter 'source: user, category: health, priority: {$gte: 2}' important health
#   → source is "user" AND category is "health" AND priority >= 2
```

---

## 4. How Filtering Works

### Search flow

```
 ┌─────────┐     ┌──────────────┐     ┌────────────┐
 │ Filter  │───▶│ Candidate    │───▶│ GPU KNN    │
 │ metadata│     │ ID mask      │     │ (filtered) │
 └─────────┘     └──────────────┘     └────────────┘
```

When `--filter` is provided:

1. **MetaStore_Filter** iterates all metadata records and evaluates the filter
   expression, producing a bitmask of passing IDs.
2. **VDB_Search** receives the bitmask and extracts only the surviving vectors
   into a contiguous buffer (pre-filter strategy).
3. The filtered vectors are uploaded to the GPU for KNN scoring.
4. Result indices are mapped back to the original record IDs.

When `--filter` is omitted, `VDB_Search` receives a NULL mask and searches all
vectors — identical to the behavior before metadata was added.

Filter evaluation is O(N) where N is the record count (not the vector
dimension), so it remains fast for databases under 100k records.

### Records without metadata

Records saved without `-m` have no metadata. They **never match** any filter
expression. This means `--filter` implicitly excludes any records that were
saved without metadata. Unfiltered recall still returns all records.

---

## 5. Metadata Sidecar Format (`*.meta`)

A binary format mirroring how the text store (`.txt`) works:

```
Header:
  count       : i32          — number of records

Per record (count times):
  len         : i32          — byte length of YAML string (0 = no metadata)
  yaml_bytes  : u8[len]      — raw YAML Flow string, UTF-8, no newline
```

### YAML Flow subset

Only **YAML Flow** (single-line JSON-superset) is supported. This is parsed by
a minimal hand-rolled scanner with no library dependencies:

- `key: value, key2: value2` — flat key-value pairs
- Values are bare strings, integers, or `[tag1, tag2]` arrays
- No multi-line, no anchors, no nested objects, no complex types

---

## 6. Example End-to-End

```bash
# Store facts with metadata
memo save -m "source: user, category: pref" My favorite color is blue
memo save -m "source: user, category: health, tags: [medical, allergy]" I am allergic to peanuts
memo save -m "source: chat, category: pref, priority: 3" User prefers dark mode
memo save My cat is named Luna
memo save -m "source: system, category: pref, priority: 1" Default font size is 14

# Semantic recall, no filter — searches all
memo recall -k 5 what are my preferences
#   [1] Score: 0.45 | User prefers dark mode
#   [2] Score: 0.37 | I am allergic to peanuts
#   [3] Score: 0.31 | Default font size is 14
#   [4] Score: 0.23 | My cat is named Luna
#   [5] Score: 0.21 | My favorite color is blue

# Filter by source
memo recall -k 5 --filter 'source: user' what do I know
#   [1] Score: 0.25 | I am allergic to peanuts
#   [2] Score: 0.16 | My favorite color is blue

# Filter by category with operator
memo recall -k 5 --filter 'priority: {$gte: 2}' important things
#   [1] Score: 0.44 | User prefers dark mode

# Filter by array contents
memo recall -k 5 --filter 'tags: {$contains: allergy}' health info
#   [1] Score: 0.23 | I am allergic to peanuts

# Combined AND filter
memo recall -k 5 --filter 'source: user, category: health' health facts
#   [1] Score: 0.29 | I am allergic to peanuts
```
