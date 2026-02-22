# memo — Usage Guide for AI Agents

`memo` is a lightweight vector database for agents to store and recall important facts as long-term memory.

## Command help (actual output)

```text
Usage:
  memo [--help] [-v] [-f <file>]
  memo save [-f <file>] [-v] [-m <yaml>] [<id>] <note|->
  memo recall [-f <file>] [-v] [-k <N>] [--filter <expr>] <query>
  memo clean [-f <file>] [-v]

Options:
  [-f <file>]        Optional DB basename (default: memo)
  -v                 Verbose logs to stderr
  -m <yaml>          Attach YAML Flow metadata to a saved record
  --filter <expr>    Filter recall results by metadata
  --help             Show this help
```

## Core behavior

- `memo` or `memo --help` prints help.
- `memo save <note>` stores a new memory.
- `memo save -` reads the note from stdin (useful for multi-line notes/documents).
- `memo save -m '<yaml>' <note>` stores a memory with YAML Flow metadata (e.g. `source: user, tags: [health]`).
- `memo save <id> <note>` overwrites memory at an existing ID.
- `memo recall <query>` recalls top matches (default `k=2`).
- `memo recall -k <N> <query>` recalls top `N` matches (`N` capped at 100).
- `memo recall --filter '<expr>' <query>` pre-filters by metadata before KNN search. Uses YAML Flow `$operator` syntax (e.g. `source: user`, `priority: {$gte: 2}`, `tags: {$contains: food}`).
- `memo clean` wipes the current memory DB files (`.memo`, `.txt`, `.meta`).
- `-f <file>` is optional and changes DB basename; relative values are resolved from the process CWD (where `memo` is run), and absolute paths are used as-is.
- `-v` enables debug/initialization logs on stderr only.

## Real examples (actual commands + output)

### Save and recall

```bash
$ memo save my name is Bob
Memorized: 'my name is Bob' (ID: 0)

$ memo save cake is for birthdays
Memorized: 'cake is for birthdays' (ID: 1)

$ memo save carrots are orange
Memorized: 'carrots are orange' (ID: 2)

$ memo recall -k 2 party food
Top 2 results for 'party food':
  [1] Score: 0.3026 | cake is for birthdays
  [2] Score: 0.2987 | carrots are orange
```

### Overwrite an existing memory by ID

```bash
$ memo save 1 cake is for celebrations
Memorized: 'cake is for celebrations' (ID: 1)

$ memo recall -k 2 party food
Top 2 results for 'party food':
  [1] Score: 0.3266 | cake is for celebrations
  [2] Score: 0.2987 | carrots are orange

$ memo clean
Cleared memory database (memo.memo, memo.txt, memo.meta)
```

### Save with metadata and filtered recall

```bash
$ memo save -m "source: user, category: pref" My favorite color is blue
Memorized: 'My favorite color is blue' (ID: 0)

$ memo save -m "source: user, category: health, tags: [medical, allergy]" I am allergic to peanuts
Memorized: 'I am allergic to peanuts' (ID: 1)

$ memo save -m "source: chat, category: pref" User prefers dark mode
Memorized: 'User prefers dark mode' (ID: 2)

$ memo recall -k 3 --filter 'source: user' what do I know about myself
Top 3 results for 'what do I know about myself':
  [1] Score: 0.25 | I am allergic to peanuts
  [2] Score: 0.16 | My favorite color is blue

$ memo recall -k 3 --filter 'tags: {$contains: allergy}' health info
Top 3 results for 'health info':
  [1] Score: 0.23 | I am allergic to peanuts
```

*IMPORTANT:* Learn more about metadata by reading `docs/METADATA.md`.

## Output contract

- Normal mode: only final subcommand result goes to stdout.
- `memo recall` prints each result as a block; multi-line notes are indented under the matching result header.
- Verbose mode (`-v`): extra debug/startup info goes to stderr.

## Fast usage patterns

- Default DB (recommended):
  - `memo save remember this`
  - `memo recall remember`
- Optional custom DB path with `-f`:
  - `memo -f /tmp/memo_test save hello world`
  - `memo -f /tmp/memo_test recall hello`
