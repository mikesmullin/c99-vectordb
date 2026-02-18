# memo — Usage Guide for AI Agents

`memo` is a lightweight vector database for agents to store and recall important facts as long-term memory.

## Command help (actual output)

```text
Usage:
  memo [--help] [-v] [-f <file>]
  memo save [-f <file>] [-v] [<id>] <note>
  memo recall [-f <file>] [-v] [-k <N>] <query>
  memo clear [-f <file>] [-v]

Options:
  [-f <file>] Optional DB basename (default: db/memo)
  -v          Verbose logs to stderr
  --help      Show this help
```

## Core behavior

- `memo` or `memo --help` prints help.
- `memo save <note>` stores a new memory.
- `memo save <id> <note>` overwrites memory at an existing ID.
- `memo recall <query>` recalls top matches (default `k=2`).
- `memo recall -k <N> <query>` recalls top `N` matches (`N` capped at 100).
- `memo clear` wipes the current memory DB files.
- `-f <file>` is optional and changes DB basename (default files are under `db/`).
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

$ memo clear
Cleared memory database (db/memo.memo, db/memo.txt)
```

## Output contract

- Normal mode: only final subcommand result goes to stdout.
- Verbose mode (`-v`): extra debug/startup info goes to stderr.

## Fast usage patterns

- Default DB (recommended):
  - `memo save remember this`
  - `memo recall remember`
- Optional custom DB path with `-f`:
  - `memo -f /tmp/memo_test save hello world`
  - `memo -f /tmp/memo_test recall hello`
