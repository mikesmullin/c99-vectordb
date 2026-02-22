#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml

DIM = 384
MAX_K = 100
DEFAULT_DB = "memo"


@dataclass
class Result:
    doc_id: int
    score: float


def vlog(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg, file=sys.stderr)


def has_path_separator(s: str) -> bool:
    return "/" in s


def build_db_paths(base: str, user_cwd: str) -> tuple[Path, Path, Path]:
    if has_path_separator(base):
        if base.startswith("/"):
            prefix = Path(base)
        else:
            prefix = Path(user_cwd) / base
    else:
        prefix = Path(user_cwd) / base
    return (
        prefix.with_suffix(".memo"),
        prefix.with_suffix(".txt"),
        prefix.with_suffix(".meta"),
    )


def ensure_parent_dir(file_path: Path) -> None:
    parent = file_path.parent
    parent.mkdir(parents=True, exist_ok=True)


def save_binary_string_table(path: Path, values: list[str | None]) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<i", len(values)))
        for value in values:
            payload = (value or "").encode("utf-8")
            f.write(struct.pack("<i", len(payload)))
            if payload:
                f.write(payload)


def load_binary_string_table(path: Path) -> list[str | None]:
    if not path.exists():
        return []
    out: list[str | None] = []
    with path.open("rb") as f:
        count_raw = f.read(4)
        if len(count_raw) != 4:
            return out
        count = struct.unpack("<i", count_raw)[0]
        for _ in range(max(0, count)):
            len_raw = f.read(4)
            if len(len_raw) != 4:
                break
            n = struct.unpack("<i", len_raw)[0]
            if n < 0:
                n = 0
            data = f.read(n)
            if len(data) != n:
                break
            out.append(data.decode("utf-8") if n > 0 else None)
    return out


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 1e-8:
        return np.zeros_like(v)
    return v / n


def embed_text_hash(text: str, dim: int = DIM) -> np.ndarray:
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    vec = np.zeros((dim,), dtype=np.float32)
    for token in tokens:
        h = hash(token)
        idx = abs(h) % dim
        sign = 1.0 if (h & 1) else -1.0
        vec[idx] += sign
    return normalize(vec).astype(np.float32)


def parse_yaml_flow_map(expr: str) -> dict[str, Any]:
    parsed = yaml.safe_load(expr)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("filter expression must parse to a YAML mapping")
    return parsed


def raw_meta_to_map(raw_meta: str | None) -> dict[str, Any]:
    if raw_meta is None or raw_meta.strip() == "":
        return {}
    parsed = yaml.safe_load(raw_meta)
    if parsed is None:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def compare_values(lhs: Any, rhs: Any) -> int:
    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        if lhs < rhs:
            return -1
        if lhs > rhs:
            return 1
        return 0
    lhs_s = str(lhs)
    rhs_s = str(rhs)
    if lhs_s < rhs_s:
        return -1
    if lhs_s > rhs_s:
        return 1
    return 0


def bare_equals(value: Any, expected: Any) -> bool:
    if isinstance(value, list):
        return any(str(v) == str(expected) for v in value)
    return str(value) == str(expected)


def eval_condition(data: dict[str, Any], key: str, cond: Any) -> bool:
    if key not in data:
        return False
    value = data[key]

    if isinstance(cond, dict):
        if len(cond) != 1:
            return False
        op, operand = next(iter(cond.items()))
        if op == "$gte":
            return compare_values(value, operand) >= 0
        if op == "$lte":
            return compare_values(value, operand) <= 0
        if op == "$ne":
            return not bare_equals(value, operand)
        if op == "$prefix":
            return isinstance(value, str) and str(value).startswith(str(operand))
        if op == "$contains":
            return isinstance(value, list) and any(str(v) == str(operand) for v in value)
        return False

    return bare_equals(value, cond)


def matches_filter(data: dict[str, Any], filt: dict[str, Any]) -> bool:
    for key, cond in filt.items():
        if key == "$and":
            if not isinstance(cond, list):
                return False
            if not all(isinstance(c, dict) and matches_filter(data, c) for c in cond):
                return False
            continue
        if key == "$or":
            if not isinstance(cond, list):
                return False
            if not any(isinstance(c, dict) and matches_filter(data, c) for c in cond):
                return False
            continue
        if not eval_condition(data, key, cond):
            return False
    return True


def metadata_to_raw(meta: Any) -> str | None:
    if meta is None:
        return None
    if not isinstance(meta, dict):
        raise ValueError("metadata must be a YAML mapping")
    dumped = yaml.safe_dump(meta, default_flow_style=True, sort_keys=False).strip()
    return dumped if dumped else None


def create_index() -> faiss.IndexIDMap2:
    base = faiss.IndexHNSWFlat(DIM, 32)
    base.hnsw.efConstruction = 200
    base.hnsw.efSearch = 64
    return faiss.IndexIDMap2(base)


def load_index(path: Path, verbose: bool) -> faiss.IndexIDMap2:
    if not path.exists():
        return create_index()
    try:
        idx = faiss.read_index(str(path))
    except Exception:
        return create_index()
    if isinstance(idx, faiss.IndexIDMap2):
        return idx
    wrapped = faiss.IndexIDMap2(idx)
    vlog(verbose, "Loaded non-IDMap index; wrapping in IndexIDMap2")
    return wrapped


def get_existing_ids(index: faiss.IndexIDMap2) -> set[int]:
    if index.ntotal == 0:
        return set()
    arr = faiss.vector_to_array(index.id_map)
    return set(int(x) for x in arr.tolist())


def search_all(index: faiss.IndexIDMap2, query_vec: np.ndarray) -> list[Result]:
    if index.ntotal == 0:
        return []
    k = int(index.ntotal)
    scores, ids = index.search(query_vec.reshape(1, -1), k)
    out: list[Result] = []
    for s, doc_id in zip(scores[0].tolist(), ids[0].tolist()):
        if doc_id < 0:
            continue
        out.append(Result(int(doc_id), float(s)))
    return out


def print_recall_result_multiline(rank: int, score: float, text: str) -> None:
    print(f"  [{rank}] Score: {score:.4f} |")
    lines = text.splitlines() or [""]
    for ln in lines:
        print(f"      {ln}")


def command_clean(db_base: str, user_cwd: str) -> int:
    index_path, txt_path, meta_path = build_db_paths(db_base, user_cwd)
    removed_any = False
    for p in (index_path, txt_path, meta_path):
        try:
            p.unlink()
            removed_any = True
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"Error: failed to remove {p}: {e}", file=sys.stderr)
            return 1

    if removed_any:
        print(f"Cleared memory database ({index_path}, {txt_path}, {meta_path})")
    else:
        print(f"Database already empty ({index_path}, {txt_path}, {meta_path})")
    return 0


def parse_save_yaml_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"failed to read input file '{path}'")
    text = path.read_text(encoding="utf-8")
    docs = list(yaml.safe_load_all(text))
    entries: list[dict[str, Any]] = []

    for doc in docs:
        if doc is None:
            continue
        if not isinstance(doc, dict):
            raise ValueError("each YAML document must be a mapping")
        if "body" not in doc:
            raise ValueError("each YAML document requires 'body'")
        body = doc.get("body")
        if not isinstance(body, str) or body.strip() == "":
            raise ValueError("body must be a non-empty string")

        metadata = doc.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a mapping when provided")

        rec: dict[str, Any] = {"body": body, "metadata": metadata}
        if "id" in doc:
            if not isinstance(doc["id"], int) or doc["id"] < 0:
                raise ValueError("id must be a non-negative integer when provided")
            rec["id"] = int(doc["id"])
        entries.append(rec)

    if not entries:
        raise ValueError("input YAML contains no entries")
    return entries


def command_save(db_base: str, save_yaml_path: str, user_cwd: str, verbose: bool) -> int:
    index_path, txt_path, meta_path = build_db_paths(db_base, user_cwd)
    entries = parse_save_yaml_file(Path(save_yaml_path))

    texts = load_binary_string_table(txt_path)
    metas = load_binary_string_table(meta_path)
    if len(metas) < len(texts):
        metas.extend([None] * (len(texts) - len(metas)))
    elif len(texts) < len(metas):
        texts.extend([""] * (len(metas) - len(texts)))

    index = load_index(index_path, verbose)
    existing_ids = get_existing_ids(index)

    for entry in entries:
        note = entry["body"]
        raw_meta = metadata_to_raw(entry.get("metadata"))
        vec = embed_text_hash(note)

        override_id = entry.get("id")
        if override_id is not None:
            if override_id >= len(texts) or override_id not in existing_ids:
                print(f"Error: override id {override_id} does not exist", file=sys.stderr)
                return 1

            remove_vec = np.array([override_id], dtype=np.int64)
            index.remove_ids(remove_vec)
            index.add_with_ids(vec.reshape(1, -1), np.array([override_id], dtype=np.int64))

            texts[override_id] = note
            metas[override_id] = raw_meta
            print(f"Memorized: '{note}' (ID: {override_id})")
        else:
            new_id = len(texts)
            index.add_with_ids(vec.reshape(1, -1), np.array([new_id], dtype=np.int64))
            texts.append(note)
            metas.append(raw_meta)
            print(f"Memorized: '{note}' (ID: {new_id})")

    ensure_parent_dir(index_path)
    ensure_parent_dir(txt_path)
    ensure_parent_dir(meta_path)

    faiss.write_index(index, str(index_path))
    save_binary_string_table(txt_path, texts)
    save_binary_string_table(meta_path, metas)
    return 0


def command_recall(db_base: str, query: str, k: int, filter_expr: str | None, user_cwd: str) -> int:
    index_path, txt_path, meta_path = build_db_paths(db_base, user_cwd)

    texts = load_binary_string_table(txt_path)
    metas = load_binary_string_table(meta_path)
    if len(metas) < len(texts):
        metas.extend([None] * (len(texts) - len(metas)))

    index = load_index(index_path, verbose=False)

    print(f"Top {k} results for '{query}':")
    if index.ntotal == 0:
        return 0

    query_vec = embed_text_hash(query)
    all_results = search_all(index, query_vec)

    active_filter: dict[str, Any] | None = None
    if filter_expr is not None:
        try:
            active_filter = parse_yaml_flow_map(filter_expr)
        except Exception as e:
            print(f"Error: invalid --filter expression: {e}", file=sys.stderr)
            return 1

    shown = 0
    for result in all_results:
        if shown >= k:
            break
        if result.score < -0.9:
            continue

        doc_id = result.doc_id
        if doc_id < 0 or doc_id >= len(texts):
            continue

        if active_filter is not None:
            record = raw_meta_to_map(metas[doc_id] if doc_id < len(metas) else None)
            if not record:
                continue
            if not matches_filter(record, active_filter):
                continue

        text = texts[doc_id] or ""
        print_recall_result_multiline(shown + 1, result.score, text)
        shown += 1

    return 0


def print_help() -> None:
    print("Usage:")
    print("  memo [--help] [-v] [-f <file>]")
    print("  memo save [-f <file>] [-v] <yaml_file>")
    print("  memo recall [-f <file>] [-v] [-k <N>] [--filter <expr>] <query>")
    print("  memo clean [-f <file>] [-v>")
    print()
    print("Options:")
    print("  [-f <file>]        Optional DB basename (default: memo)")
    print("  -v                 Verbose logs to stderr")
    print("  <yaml_file>        YAML file for save input (single or multi-doc using ---)")
    print("                     Each doc requires: metadata: <map>, body: <string>")
    print("                     Optional per-doc id: <int> to overwrite existing record")
    print("  --filter <expr>    Filter recall results by metadata")
    print("  --help             Show this help")


def parse_args(argv: list[str]) -> tuple[dict[str, Any], int]:
    db_base = DEFAULT_DB
    verbose = False
    filter_expr: str | None = None
    positional: list[str] = []

    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "-v":
            verbose = True
            i += 1
            continue
        if arg == "-f":
            if i + 1 >= len(argv):
                print("Error: -f requires a value", file=sys.stderr)
                return {}, 1
            db_base = argv[i + 1]
            i += 2
            continue
        if arg == "--filter":
            if i + 1 >= len(argv):
                print("Error: --filter requires a filter expression", file=sys.stderr)
                return {}, 1
            filter_expr = argv[i + 1]
            i += 2
            continue
        positional.append(arg)
        i += 1

    return {
        "db_base": db_base,
        "verbose": verbose,
        "filter_expr": filter_expr,
        "positional": positional,
    }, 0


def main() -> int:
    parsed, rc = parse_args(sys.argv)
    if rc != 0:
        return rc

    positional = parsed["positional"]
    if not positional or positional[0] in {"--help", "help"}:
        print_help()
        return 0

    user_cwd = os.getcwd()
    command = positional[0]
    db_base = parsed["db_base"]
    verbose = parsed["verbose"]
    filter_expr = parsed["filter_expr"]

    if command == "clean":
        if len(positional) != 1:
            print("Error: clean does not accept extra arguments", file=sys.stderr)
            return 1
        return command_clean(db_base, user_cwd)

    if command == "save":
        if filter_expr is not None:
            print("Error: --filter is only valid with recall", file=sys.stderr)
            return 1
        if len(positional) != 2:
            print("Error: save requires exactly one <yaml_file>", file=sys.stderr)
            return 1
        return command_save(db_base, positional[1], user_cwd, verbose)

    if command == "recall":
        k = 2
        query_start = 1
        if len(positional) >= 4 and positional[1] == "-k":
            try:
                k = int(positional[2])
            except ValueError:
                print("Error: -k requires an integer", file=sys.stderr)
                return 1
            query_start = 3

        if query_start >= len(positional):
            print("Error: recall requires <query>", file=sys.stderr)
            return 1

        if k < 1:
            k = 1
        if k > MAX_K:
            k = MAX_K

        query = " ".join(positional[query_start:]).strip()
        if not query:
            print("Error: recall requires non-empty query", file=sys.stderr)
            return 1

        return command_recall(db_base, query, k, filter_expr, user_cwd)

    print(f"Error: unknown command '{command}'", file=sys.stderr)
    print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
