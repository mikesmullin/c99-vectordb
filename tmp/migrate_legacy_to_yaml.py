#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Any

import yaml


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


def parse_metadata(raw: str | None) -> dict[str, Any]:
    if raw is None or raw.strip() == "":
        return {}
    try:
        parsed = yaml.safe_load(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def migrate(base: Path, force: bool) -> int:
    txt_path = base.with_suffix(".txt")
    meta_path = base.with_suffix(".meta")
    yaml_path = base.with_suffix(".yaml")

    if not txt_path.exists() and not meta_path.exists():
        print(f"Nothing to migrate: neither {txt_path} nor {meta_path} exists")
        return 0

    if yaml_path.exists() and not force:
        print(f"Refusing to overwrite existing {yaml_path}; pass --force to overwrite")
        return 1

    texts = load_binary_string_table(txt_path)
    metas = load_binary_string_table(meta_path)

    if len(metas) < len(texts):
        metas.extend([None] * (len(texts) - len(metas)))
    elif len(texts) < len(metas):
        texts.extend([""] * (len(metas) - len(texts)))

    docs: list[dict[str, Any]] = []
    for doc_id, body in enumerate(texts):
        docs.append(
            {
                "id": doc_id,
                "metadata": parse_metadata(metas[doc_id] if doc_id < len(metas) else None),
                "body": body or "",
            }
        )

    payload = yaml.safe_dump_all(
        docs,
        explicit_start=True,
        sort_keys=False,
        allow_unicode=True,
    )
    yaml_path.write_text(payload, encoding="utf-8")

    print(f"Wrote {len(docs)} records to {yaml_path}")
    print("FAISS index is preserved as-is (no change to .memo file)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-time migration from legacy .txt/.meta sidecars to .yaml records",
    )
    parser.add_argument(
        "base",
        nargs="?",
        default="memo",
        help="DB base path (default: memo)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .yaml output",
    )
    args = parser.parse_args()

    return migrate(Path(args.base), args.force)


if __name__ == "__main__":
    raise SystemExit(main())
