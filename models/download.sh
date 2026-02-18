#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

STORIES_URL="https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin"
TOKENIZER_URL="https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin"

STORIES_FILE="stories110M.bin"
TOKENIZER_FILE="tokenizer.bin"

STORIES_SHA1="ccc55bc06f7d69ef4e090ee3092dca749a88834d"
TOKENIZER_SHA1="04f29a17314a034760ca5cc0ac01849f86948cb7"

download() {
  local url="$1"
  local out="$2"

  if [[ -f "$out" ]]; then
    echo "$out already exists; skipping download"
    return 0
  fi

  echo "Downloading $out..."
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail "$url" -o "$out"
  elif command -v wget >/dev/null 2>&1; then
    wget "$url" -O "$out"
  else
    echo "Error: need curl or wget installed" >&2
    exit 1
  fi
}

verify_sha1() {
  local file="$1"
  local expected="$2"

  if ! command -v sha1sum >/dev/null 2>&1; then
    echo "Warning: sha1sum not found; skipping checksum verification" >&2
    return 0
  fi

  local actual
  actual="$(sha1sum "$file" | awk '{print $1}')"

  if [[ "$actual" != "$expected" ]]; then
    echo "Error: checksum mismatch for $file" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    exit 1
  fi

  echo "Verified $file"
}

download "$STORIES_URL" "$STORIES_FILE"
download "$TOKENIZER_URL" "$TOKENIZER_FILE"

verify_sha1 "$STORIES_FILE" "$STORIES_SHA1"
verify_sha1 "$TOKENIZER_FILE" "$TOKENIZER_SHA1"

echo "Model setup complete: $SCRIPT_DIR/$STORIES_FILE and $SCRIPT_DIR/$TOKENIZER_FILE"
