# C99 VectorDB (Vulkan Accelerated)

A minimalist, standalone Vector Database prototype written in C99, using Vulkan Compute for hardware-accelerated similarity search.

## Features

- **Language**: Pure C99.
- **Acceleration**: Vulkan Compute (GLSL shaders) via `volk` meta-loader.
- **Architecture**: Flat Index (Brute-force), In-Memory.
- **Metric**: Cosine Similarity.
- **Memory Management**: Arena-based allocation.

## Prerequisites

- C Compiler (clang/gcc)
- Vulkan SDK (specifically `glslc` for shader compilation)
- Make

## Build

```bash
make
```

This will compile the C source code and the GLSL compute shader (`shaders/vdb_search.comp`) into `build/vdb` and `build/vdb_search.spv`.

## Usage

Run the interactive CLI:

```bash
./build/vdb
```

### Commands

- `memo <text>`: Embed and store the text.
- `recall <k> <text>`: Search for the top `k` most similar stored texts.
- `save <filename>`: Save the index and text store to disk.
- `load <filename>`: Load the index and text store from disk.
- `exit`: Quit the program.

### Example

```bash
> memo my name is Bob
Memorized: 'my name is Bob' (ID: 0)
> memo cake is for birthdays
Memorized: 'cake is for birthdays' (ID: 1)
> memo carrots are orange
Memorized: 'carrots are orange' (ID: 2)
> recall 2 party food
Top 2 results for 'party food':
  [1] Score: 0.3026 | cake is for birthdays
  [2] Score: 0.2987 | carrots are orange
```

## Implementation Details

- **Embeddings**: Uses **TinyLlama 110M** to generate 768-dimensional semantic embeddings.
- **Backend**: The `VDB_Search` function dispatches a compute shader (`vdb_search.comp`) that calculates dot products in parallel across the GPU.
