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
> memo hello world
Memorized: 'hello world' (ID: 0)
> memo foo bar
Memorized: 'foo bar' (ID: 1)
> recall 1 hello
Top 1 results for 'hello':
  [1] Score: -0.0137 | hello world
```

## Implementation Details

- **Embeddings**: Currently uses a deterministic pseudo-random generator for demonstration. In a real application, you would plug in an inference engine (like the C99-LLM) to generate semantic embeddings.
- **Backend**: The `VDB_Search` function dispatches a compute shader (`vdb_search.comp`) that calculates dot products in parallel across the GPU.
