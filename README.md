# Memo

A minimalist standalone vectordb w/ Vulkan hardware-acceleration.

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

### macOS (MoltenVK)

This project uses Vulkan compute. On macOS, Vulkan is provided through MoltenVK.

1. Install LunarG Vulkan SDK for macOS (includes MoltenVK and `glslc`).
2. Ensure the SDK binaries are on `PATH` (so `glslc` is found).
3. Point Vulkan loader to MoltenVK ICD, e.g.:

```bash
export VK_ICD_FILENAMES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json"
```

If `vkCreateInstance` or GPU enumeration fails on macOS, check that `VK_ICD_FILENAMES` points to a valid MoltenVK JSON file from your installed SDK.

## Build

```bash
make
```

This will compile the C source code and the GLSL compute shader (`shaders/memo_search.comp`) into `build/memo` and `build/memo_search.spv`.

The primary CLI binary is `build/memo`.

Before running, place model files in `./models/`:

- `models/stories110M.bin`
- `models/tokenizer.bin`

One-time install:

```bash
./models/download.sh
```

`memo save` and `memo recall` require these model files and will now exit with a clear error message if they are missing.

That script downloads the known-good model files and verifies their SHA1 checksums.

## Implementation Details

- **Embeddings**: Uses **TinyLlama 110M** to generate 768-dimensional semantic embeddings.
- **Backend**: The search backend dispatches a compute shader (`memo_search.comp`) that calculates dot products in parallel across the GPU.
