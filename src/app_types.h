#ifndef APP_TYPES_H
#define APP_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Fixed-width types
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t   s8;
typedef int16_t  s16;
typedef int32_t  s32;
typedef int64_t  s64;

typedef float    f32;
typedef double   f64;

// Common Macros
#define ASSERT(c) if (!(c)) { __builtin_trap(); }
#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

// Arena Allocator
typedef struct {
    u8* base;
    size_t size;
    size_t used;
} Arena;

// LLM Config (llama2.c architecture)
typedef struct {
    s32 dim;        // Transformer dimension
    s32 hidden_dim; // for FFN layers
    s32 n_layers;   // number of layers
    s32 n_heads;    // number of query heads
    s32 n_kv_heads; // number of key/value heads (can be < n_heads because of multiquery)
    s32 vocab_size; // vocabulary size, usually 256 (byte-level)
    s32 seq_len;    // max sequence length
} Config;

// Weights (CPU side for loading)
typedef struct {
    // Token embedding table
    f32* token_embedding_table; // (vocab_size, dim)
    // Weights for RMSNorm
    f32* rms_att_weight; // (layer, dim)
    f32* rms_ffn_weight; // (layer, dim)
    // Weights for matmuls
    f32* wq; // (layer, dim, n_heads * head_size)
    f32* wk; // (layer, dim, n_kv_heads * head_size)
    f32* wv; // (layer, dim, n_kv_heads * head_size)
    f32* wo; // (layer, n_heads * head_size, dim)
    // Weights for FFN
    f32* w1; // (layer, hidden_dim, dim)
    f32* w2; // (layer, dim, hidden_dim)
    f32* w3; // (layer, hidden_dim, dim)
    // Final RMSNorm
    f32* rms_final_weight; // (dim,)
    // (Optional) Classifier weights for the logits, on the last layer
    f32* wcls; // (vocab_size, dim)
} TransformerWeights;

// Run State (Buffers for inference)
typedef struct {
    // Current state
    f32* x;      // activation at current time stamp (dim,)
    f32* xb;     // same, but inside a residual branch (dim,)
    f32* xb2;    // an additional buffer just for convenience (dim,)
    f32* hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
    f32* hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
    f32* q;      // query (dim,)
    f32* k;      // key (dim,)
    f32* v;      // value (dim,)
    f32* att;    // buffer for scores/attention values (n_heads, seq_len)
    f32* logits; // output logits
    // KV Cache
    f32* key_cache;   // (layer, seq_len, dim)
    f32* value_cache; // (layer, seq_len, dim)
} RunState;

#endif // APP_TYPES_H
