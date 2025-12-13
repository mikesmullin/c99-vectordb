#include "app_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

// Load the model from file
void LLM__Load(const char* checkpoint_path, Config* config, TransformerWeights* weights, Arena* arena) {
    FILE* file = fopen(checkpoint_path, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open model file: %s\n", checkpoint_path);
        __builtin_trap();
    }

    // 1. Read Config
    if (fread(config, sizeof(Config), 1, file) != 1) {
        fprintf(stderr, "Failed to read config header\n");
        __builtin_trap();
    }

    printf("Model Config:\n");
    printf("  dim: %d\n", config->dim);
    printf("  hidden_dim: %d\n", config->hidden_dim);
    printf("  n_layers: %d\n", config->n_layers);
    printf("  n_heads: %d\n", config->n_heads);
    printf("  n_kv_heads: %d\n", config->n_kv_heads);
    printf("  vocab_size: %d\n", config->vocab_size);
    printf("  seq_len: %d\n", config->seq_len);

    // 2. Allocate Weights in Arena
    // Helper to allocate and read
    #define ALLOC_AND_READ(ptr, count, name) \
        printf("Reading %s (count=%d, size=%zu bytes) at offset %ld...\n", name, (int)(count), (size_t)((count) * sizeof(f32)), ftell(file)); \
        ptr = (f32*)Arena__Push(arena, (count) * sizeof(f32)); \
        if (fread(ptr, sizeof(f32), count, file) != (size_t)(count)) { \
            fprintf(stderr, "Failed to read weights: %s. Expected %zu bytes.\n", name, (size_t)((count) * sizeof(f32))); \
            __builtin_trap(); \
        }

    int head_size = config->dim / config->n_heads;
    
    ALLOC_AND_READ(weights->token_embedding_table, config->vocab_size * config->dim, "token_embedding_table");
    ALLOC_AND_READ(weights->rms_att_weight, config->n_layers * config->dim, "rms_att_weight");
    ALLOC_AND_READ(weights->wq, config->n_layers * config->dim * (config->n_heads * head_size), "wq");
    ALLOC_AND_READ(weights->wk, config->n_layers * config->dim * (config->n_kv_heads * head_size), "wk");
    ALLOC_AND_READ(weights->wv, config->n_layers * config->dim * (config->n_kv_heads * head_size), "wv");
    ALLOC_AND_READ(weights->wo, config->n_layers * (config->n_heads * head_size) * config->dim, "wo");
    ALLOC_AND_READ(weights->rms_ffn_weight, config->n_layers * config->dim, "rms_ffn_weight");
    ALLOC_AND_READ(weights->w1, config->n_layers * config->hidden_dim * config->dim, "w1");
    ALLOC_AND_READ(weights->w2, config->n_layers * config->dim * config->hidden_dim, "w2");
    ALLOC_AND_READ(weights->w3, config->n_layers * config->hidden_dim * config->dim, "w3");
    ALLOC_AND_READ(weights->rms_final_weight, config->dim, "rms_final_weight");
    
    long current_pos = ftell(file);
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, current_pos, SEEK_SET);
    
    long remaining = file_size - current_pos;
    printf("File position: %ld, File size: %ld, Remaining: %ld\n", current_pos, file_size, remaining);

    size_t wcls_size = config->vocab_size * config->dim * sizeof(f32);
    if (remaining >= (long)wcls_size) {
        ALLOC_AND_READ(weights->wcls, config->vocab_size * config->dim, "wcls");
    } else {
        printf("Remaining bytes (%ld) insufficient for wcls (%zu). Using shared weights.\n", remaining, wcls_size);
        weights->wcls = weights->token_embedding_table;
    }

    fclose(file);
    printf("Model Loaded Successfully.\n");
}

void LLM__InitState(RunState* s, Config* p, Arena* arena) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    
    s->x = (f32*)Arena__Push(arena, p->dim * sizeof(f32));
    s->xb = (f32*)Arena__Push(arena, p->dim * sizeof(f32));
    s->xb2 = (f32*)Arena__Push(arena, p->dim * sizeof(f32));
    s->hb = (f32*)Arena__Push(arena, p->hidden_dim * sizeof(f32));
    s->hb2 = (f32*)Arena__Push(arena, p->hidden_dim * sizeof(f32));
    s->q = (f32*)Arena__Push(arena, p->dim * sizeof(f32));
    s->k = (f32*)Arena__Push(arena, kv_dim * sizeof(f32));
    s->v = (f32*)Arena__Push(arena, kv_dim * sizeof(f32));
    s->att = (f32*)Arena__Push(arena, p->n_heads * p->seq_len * sizeof(f32));
    s->logits = (f32*)Arena__Push(arena, p->vocab_size * sizeof(f32));
    s->key_cache = (f32*)Arena__Push(arena, p->n_layers * p->seq_len * kv_dim * sizeof(f32));
    s->value_cache = (f32*)Arena__Push(arena, p->n_layers * p->seq_len * kv_dim * sizeof(f32));
}
