#include "vectordb.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void VDB_Init(VDB_Context* ctx, VDB_Backend backend, Arena* arena) {
    ctx->backend = backend;
    ctx->arena = arena;
}

VDB_Index* VDB_Index_Create(VDB_Context* ctx, int dim, VDB_Metric metric, int capacity) {
    VDB_Index* idx = (VDB_Index*)Arena__Push(ctx->arena, sizeof(VDB_Index));
    idx->dim = dim;
    idx->metric = metric;
    idx->count = 0;
    idx->capacity = capacity;
    
    idx->ids = (u64*)Arena__Push(ctx->arena, capacity * sizeof(u64));
    idx->vectors = (f32*)Arena__Push(ctx->arena, capacity * dim * sizeof(f32));
    
    return idx;
}

void VDB_Index_Add(VDB_Index* idx, u64 id, const f32* vector) {
    if (idx->count >= idx->capacity) {
        fprintf(stderr, "VDB Error: Index full (cap=%d)\n", idx->capacity);
        return;
    }
    
    int i = idx->count;
    idx->ids[i] = id;
    
    // Copy vector
    // memcpy(idx->vectors + i * idx->dim, vector, idx->dim * sizeof(f32));
    // Manual copy to be safe
    for (int j = 0; j < idx->dim; j++) {
        idx->vectors[i * idx->dim + j] = vector[j];
    }
    
    idx->count++;
}

// Helper: Cosine Similarity
static f32 cosine_similarity(const f32* a, const f32* b, int dim) {
    f32 dot = 0.0f;
    f32 norm_a = 0.0f;
    f32 norm_b = 0.0f;
    
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a == 0 || norm_b == 0) return 0.0f;
    return dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

// Helper: Compare for qsort (descending score)
static int compare_results(const void* a, const void* b) {
    VDB_Result* r1 = (VDB_Result*)a;
    VDB_Result* r2 = (VDB_Result*)b;
    if (r2->score > r1->score) return 1;
    if (r2->score < r1->score) return -1;
    return 0;
}

void VDB_Search(VDB_Index* idx, const f32* query_vec, int k, VDB_Result* results) {
    // 1. Calculate all scores
    // Note: For large N, we shouldn't alloc on stack. But for prototype, let's use a temp buffer.
    // Ideally we use a scratch arena or the main arena.
    // For now, let's just malloc a temp array of results to sort.
    
    VDB_Result* all_results = (VDB_Result*)malloc(idx->count * sizeof(VDB_Result));
    
    for (int i = 0; i < idx->count; i++) {
        f32 score = 0.0f;
        f32* vec = idx->vectors + (i * idx->dim);
        
        if (idx->metric == VDB_METRIC_COSINE) {
            score = cosine_similarity(query_vec, vec, idx->dim);
        } else {
            // Default to Dot Product
             for (int j = 0; j < idx->dim; j++) {
                score += query_vec[j] * vec[j];
            }
        }
        
        all_results[i].id = idx->ids[i];
        all_results[i].score = score;
    }
    
    // 2. Sort (Naive full sort for now)
    qsort(all_results, idx->count, sizeof(VDB_Result), compare_results);
    
    // 3. Copy top k
    int n = (k < idx->count) ? k : idx->count;
    for (int i = 0; i < n; i++) {
        results[i] = all_results[i];
    }
    // Fill rest with empty if needed
    for (int i = n; i < k; i++) {
        results[i].id = 0;
        results[i].score = -1.0f;
    }
    
    free(all_results);
}

void VDB_Index_Save(VDB_Index* idx, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) { perror("fopen save"); return; }
    
    fwrite(&idx->dim, sizeof(int), 1, f);
    fwrite(&idx->count, sizeof(int), 1, f);
    fwrite(&idx->metric, sizeof(int), 1, f);
    
    fwrite(idx->ids, sizeof(u64), idx->count, f);
    fwrite(idx->vectors, sizeof(f32), idx->count * idx->dim, f);
    
    fclose(f);
    printf("Saved index to %s (%d vectors)\n", filename, idx->count);
}

VDB_Index* VDB_Index_Load(VDB_Context* ctx, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("fopen load"); return NULL; }
    
    int dim, count, metric;
    if (fread(&dim, sizeof(int), 1, f) != 1) return NULL;
    if (fread(&count, sizeof(int), 1, f) != 1) return NULL;
    if (fread(&metric, sizeof(int), 1, f) != 1) return NULL;
    
    // Alloc
    VDB_Index* idx = VDB_Index_Create(ctx, dim, (VDB_Metric)metric, count + 1000); // Alloc with some room
    
    idx->count = count;
    fread(idx->ids, sizeof(u64), count, f);
    fread(idx->vectors, sizeof(f32), count * dim, f);
    
    fclose(f);
    printf("Loaded index from %s (%d vectors)\n", filename, count);
    return idx;
}
