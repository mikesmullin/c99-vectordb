#include "vectordb.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void VDB_Init(VDB_Context* ctx, Arena* arena) {
    ctx->arena = arena;
    Vulkan_Init(&ctx->vk_ctx);
}

VDB_Index* VDB_Index_Create(VDB_Context* ctx, int dim, VDB_Metric metric, int capacity) {
    VDB_Index* idx = (VDB_Index*)Arena__Push(ctx->arena, sizeof(VDB_Index));
    idx->ctx = ctx;
    idx->dim = dim;
    idx->metric = metric;
    idx->count = 0;
    idx->capacity = capacity;
    
    idx->ids = (u64*)Arena__Push(ctx->arena, capacity * sizeof(u64));
    idx->vectors = (f32*)Arena__Push(ctx->arena, capacity * dim * sizeof(f32));
    
    // Pre-allocate GPU buffers
    // Note: In a real app we'd resize dynamically. Here we alloc capacity.
    size_t idx_size = capacity * dim * sizeof(f32);
    size_t q_size = dim * sizeof(f32);
    size_t score_size = capacity * sizeof(f32);
    Vulkan_PrepareBuffers(&ctx->vk_ctx, idx_size, q_size, score_size);
    
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

// Helper: Compare for qsort (descending score)
static int compare_results(const void* a, const void* b) {
    VDB_Result* r1 = (VDB_Result*)a;
    VDB_Result* r2 = (VDB_Result*)b;
    if (r2->score > r1->score) return 1;
    if (r2->score < r1->score) return -1;
    return 0;
}

void VDB_Search(VDB_Index* idx, const f32* query_vec, int k, const u8* filter_mask, VDB_Result* results) {
    // Count how many vectors survive the filter
    int n_search = 0;
    if (filter_mask) {
        for (int i = 0; i < idx->count; i++) {
            if (filter_mask[i]) n_search++;
        }
        if (n_search == 0) {
            // No survivors — return empty results
            for (int i = 0; i < k; i++) { results[i].id = 0; results[i].score = -1.0f; }
            return;
        }
    } else {
        n_search = idx->count;
    }

    // Build contiguous arrays for the survivors
    f32* search_vecs;
    u64* search_ids;
    if (filter_mask) {
        search_vecs = (f32*)malloc((size_t)n_search * (size_t)idx->dim * sizeof(f32));
        search_ids  = (u64*)malloc((size_t)n_search * sizeof(u64));
        int j = 0;
        for (int i = 0; i < idx->count; i++) {
            if (filter_mask[i]) {
                memcpy(search_vecs + j * idx->dim, idx->vectors + i * idx->dim, (size_t)idx->dim * sizeof(f32));
                search_ids[j] = idx->ids[i];
                j++;
            }
        }
    } else {
        search_vecs = idx->vectors;
        search_ids  = idx->ids;
    }

    VDB_Result* all_results = (VDB_Result*)malloc((size_t)n_search * sizeof(VDB_Result));
    
    // GPU Path
    VulkanCtx* vk = &idx->ctx->vk_ctx;
    
    Vulkan_UploadIndex(vk, search_vecs, (size_t)n_search * (size_t)idx->dim * sizeof(f32));
    Vulkan_UploadQuery(vk, query_vec, (size_t)idx->dim * sizeof(f32));
    
    u32 metric_id = 1;
    if (idx->metric == VDB_METRIC_DOT) metric_id = 2;
    
    Vulkan_Dispatch(vk, n_search, idx->dim, metric_id);
    
    f32* gpu_scores = (f32*)malloc((size_t)n_search * sizeof(f32));
    Vulkan_DownloadScores(vk, gpu_scores, (size_t)n_search * sizeof(f32));
    
    for (int i = 0; i < n_search; i++) {
        all_results[i].id = search_ids[i];
        all_results[i].score = gpu_scores[i];
    }
    free(gpu_scores);
    
    qsort(all_results, (size_t)n_search, sizeof(VDB_Result), compare_results);
    
    int n = (k < n_search) ? k : n_search;
    for (int i = 0; i < n; i++) {
        results[i] = all_results[i];
    }
    for (int i = n; i < k; i++) {
        results[i].id = 0;
        results[i].score = -1.0f;
    }
    
    free(all_results);
    if (filter_mask) {
        free(search_vecs);
        free(search_ids);
    }
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
    MEMO_VLOG("Saved index to %s (%d vectors)\n", filename, idx->count);
}

VDB_Index* VDB_Index_Load(VDB_Context* ctx, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) { return NULL; }
    
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
    MEMO_VLOG("Loaded index from %s (%d vectors)\n", filename, count);
    return idx;
}
