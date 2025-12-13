#ifndef VECTORDB_H
#define VECTORDB_H

#include "app_types.h"
#include "vulkan_backend.h"

typedef enum {
    VDB_BACKEND_CPU,
    VDB_BACKEND_GPU
} VDB_Backend;

typedef enum {
    VDB_METRIC_L2,
    VDB_METRIC_COSINE,
    VDB_METRIC_DOT
} VDB_Metric;

typedef struct {
    Arena* arena;
    VDB_Backend backend;
    VulkanCtx vk_ctx;
} VDB_Context;

typedef struct {
    VDB_Context* ctx; // Backpointer
    int dim;
    int count;
    int capacity;
    VDB_Metric metric;
    
    // Data (SoA or AoS? Let's do flat array for vectors)
    u64* ids;       // [count]
    f32* vectors;   // [count * dim]
} VDB_Index;

typedef struct {
    u64 id;
    f32 score;
} VDB_Result;

// API
void VDB_Init(VDB_Context* ctx, VDB_Backend backend, Arena* arena);
VDB_Index* VDB_Index_Create(VDB_Context* ctx, int dim, VDB_Metric metric, int capacity);
void VDB_Index_Add(VDB_Index* idx, u64 id, const f32* vector);
void VDB_Search(VDB_Index* idx, const f32* query_vec, int k, VDB_Result* results);

// IO
void VDB_Index_Save(VDB_Index* idx, const char* filename);
VDB_Index* VDB_Index_Load(VDB_Context* ctx, const char* filename);

#endif
