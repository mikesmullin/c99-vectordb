#ifndef VULKAN_BACKEND_H
#define VULKAN_BACKEND_H

#include "app_types.h"

typedef struct {
    // Opaque handles to avoid including vulkan.h in header
    void* instance;
    void* phys_device;
    void* device;
    void* queue;
    u32   queue_family_idx;
    
    void* pipeline;
    void* pipeline_layout;
    void* descriptor_set_layout;
    void* descriptor_pool;
    void* descriptor_set;
    
    void* cmd_pool;
    void* cmd_buffer;
    
    // Buffers
    void* index_buffer;
    void* query_buffer;
    void* score_buffer;
    void* index_memory;
    void* query_memory;
    void* score_memory;
    
    size_t index_buffer_size;
    size_t query_buffer_size;
    size_t score_buffer_size;
    
} VulkanCtx;

void Vulkan_Init(VulkanCtx* ctx);
void Vulkan_Cleanup(VulkanCtx* ctx);
void Vulkan_PrepareBuffers(VulkanCtx* ctx, size_t index_size, size_t query_size, size_t score_size);
void Vulkan_UploadIndex(VulkanCtx* ctx, const void* data, size_t size);
void Vulkan_UploadQuery(VulkanCtx* ctx, const void* data, size_t size);
void Vulkan_Dispatch(VulkanCtx* ctx, u32 count, u32 dim, u32 metric);
void Vulkan_DownloadScores(VulkanCtx* ctx, void* data, size_t size);

#endif
