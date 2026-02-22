#include "../vendor/volk.h"
#include "vulkan_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
#define VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME "VK_KHR_portability_enumeration"
#endif

#ifndef VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
#define VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR 0x00000001
#endif

#ifndef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
#define VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME "VK_KHR_portability_subset"
#endif

// Helper macros
#define VK_CHECK(x) do { VkResult err = x; if (err) { fprintf(stderr, "Vulkan Error: %d\n", err); __builtin_trap(); } } while (0)

// Shader bytecode (will be loaded from file for simplicity in prototype)
static u32* load_shader(const char* path, size_t* size) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("fopen shader"); return NULL; }
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    u32* code = malloc(*size);
    fread(code, 1, *size, f);
    fclose(f);
    return code;
}

// Memory Type Helper
static uint32_t find_memory_type(VkPhysicalDevice phys_dev, uint32_t type_filter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys_dev, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    __builtin_trap();
    return 0;
}

// Buffer Helper
static void create_buffer(VulkanCtx* ctx, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer* buffer, VkDeviceMemory* buffer_memory) {
    VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateBuffer((VkDevice)ctx->device, &bufferInfo, NULL, buffer));

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements((VkDevice)ctx->device, *buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = find_memory_type((VkPhysicalDevice)ctx->phys_device, memRequirements.memoryTypeBits, properties);

    VK_CHECK(vkAllocateMemory((VkDevice)ctx->device, &allocInfo, NULL, buffer_memory));
    VK_CHECK(vkBindBufferMemory((VkDevice)ctx->device, *buffer, *buffer_memory, 0));
}

void Vulkan_Init(VulkanCtx* ctx) {
    VK_CHECK(volkInitialize());

    // 1. Instance
    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "C99 VectorDB";
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    createInfo.pApplicationInfo = &appInfo;

    const char* instance_extensions[1] = {0};
    uint32_t instance_extension_count = 0;
#ifdef __APPLE__
    instance_extensions[instance_extension_count++] = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
    createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    createInfo.enabledExtensionCount = instance_extension_count;
    createInfo.ppEnabledExtensionNames = instance_extension_count ? instance_extensions : NULL;
    
    // Validation Layers (Optional, skip for minimal)
    // const char* layers[] = {"VK_LAYER_KHRONOS_validation"};
    // createInfo.enabledLayerCount = 1;
    // createInfo.ppEnabledLayerNames = layers;

    VK_CHECK(vkCreateInstance(&createInfo, NULL, (VkInstance*)&ctx->instance));
    volkLoadInstance((VkInstance)ctx->instance);

    // 2. Physical Device
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices((VkInstance)ctx->instance, &deviceCount, NULL);
    if (deviceCount == 0) __builtin_trap();
    
    VkPhysicalDevice* devices = malloc(sizeof(VkPhysicalDevice) * deviceCount);
    vkEnumeratePhysicalDevices((VkInstance)ctx->instance, &deviceCount, devices);
    ctx->phys_device = devices[0]; // Pick first
    free(devices);

    // 3. Queue Family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties((VkPhysicalDevice)ctx->phys_device, &queueFamilyCount, NULL);
    VkQueueFamilyProperties* queueProps = malloc(sizeof(VkQueueFamilyProperties) * queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties((VkPhysicalDevice)ctx->phys_device, &queueFamilyCount, queueProps);

    int computeFamily = -1;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeFamily = i;
            break;
        }
    }
    if (computeFamily == -1) __builtin_trap();
    ctx->queue_family_idx = computeFamily;
    free(queueProps);

    // 4. Logical Device
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueCreateInfo.queueFamilyIndex = computeFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;

    const char* device_extensions[1] = {0};
    uint32_t device_extension_count = 0;
#ifdef __APPLE__
    device_extensions[device_extension_count++] = VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME;
#endif
    deviceCreateInfo.enabledExtensionCount = device_extension_count;
    deviceCreateInfo.ppEnabledExtensionNames = device_extension_count ? device_extensions : NULL;

    VK_CHECK(vkCreateDevice((VkPhysicalDevice)ctx->phys_device, &deviceCreateInfo, NULL, (VkDevice*)&ctx->device));
    volkLoadDevice((VkDevice)ctx->device);

    vkGetDeviceQueue((VkDevice)ctx->device, computeFamily, 0, (VkQueue*)&ctx->queue);
    
    // 5. Command Pool
    VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = computeFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool((VkDevice)ctx->device, &poolInfo, NULL, (VkCommandPool*)&ctx->cmd_pool));
    
    // 6. Command Buffer
    VkCommandBufferAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = (VkCommandPool)ctx->cmd_pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers((VkDevice)ctx->device, &allocInfo, (VkCommandBuffer*)&ctx->cmd_buffer));
    
    // 7. Pipeline Layout & Descriptor Set Layout
    VkDescriptorSetLayoutBinding bindings[3];
    // Binding 0: Index (Storage Buffer)
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[0].pImmutableSamplers = NULL;
    
    // Binding 1: Query (Storage Buffer)
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].pImmutableSamplers = NULL;
    
    // Binding 2: Scores (Storage Buffer)
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].pImmutableSamplers = NULL;
    
    VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout((VkDevice)ctx->device, &layoutInfo, NULL, (VkDescriptorSetLayout*)&ctx->descriptor_set_layout));
    
    // Push Constants
    VkPushConstantRange pushConstant;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(u32) * 3; // count, dim, metric
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = (VkDescriptorSetLayout*)&ctx->descriptor_set_layout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstant;
    VK_CHECK(vkCreatePipelineLayout((VkDevice)ctx->device, &pipelineLayoutInfo, NULL, (VkPipelineLayout*)&ctx->pipeline_layout));
    
    // 8. Compute Pipeline
    size_t codeSize;
    u32* code = load_shader("build/memo_search.spv", &codeSize);
    if (!code) {
        fprintf(stderr, "Failed to load shader build/memo_search.spv\n");
        exit(1);
    }
    
    VkShaderModuleCreateInfo createInfoModule = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    createInfoModule.codeSize = codeSize;
    createInfoModule.pCode = code;
    VkShaderModule shaderModule;
    VK_CHECK(vkCreateShaderModule((VkDevice)ctx->device, &createInfoModule, NULL, &shaderModule));
    
    VkPipelineShaderStageCreateInfo shaderStageInfo = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = shaderModule;
    shaderStageInfo.pName = "main";
    
    VkComputePipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = (VkPipelineLayout)ctx->pipeline_layout;
    
    VK_CHECK(vkCreateComputePipelines((VkDevice)ctx->device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, (VkPipeline*)&ctx->pipeline));
    
    vkDestroyShaderModule((VkDevice)ctx->device, shaderModule, NULL);
    free(code);
    
    // 9. Descriptor Pool
    VkDescriptorPoolSize poolSizes[1];
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 3;
    
    VkDescriptorPoolCreateInfo poolInfo2 = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo2.poolSizeCount = 1;
    poolInfo2.pPoolSizes = poolSizes;
    poolInfo2.maxSets = 1;
    VK_CHECK(vkCreateDescriptorPool((VkDevice)ctx->device, &poolInfo2, NULL, (VkDescriptorPool*)&ctx->descriptor_pool));
    
    // 10. Allocate Descriptor Set
    VkDescriptorSetAllocateInfo allocInfo2 = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo2.descriptorPool = (VkDescriptorPool)ctx->descriptor_pool;
    allocInfo2.descriptorSetCount = 1;
    allocInfo2.pSetLayouts = (VkDescriptorSetLayout*)&ctx->descriptor_set_layout;
    VK_CHECK(vkAllocateDescriptorSets((VkDevice)ctx->device, &allocInfo2, (VkDescriptorSet*)&ctx->descriptor_set));
    
    MEMO_VLOG("Vulkan Initialized.\n");
}

void Vulkan_PrepareBuffers(VulkanCtx* ctx, size_t index_size, size_t query_size, size_t score_size) {
    // Cleanup old if exists (omitted for prototype simplicity, assume called once or handle resize later)
    
    ctx->index_buffer_size = index_size;
    ctx->query_buffer_size = query_size;
    ctx->score_buffer_size = score_size;
    
    // Create Buffers (Host Visible for simplicity, though Device Local + Staging is faster)
    // Using HOST_VISIBLE | HOST_COHERENT for easy mapping
    create_buffer(ctx, index_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, (VkBuffer*)&ctx->index_buffer, (VkDeviceMemory*)&ctx->index_memory);
    create_buffer(ctx, query_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, (VkBuffer*)&ctx->query_buffer, (VkDeviceMemory*)&ctx->query_memory);
    create_buffer(ctx, score_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, (VkBuffer*)&ctx->score_buffer, (VkDeviceMemory*)&ctx->score_memory);
    
    // Update Descriptor Set
    VkDescriptorBufferInfo bufferInfos[3];
    bufferInfos[0].buffer = (VkBuffer)ctx->index_buffer;
    bufferInfos[0].offset = 0;
    bufferInfos[0].range = VK_WHOLE_SIZE;
    
    bufferInfos[1].buffer = (VkBuffer)ctx->query_buffer;
    bufferInfos[1].offset = 0;
    bufferInfos[1].range = VK_WHOLE_SIZE;
    
    bufferInfos[2].buffer = (VkBuffer)ctx->score_buffer;
    bufferInfos[2].offset = 0;
    bufferInfos[2].range = VK_WHOLE_SIZE;
    
    VkWriteDescriptorSet descriptorWrites[3];
    for (int i = 0; i < 3; i++) {
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].pNext = NULL;
        descriptorWrites[i].dstSet = (VkDescriptorSet)ctx->descriptor_set;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
        descriptorWrites[i].pImageInfo = NULL;
        descriptorWrites[i].pTexelBufferView = NULL;
    }
    vkUpdateDescriptorSets((VkDevice)ctx->device, 3, descriptorWrites, 0, NULL);
}

void Vulkan_UploadIndex(VulkanCtx* ctx, const void* data, size_t size) {
    void* mapped;
    vkMapMemory((VkDevice)ctx->device, (VkDeviceMemory)ctx->index_memory, 0, size, 0, &mapped);
    memcpy(mapped, data, size);
    vkUnmapMemory((VkDevice)ctx->device, (VkDeviceMemory)ctx->index_memory);
}

void Vulkan_UploadQuery(VulkanCtx* ctx, const void* data, size_t size) {
    void* mapped;
    vkMapMemory((VkDevice)ctx->device, (VkDeviceMemory)ctx->query_memory, 0, size, 0, &mapped);
    memcpy(mapped, data, size);
    vkUnmapMemory((VkDevice)ctx->device, (VkDeviceMemory)ctx->query_memory);
}

void Vulkan_DownloadScores(VulkanCtx* ctx, void* data, size_t size) {
    void* mapped;
    vkMapMemory((VkDevice)ctx->device, (VkDeviceMemory)ctx->score_memory, 0, size, 0, &mapped);
    memcpy(data, mapped, size);
    vkUnmapMemory((VkDevice)ctx->device, (VkDeviceMemory)ctx->score_memory);
}

void Vulkan_Dispatch(VulkanCtx* ctx, u32 count, u32 dim, u32 metric) {
    VkCommandBuffer cb = (VkCommandBuffer)ctx->cmd_buffer;
    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &beginInfo);
    
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)ctx->pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipelineLayout)ctx->pipeline_layout, 0, 1, (VkDescriptorSet*)&ctx->descriptor_set, 0, NULL);
    
    u32 pushConstants[3] = {count, dim, metric};
    vkCmdPushConstants(cb, (VkPipelineLayout)ctx->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), pushConstants);
    
    // Group size is 256
    u32 groupCountX = (count + 255) / 256;
    vkCmdDispatch(cb, groupCountX, 1, 1);
    
    vkEndCommandBuffer(cb);
    
    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cb;
    
    vkQueueSubmit((VkQueue)ctx->queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle((VkQueue)ctx->queue);
}

void Vulkan_Cleanup(VulkanCtx* ctx) {
    vkDeviceWaitIdle((VkDevice)ctx->device);
    // ... (Cleanup omitted for brevity in prototype, OS will reclaim)
}
