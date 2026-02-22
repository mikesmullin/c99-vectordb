#include "app_types.h"
#include "../vendor/volk.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifndef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
#define VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME "VK_KHR_portability_enumeration"
#endif

#ifndef VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
#define VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR 0x00000001
#endif

#ifndef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
#define VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME "VK_KHR_portability_subset"
#endif

// Vulkan Context
typedef struct {
    VkInstance instance;
    VkDebugUtilsMessengerEXT debug_messenger;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    u32 compute_queue_family_index;
    VkCommandPool command_pool;
    
    // Compute Pipeline
    VkDescriptorSetLayout descriptor_set_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    
    // Memory
    VkBuffer weights_buffer;
    VkDeviceMemory weights_memory;
    
    VkBuffer input_buffer;
    VkDeviceMemory input_memory;
    void* input_mapped;
    
    VkBuffer output_buffer;
    VkDeviceMemory output_memory;
    void* output_mapped;
    
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;
    
    VkCommandBuffer command_buffer;
    VkFence fence;
} LLM_VulkanCtx;

// Validation Layers
#ifdef NDEBUG
    const bool enable_validation_layers = false;
#else
    const bool enable_validation_layers = false; // Disabled for now
#endif

const char* validation_layers[] = {
    "VK_LAYER_KHRONOS_validation"
};

// Helper: Find Memory Type
uint32_t findMemoryType(VkPhysicalDevice physical_device, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    fprintf(stderr, "Failed to find suitable memory type!\n");
    __builtin_trap();
    return 0;
}

// Helper: Create Buffer
void createBuffer(LLM_VulkanCtx* ctx, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer* buffer, VkDeviceMemory* bufferMemory) {
    VkBufferCreateInfo bufferInfo = {0};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(ctx->device, &bufferInfo, NULL, buffer) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create buffer!\n");
        __builtin_trap();
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(ctx->device, *buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(ctx->physical_device, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(ctx->device, &allocInfo, NULL, bufferMemory) != VK_SUCCESS) {
        fprintf(stderr, "Failed to allocate buffer memory!\n");
        __builtin_trap();
    }

    vkBindBufferMemory(ctx->device, *buffer, *bufferMemory, 0);
}

// Debug Callback
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    (void)messageSeverity;
    (void)messageType;
    (void)pUserData;
    MEMO_VLOG("Validation Layer: %s\n", pCallbackData->pMessage);
    return VK_FALSE;
}

static void print_macos_vulkan_hint(void) {
#ifdef __APPLE__
    fprintf(stderr,
        "macOS hint: install Vulkan SDK with MoltenVK and ensure VK_ICD_FILENAMES points to MoltenVK_icd.json.\n");
#endif
}

void LLM_VulkanCtx__Init(LLM_VulkanCtx* ctx) {
    // 1. Initialize Volk
    if (volkInitialize() != VK_SUCCESS) {
        fprintf(stderr, "Failed to initialize Volk. Is Vulkan installed?\n");
        print_macos_vulkan_hint();
        __builtin_trap();
    }

    // 2. Create Instance
    VkApplicationInfo app_info = {0};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "C99 LLM";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "No Engine";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo create_info = {0};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    const char* extensions[2] = {0};
    u32 extension_count = 0;
#ifdef __APPLE__
    extensions[extension_count++] = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
    create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    if (enable_validation_layers) {
        extensions[extension_count++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    }

    if (enable_validation_layers) {
        create_info.enabledLayerCount = ARRAY_LEN(validation_layers);
        create_info.ppEnabledLayerNames = validation_layers;
    } else {
        create_info.enabledLayerCount = 0;
    }
    create_info.enabledExtensionCount = extension_count;
    create_info.ppEnabledExtensionNames = extension_count ? extensions : NULL;

    VkResult result = vkCreateInstance(&create_info, NULL, &ctx->instance);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "Failed to create Vulkan Instance! Error: %d\n", result);
        print_macos_vulkan_hint();
        __builtin_trap();
    }

    // Load Instance Functions
    volkLoadInstance(ctx->instance);

    // 3. Setup Debug Messenger
    if (enable_validation_layers) {
        VkDebugUtilsMessengerCreateInfoEXT debug_info = {0};
        debug_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debug_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debug_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debug_info.pfnUserCallback = debugCallback;
        
        if (vkCreateDebugUtilsMessengerEXT(ctx->instance, &debug_info, NULL, &ctx->debug_messenger) != VK_SUCCESS) {
            fprintf(stderr, "Failed to set up debug messenger!\n");
        }
    }

    // 4. Pick Physical Device
    u32 device_count = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &device_count, NULL);
    if (device_count == 0) {
        fprintf(stderr, "Failed to find GPUs with Vulkan support!\n");
        print_macos_vulkan_hint();
        __builtin_trap();
    }
    VkPhysicalDevice devices[8];
    vkEnumeratePhysicalDevices(ctx->instance, &device_count, devices);
    
    // Just pick the first one for now (usually the discrete GPU)
    ctx->physical_device = devices[0];
    
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx->physical_device, &props);
    MEMO_VLOG("Selected GPU: %s\n", props.deviceName);

    // 5. Find Compute Queue Family
    u32 queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device, &queue_family_count, NULL);
    VkQueueFamilyProperties queue_families[16];
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device, &queue_family_count, queue_families);

    int compute_family = -1;
    for (u32 i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_family = i;
            break;
        }
    }

    if (compute_family == -1) {
        fprintf(stderr, "Failed to find a compute queue family!\n");
        __builtin_trap();
    }
    ctx->compute_queue_family_index = (u32)compute_family;

    // 6. Create Logical Device
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info = {0};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = ctx->compute_queue_family_index;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_create_info = {0};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.pQueueCreateInfos = &queue_create_info;
    device_create_info.queueCreateInfoCount = 1;

    const char* device_extensions[1] = {0};
    u32 device_extension_count = 0;
#ifdef __APPLE__
    device_extensions[device_extension_count++] = VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME;
#endif
    device_create_info.enabledExtensionCount = device_extension_count;
    device_create_info.ppEnabledExtensionNames = device_extension_count ? device_extensions : NULL;

    if (enable_validation_layers) {
        device_create_info.enabledLayerCount = ARRAY_LEN(validation_layers);
        device_create_info.ppEnabledLayerNames = validation_layers;
    } else {
        device_create_info.enabledLayerCount = 0;
    }

    if (vkCreateDevice(ctx->physical_device, &device_create_info, NULL, &ctx->device) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create logical device!\n");
        __builtin_trap();
    }

    // Load Device Functions
    volkLoadDevice(ctx->device);

    vkGetDeviceQueue(ctx->device, ctx->compute_queue_family_index, 0, &ctx->compute_queue);

    // 7. Create Command Pool
    VkCommandPoolCreateInfo pool_info = {0};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = ctx->compute_queue_family_index;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(ctx->device, &pool_info, NULL, &ctx->command_pool) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create command pool!\n");
        __builtin_trap();
    }
}

// Helper to load shader
VkShaderModule createShaderModule(LLM_VulkanCtx* ctx, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open shader file: %s\n", filename);
        __builtin_trap();
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    u32* code = (u32*)malloc(size);
    fread(code, 1, size, file);
    fclose(file);
    
    VkShaderModuleCreateInfo createInfo = {0};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = size;
    createInfo.pCode = code;
    
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(ctx->device, &createInfo, NULL, &shaderModule) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create shader module!\n");
        __builtin_trap();
    }
    free(code);
    return shaderModule;
}

void LLM_VulkanCtx__SetupPipeline(LLM_VulkanCtx* ctx) {
    // Descriptor Set Layout
    VkDescriptorSetLayoutBinding bindings[3] = {0};
    // Binding 0: Weights (Storage Buffer)
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // Binding 1: Input (Storage Buffer)
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // Binding 2: Output (Storage Buffer)
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {0};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(ctx->device, &layoutInfo, NULL, &ctx->descriptor_set_layout) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create descriptor set layout!\n");
        __builtin_trap();
    }

    // Push Constants
    VkPushConstantRange pushConstantRange = {0};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = 3 * sizeof(uint32_t); // weight_offset, n, d

    // Pipeline Layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {0};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &ctx->descriptor_set_layout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(ctx->device, &pipelineLayoutInfo, NULL, &ctx->pipeline_layout) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create pipeline layout!\n");
        __builtin_trap();
    }

    // Pipeline
    VkShaderModule compShaderModule = createShaderModule(ctx, "build/headless.spv");

    VkPipelineShaderStageCreateInfo shaderStageInfo = {0};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = compShaderModule;
    shaderStageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo = {0};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = ctx->pipeline_layout;

    if (vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &ctx->pipeline) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create compute pipeline!\n");
        __builtin_trap();
    }

    vkDestroyShaderModule(ctx->device, compShaderModule, NULL);
}

void LLM_VulkanCtx__UploadWeights(LLM_VulkanCtx* ctx, void* data, size_t size) {
    // Create Staging Buffer
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(ctx, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBuffer, &stagingBufferMemory);

    void* mappedData;
    vkMapMemory(ctx->device, stagingBufferMemory, 0, size, 0, &mappedData);
    memcpy(mappedData, data, size);
    vkUnmapMemory(ctx->device, stagingBufferMemory);

    // Create Device Local Buffer
    createBuffer(ctx, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &ctx->weights_buffer, &ctx->weights_memory);

    // Copy
    VkCommandBufferAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = ctx->command_pool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(ctx->device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo = {0};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    VkBufferCopy copyRegion = {0};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, stagingBuffer, ctx->weights_buffer, 1, &copyRegion);
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(ctx->compute_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx->compute_queue);

    vkFreeCommandBuffers(ctx->device, ctx->command_pool, 1, &commandBuffer);
    vkDestroyBuffer(ctx->device, stagingBuffer, NULL);
    vkFreeMemory(ctx->device, stagingBufferMemory, NULL);
    
    MEMO_VLOG("Weights uploaded to GPU (%zu bytes)\n", size);
}

void LLM_VulkanCtx__PrepareBuffers(LLM_VulkanCtx* ctx, size_t max_input_size, size_t max_output_size) {
    // Input Buffer (Host Visible for frequent updates)
    createBuffer(ctx, max_input_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &ctx->input_buffer, &ctx->input_memory);
    vkMapMemory(ctx->device, ctx->input_memory, 0, max_input_size, 0, &ctx->input_mapped);

    // Output Buffer (Host Visible for reading back)
    createBuffer(ctx, max_output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &ctx->output_buffer, &ctx->output_memory);
    vkMapMemory(ctx->device, ctx->output_memory, 0, max_output_size, 0, &ctx->output_mapped);

    // Descriptor Pool
    VkDescriptorPoolSize poolSizes[1] = {0};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 3;

    VkDescriptorPoolCreateInfo poolInfo = {0};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = 1;

    if (vkCreateDescriptorPool(ctx->device, &poolInfo, NULL, &ctx->descriptor_pool) != VK_SUCCESS) {
        fprintf(stderr, "Failed to create descriptor pool!\n");
        __builtin_trap();
    }

    // Descriptor Set
    VkDescriptorSetAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = ctx->descriptor_pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &ctx->descriptor_set_layout;

    if (vkAllocateDescriptorSets(ctx->device, &allocInfo, &ctx->descriptor_set) != VK_SUCCESS) {
        fprintf(stderr, "Failed to allocate descriptor sets!\n");
        __builtin_trap();
    }

    // Update Descriptor Set
    VkDescriptorBufferInfo bufferInfo[3];
    bufferInfo[0].buffer = ctx->weights_buffer;
    bufferInfo[0].offset = 0;
    bufferInfo[0].range = VK_WHOLE_SIZE;

    bufferInfo[1].buffer = ctx->input_buffer;
    bufferInfo[1].offset = 0;
    bufferInfo[1].range = VK_WHOLE_SIZE;

    bufferInfo[2].buffer = ctx->output_buffer;
    bufferInfo[2].offset = 0;
    bufferInfo[2].range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet descriptorWrites[3] = {0};
    for (int i = 0; i < 3; i++) {
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = ctx->descriptor_set;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfo[i];
    }

    vkUpdateDescriptorSets(ctx->device, 3, descriptorWrites, 0, NULL);

    // Create Command Buffer & Fence
    VkCommandBufferAllocateInfo cmdAllocInfo = {0};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = ctx->command_pool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    vkAllocateCommandBuffers(ctx->device, &cmdAllocInfo, &ctx->command_buffer);

    VkFenceCreateInfo fenceInfo = {0};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(ctx->device, &fenceInfo, NULL, &ctx->fence);
}

void LLM_VulkanCtx__MatMul(LLM_VulkanCtx* ctx, f32* xout, f32* x, u32 weight_offset, int n, int d) {
    // Copy input to mapped memory
    memcpy(ctx->input_mapped, x, n * sizeof(f32));

    // Record Command Buffer
    vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX);
    vkResetFences(ctx->device, 1, &ctx->fence);
    vkResetCommandBuffer(ctx->command_buffer, 0);

    VkCommandBufferBeginInfo beginInfo = {0};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(ctx->command_buffer, &beginInfo);

    vkCmdBindPipeline(ctx->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline);
    vkCmdBindDescriptorSets(ctx->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline_layout, 0, 1, &ctx->descriptor_set, 0, NULL);

    uint32_t pushConstants[3] = {weight_offset, (uint32_t)n, (uint32_t)d};
    vkCmdPushConstants(ctx->command_buffer, ctx->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), pushConstants);

    // Dispatch
    // Local size is 256. We need d threads.
    uint32_t groupCountX = (d + 255) / 256;
    vkCmdDispatch(ctx->command_buffer, groupCountX, 1, 1);

    vkEndCommandBuffer(ctx->command_buffer);

    // Submit
    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &ctx->command_buffer;

    vkQueueSubmit(ctx->compute_queue, 1, &submitInfo, ctx->fence);

    // Wait
    vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX);

    // Read output
    memcpy(xout, ctx->output_mapped, d * sizeof(f32));
}

void LLM_VulkanCtx__Cleanup(LLM_VulkanCtx* ctx) {
    vkDeviceWaitIdle(ctx->device);
    
    vkDestroyFence(ctx->device, ctx->fence, NULL);
    vkDestroyDescriptorPool(ctx->device, ctx->descriptor_pool, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, ctx->descriptor_set_layout, NULL);
    vkDestroyPipeline(ctx->device, ctx->pipeline, NULL);
    vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layout, NULL);
    
    vkDestroyBuffer(ctx->device, ctx->weights_buffer, NULL);
    vkFreeMemory(ctx->device, ctx->weights_memory, NULL);
    
    vkDestroyBuffer(ctx->device, ctx->input_buffer, NULL);
    vkFreeMemory(ctx->device, ctx->input_memory, NULL);
    
    vkDestroyBuffer(ctx->device, ctx->output_buffer, NULL);
    vkFreeMemory(ctx->device, ctx->output_memory, NULL);

    vkDestroyCommandPool(ctx->device, ctx->command_pool, NULL);
    vkDestroyDevice(ctx->device, NULL);
    if (enable_validation_layers) {
        vkDestroyDebugUtilsMessengerEXT(ctx->instance, ctx->debug_messenger, NULL);
    }
    vkDestroyInstance(ctx->instance, NULL);
}
