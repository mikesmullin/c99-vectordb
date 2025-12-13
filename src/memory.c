#include "app_types.h"
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>

// Simple Arena Implementation

void Arena__Init(Arena* arena, size_t size) {
    // Use mmap for large allocations
    void* mem = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) {
        perror("mmap failed");
        __builtin_trap();
    }
    arena->base = (u8*)mem;
    arena->size = size;
    arena->used = 0;
}

void* Arena__Push(Arena* arena, size_t size) {
    // Align to 8 bytes
    size_t alignment = 8;
    size_t current_addr = (size_t)arena->base + arena->used;
    size_t padding = (alignment - (current_addr % alignment)) % alignment;
    
    if (arena->used + padding + size > arena->size) {
        fprintf(stderr, "Arena OOM: used %zu, requested %zu, capacity %zu\n", arena->used, size, arena->size);
        __builtin_trap();
    }
    
    void* ptr = (void*)(arena->base + arena->used + padding);
    arena->used += padding + size;
    return ptr;
}

void Arena__Reset(Arena* arena) {
    arena->used = 0;
}

void Arena__Free(Arena* arena) {
    munmap(arena->base, arena->size);
    arena->base = NULL;
    arena->size = 0;
    arena->used = 0;
}
