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

void Arena__Init(Arena* arena, size_t size);
void* Arena__Push(Arena* arena, size_t size);
void Arena__Reset(Arena* arena);
void Arena__Free(Arena* arena);

#endif
