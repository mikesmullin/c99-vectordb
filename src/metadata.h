#ifndef METADATA_H
#define METADATA_H

#include "app_types.h"

// ---------- Value types ----------

typedef enum {
    META_VAL_STRING,
    META_VAL_INT,
    META_VAL_ARRAY,      // array of strings
} MetaValType;

typedef struct {
    MetaValType type;
    union {
        struct { char* str; int len; } s;
        s64 i;
        struct { char** items; int count; } arr;
    };
} MetaVal;

typedef struct {
    char*    key;
    MetaVal  val;
} MetaField;

typedef struct {
    MetaField* fields;   // arena-allocated
    int        count;    // number of key-value pairs
} MetaRecord;

// ---------- Store ----------

typedef struct {
    char**  raw;         // raw YAML Flow strings per record (NULL = no metadata)
    int     count;
    int     capacity;
    Arena*  arena;
} MetaStore;

// ---------- API ----------

void       MetaStore_Init(MetaStore* ms, int capacity, Arena* arena);
int        MetaStore_Add(MetaStore* ms, const char* yaml_flow);   // returns id; yaml_flow may be NULL
int        MetaStore_Set(MetaStore* ms, int id, const char* yaml_flow);
void       MetaStore_Save(MetaStore* ms, const char* filename);
void       MetaStore_Load(MetaStore* ms, const char* filename);

// Parse one YAML Flow line into a MetaRecord — arena-allocated.
// Handles optional outer braces.
MetaRecord MetaStore_Parse(Arena* arena, const char* yaml_flow);

// Filter: set out_mask[i] = 1 for records that pass filter_expr, 0 otherwise.
// filter_expr is YAML Flow with $operator keys (optional outer braces).
void       MetaStore_Filter(MetaStore* ms, const char* filter_expr,
                            u8* out_mask, int mask_len);

#endif // METADATA_H
