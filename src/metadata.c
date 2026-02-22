#include "metadata.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <inttypes.h>

// Arena functions (implemented in memory.c)
void* Arena__Push(Arena* arena, size_t size);

// ============================================================
// Arena helpers
// ============================================================

static char* arena_strdup(Arena* a, const char* s, int len) {
    char* p = (char*)Arena__Push(a, (size_t)len + 1);
    memcpy(p, s, (size_t)len);
    p[len] = '\0';
    return p;
}

// ============================================================
// MetaStore basics
// ============================================================

void MetaStore_Init(MetaStore* ms, int capacity, Arena* arena) {
    ms->raw = (char**)Arena__Push(arena, (size_t)capacity * sizeof(char*));
    memset(ms->raw, 0, (size_t)capacity * sizeof(char*));
    ms->count = 0;
    ms->capacity = capacity;
    ms->arena = arena;
}

int MetaStore_Add(MetaStore* ms, const char* yaml_flow) {
    if (ms->count >= ms->capacity) return -1;
    int id = ms->count++;
    if (yaml_flow && yaml_flow[0] != '\0') {
        int len = (int)strlen(yaml_flow);
        ms->raw[id] = arena_strdup(ms->arena, yaml_flow, len);
    } else {
        ms->raw[id] = NULL;
    }
    return id;
}

int MetaStore_Set(MetaStore* ms, int id, const char* yaml_flow) {
    if (id < 0 || id >= ms->count) return 0;
    if (yaml_flow && yaml_flow[0] != '\0') {
        int len = (int)strlen(yaml_flow);
        ms->raw[id] = arena_strdup(ms->arena, yaml_flow, len);
    } else {
        ms->raw[id] = NULL;
    }
    return 1;
}

// ============================================================
// Persistence — binary sidecar (same format as TextStore)
// ============================================================

void MetaStore_Save(MetaStore* ms, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    fwrite(&ms->count, sizeof(int), 1, f);
    for (int i = 0; i < ms->count; i++) {
        int len = ms->raw[i] ? (int)strlen(ms->raw[i]) : 0;
        fwrite(&len, sizeof(int), 1, f);
        if (len > 0) fwrite(ms->raw[i], 1, (size_t)len, f);
    }
    fclose(f);
}

void MetaStore_Load(MetaStore* ms, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return;
    int count;
    if (fread(&count, sizeof(int), 1, f) != 1) { fclose(f); return; }
    for (int i = 0; i < count; i++) {
        int len;
        if (fread(&len, sizeof(int), 1, f) != 1) break;
        if (len > 0) {
            char* str = (char*)Arena__Push(ms->arena, (size_t)len + 1);
            if (fread(str, 1, (size_t)len, f) != (size_t)len) break;
            str[len] = '\0';
            ms->raw[ms->count++] = str;
        } else {
            ms->raw[ms->count++] = NULL;
        }
    }
    fclose(f);
}

// ============================================================
// Minimal YAML Flow scanner
// ============================================================
// Handles: {key: val, key2: val2}  or  key: val, key2: val2
// Values: bare words, integers, [arr, items]
// Nested: {$gte: 2}  (operator sub-maps)

// Skip whitespace
static const char* skip_ws(const char* p) {
    while (*p && (*p == ' ' || *p == '\t')) p++;
    return p;
}

// Read a bare token (word / number / date / $op) up to delimiter
// Stops at , : { } [ ] and whitespace
static const char* read_token(const char* p, const char** out_start, int* out_len) {
    p = skip_ws(p);
    *out_start = p;
    *out_len = 0;
    while (*p && *p != ',' && *p != ':' && *p != '{' && *p != '}'
           && *p != '[' && *p != ']' && *p != ' ' && *p != '\t') {
        p++;
    }
    *out_len = (int)(p - *out_start);
    return p;
}

// Check if string is a decimal integer (optional leading +/-)
static int is_int_str(const char* s, int len) {
    if (len == 0) return 0;
    int i = 0;
    if (s[0] == '+' || s[0] == '-') { i = 1; if (len == 1) return 0; }
    for (; i < len; i++) {
        if (s[i] < '0' || s[i] > '9') return 0;
    }
    return 1;
}

static s64 parse_int(const char* s, int len) {
    char buf[64];
    int n = len < 63 ? len : 63;
    memcpy(buf, s, (size_t)n);
    buf[n] = '\0';
    return (s64)strtoll(buf, NULL, 10);
}

// Parse an array: [item1, item2, ...]
// Returns pointer past the closing ']'
static const char* parse_array(Arena* arena, const char* p, MetaVal* out) {
    out->type = META_VAL_ARRAY;
    // Temp storage — max 64 items
    char* items[64];
    int count = 0;

    p = skip_ws(p);
    if (*p == '[') p++;
    while (*p && *p != ']') {
        p = skip_ws(p);
        if (*p == ']') break;
        const char* tok; int tok_len;
        p = read_token(p, &tok, &tok_len);
        if (tok_len > 0 && count < 64) {
            items[count++] = arena_strdup(arena, tok, tok_len);
        }
        p = skip_ws(p);
        if (*p == ',') p++;
    }
    if (*p == ']') p++;

    out->arr.items = (char**)Arena__Push(arena, (size_t)count * sizeof(char*));
    out->arr.count = count;
    for (int i = 0; i < count; i++) out->arr.items[i] = items[i];
    return p;
}

// Parse a single value (bare token, integer, or array)
static const char* parse_value(Arena* arena, const char* p, MetaVal* out) {
    p = skip_ws(p);
    if (*p == '[') {
        return parse_array(arena, p, out);
    }
    // Read token
    const char* tok; int tok_len;
    p = read_token(p, &tok, &tok_len);
    if (is_int_str(tok, tok_len)) {
        out->type = META_VAL_INT;
        out->i = parse_int(tok, tok_len);
    } else {
        out->type = META_VAL_STRING;
        out->s.str = arena_strdup(arena, tok, tok_len);
        out->s.len = tok_len;
    }
    return p;
}

// Parse a YAML Flow mapping body (without outer braces) into fields array.
// Returns number of fields parsed.
static int parse_fields(Arena* arena, const char* p, MetaField* fields, int max_fields) {
    int count = 0;
    while (*p && count < max_fields) {
        p = skip_ws(p);
        if (*p == '\0' || *p == '}' || *p == ']') break;

        // Read key
        const char* key_start; int key_len;
        p = read_token(p, &key_start, &key_len);
        if (key_len == 0) break;

        char* key = arena_strdup(arena, key_start, key_len);

        p = skip_ws(p);
        if (*p == ':') p++; // skip colon
        p = skip_ws(p);

        // Value might be a sub-map {$op: val} or array or bare
        if (*p == '{') {
            // Sub-map: read the inner key-value pair as a single field
            // We store: key = outer key, val.type = STRING, val.s = "$op:val"
            // Actually let's parse properly: the value IS the sub-map.
            // For filter evaluation we'll re-parse. Store raw for now.
            // Better approach: parse recursively and store as string "$op\0val".
            // Simplest: just store the raw sub-map text.
            const char* brace_start = p;
            int depth = 0;
            while (*p) {
                if (*p == '{') depth++;
                else if (*p == '}') { depth--; if (depth == 0) { p++; break; } }
                p++;
            }
            int raw_len = (int)(p - brace_start);
            fields[count].key = key;
            fields[count].val.type = META_VAL_STRING;
            fields[count].val.s.str = arena_strdup(arena, brace_start, raw_len);
            fields[count].val.s.len = raw_len;
            count++;
        } else if (*p == '[') {
            fields[count].key = key;
            p = parse_value(arena, p, &fields[count].val);
            count++;
        } else {
            fields[count].key = key;
            p = parse_value(arena, p, &fields[count].val);
            count++;
        }

        p = skip_ws(p);
        if (*p == ',') p++;
    }
    return count;
}

MetaRecord MetaStore_Parse(Arena* arena, const char* yaml_flow) {
    MetaRecord rec;
    rec.fields = NULL;
    rec.count = 0;
    if (!yaml_flow || yaml_flow[0] == '\0') return rec;

    const char* p = skip_ws(yaml_flow);
    // Strip optional outer braces
    if (*p == '{') {
        p++;
        // Find matching close brace? We'll just parse until end/close
    }

    MetaField* fields = (MetaField*)Arena__Push(arena, 32 * sizeof(MetaField));
    int n = parse_fields(arena, p, fields, 32);
    rec.fields = fields;
    rec.count = n;
    return rec;
}

// ============================================================
// Filter evaluation
// ============================================================

// Compare a MetaVal to a string/int for operator evaluation.
// Returns: <0 if val < cmp, 0 if equal, >0 if val > cmp
static int meta_val_cmp(const MetaVal* val, const char* cmp_str, int cmp_len) {
    if (val->type == META_VAL_INT) {
        if (is_int_str(cmp_str, cmp_len)) {
            s64 cv = parse_int(cmp_str, cmp_len);
            if (val->i < cv) return -1;
            if (val->i > cv) return 1;
            return 0;
        }
        // type mismatch — compare as strings
    }
    if (val->type == META_VAL_STRING) {
        int n = val->s.len < cmp_len ? val->s.len : cmp_len;
        int r = strncmp(val->s.str, cmp_str, (size_t)n);
        if (r != 0) return r;
        return val->s.len - cmp_len;
    }
    return -1; // arrays don't compare with ordinal ops
}

// Check if a MetaVal equals a value token
static int meta_val_eq(const MetaVal* val, const char* tok, int tok_len) {
    if (val->type == META_VAL_INT && is_int_str(tok, tok_len)) {
        return val->i == parse_int(tok, tok_len);
    }
    if (val->type == META_VAL_STRING) {
        return val->s.len == tok_len && strncmp(val->s.str, tok, (size_t)tok_len) == 0;
    }
    if (val->type == META_VAL_ARRAY) {
        // For bare equality on arrays — check if tok matches any element
        for (int i = 0; i < val->arr.count; i++) {
            if ((int)strlen(val->arr.items[i]) == tok_len &&
                strncmp(val->arr.items[i], tok, (size_t)tok_len) == 0)
                return 1;
        }
    }
    return 0;
}

// Check prefix
static int meta_val_prefix(const MetaVal* val, const char* pfx, int pfx_len) {
    if (val->type == META_VAL_STRING) {
        return val->s.len >= pfx_len && strncmp(val->s.str, pfx, (size_t)pfx_len) == 0;
    }
    return 0;
}

// Check array contains
static int meta_val_contains(const MetaVal* val, const char* elem, int elem_len) {
    if (val->type == META_VAL_ARRAY) {
        for (int i = 0; i < val->arr.count; i++) {
            if ((int)strlen(val->arr.items[i]) == elem_len &&
                strncmp(val->arr.items[i], elem, (size_t)elem_len) == 0)
                return 1;
        }
    }
    return 0;
}

// Find a field by key in a MetaRecord
static MetaVal* find_field(MetaRecord* rec, const char* key, int key_len) {
    for (int i = 0; i < rec->count; i++) {
        if ((int)strlen(rec->fields[i].key) == key_len &&
            strncmp(rec->fields[i].key, key, (size_t)key_len) == 0)
            return &rec->fields[i].val;
    }
    return NULL;
}

// Evaluate a single filter condition on a data record.
// filter_key / filter_val come from parsing the filter expression.
// If filter_val is a sub-map string like "{$gte: 2}", we parse operator from it.
// Otherwise it's a bare exact-match.
static int eval_condition(Arena* arena, MetaRecord* data_rec,
                          const char* fkey, int fkey_len,
                          MetaVal* fval) {
    MetaVal* dval = find_field(data_rec, fkey, fkey_len);
    if (!dval) return 0; // field not present → doesn't match

    if (fval->type == META_VAL_STRING && fval->s.len > 0 && fval->s.str[0] == '{') {
        // Sub-map with operator: {$op: value}
        const char* p = fval->s.str + 1; // skip '{'
        p = skip_ws(p);
        const char* op_start; int op_len;
        p = read_token(p, &op_start, &op_len);
        p = skip_ws(p);
        if (*p == ':') p++;
        p = skip_ws(p);

        // Parse the operand value
        MetaVal operand;
        parse_value(arena, p, &operand);

        // Determine operand string for comparison
        const char* cmp_str = NULL; int cmp_len = 0;
        if (operand.type == META_VAL_STRING) { cmp_str = operand.s.str; cmp_len = operand.s.len; }
        else if (operand.type == META_VAL_INT) {
            // Convert back to string for comparison — but use numeric comparison
            char buf[64];
            int n = snprintf(buf, sizeof(buf), "%" PRId64, (int64_t)operand.i);
            cmp_str = arena_strdup(arena, buf, n);
            cmp_len = n;
        }

        if (op_len >= 4 && strncmp(op_start, "$gte", 4) == 0) {
            return meta_val_cmp(dval, cmp_str, cmp_len) >= 0;
        } else if (op_len >= 4 && strncmp(op_start, "$lte", 4) == 0) {
            return meta_val_cmp(dval, cmp_str, cmp_len) <= 0;
        } else if (op_len >= 3 && strncmp(op_start, "$ne", 3) == 0) {
            return !meta_val_eq(dval, cmp_str, cmp_len);
        } else if (op_len >= 7 && strncmp(op_start, "$prefix", 7) == 0) {
            return meta_val_prefix(dval, cmp_str, cmp_len);
        } else if (op_len >= 9 && strncmp(op_start, "$contains", 9) == 0) {
            return meta_val_contains(dval, cmp_str, cmp_len);
        }
        // Unknown operator — fail
        return 0;
    }

    // Bare value — exact match
    if (fval->type == META_VAL_INT) {
        char buf[64];
        int n = snprintf(buf, sizeof(buf), "%" PRId64, (int64_t)fval->i);
        return meta_val_eq(dval, buf, n);
    }
    if (fval->type == META_VAL_STRING) {
        return meta_val_eq(dval, fval->s.str, fval->s.len);
    }
    return 0;
}

// Evaluate a filter expression (parsed as MetaRecord of conditions) against a data record.
// $and / $or are handled specially.
static int eval_filter_record(Arena* arena, MetaRecord* data_rec, MetaRecord* filter_rec);

// Parse and evaluate a $and/$or array: [{cond1}, {cond2}, ...]
static int eval_logical_array(Arena* arena, MetaRecord* data_rec,
                              const char* array_body, int is_or) {
    // array_body looks like: "[{ts: {$gte: ...}}, {ts: {$lte: ...}}]"
    const char* p = array_body;
    p = skip_ws(p);
    if (*p == '[') p++;

    while (*p) {
        p = skip_ws(p);
        if (*p == ']' || *p == '\0') break;
        if (*p == '{') {
            const char* start = p + 1; // skip '{'
            int depth = 1;
            p++;
            while (*p && depth > 0) {
                if (*p == '{') depth++;
                else if (*p == '}') depth--;
                p++;
            }
            // 'start' to 'p-1' is the inner content
            int inner_len = (int)(p - start - 1);
            char* inner = arena_strdup(arena, start, inner_len);

            MetaRecord sub = MetaStore_Parse(arena, inner);
            int result = eval_filter_record(arena, data_rec, &sub);

            if (is_or && result) return 1;
            if (!is_or && !result) return 0;
        }
        p = skip_ws(p);
        if (*p == ',') p++;
    }

    return is_or ? 0 : 1; // $and: all passed; $or: none passed
}

static int eval_filter_record(Arena* arena, MetaRecord* data_rec, MetaRecord* filter_rec) {
    // All conditions must match (implicit AND)
    for (int i = 0; i < filter_rec->count; i++) {
        const char* key = filter_rec->fields[i].key;
        int key_len = (int)strlen(key);
        MetaVal* fval = &filter_rec->fields[i].val;

        // Check for $and / $or
        if (key_len == 4 && strncmp(key, "$and", 4) == 0) {
            if (fval->type == META_VAL_STRING) {
                if (!eval_logical_array(arena, data_rec, fval->s.str, 0)) return 0;
            }
            continue;
        }
        if (key_len == 3 && strncmp(key, "$or", 3) == 0) {
            if (fval->type == META_VAL_STRING) {
                if (!eval_logical_array(arena, data_rec, fval->s.str, 1)) return 0;
            }
            continue;
        }

        if (!eval_condition(arena, data_rec, key, key_len, fval)) return 0;
    }
    return 1;
}

void MetaStore_Filter(MetaStore* ms, const char* filter_expr,
                      u8* out_mask, int mask_len) {
    // Initialize all to 1 (pass) — callers AND into existing mask
    int n = mask_len < ms->count ? mask_len : ms->count;
    for (int i = 0; i < mask_len; i++) out_mask[i] = 0;

    // Scratch arena — we need temp allocations for parsing.
    // We'll use the main arena (it only grows, never freed mid-run).
    Arena* arena = ms->arena;
    size_t saved_used = arena->used;

    // Parse the filter expression itself
    MetaRecord filter = MetaStore_Parse(arena, filter_expr);

    for (int i = 0; i < n; i++) {
        if (!ms->raw[i]) { out_mask[i] = 0; continue; }

        // Parse the data record
        MetaRecord data = MetaStore_Parse(arena, ms->raw[i]);
        out_mask[i] = eval_filter_record(arena, &data, &filter) ? 1 : 0;
    }

    // Restore arena (free scratch allocations)
    arena->used = saved_used;
}
