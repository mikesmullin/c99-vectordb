#include "vectordb.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define MAX_CMD_LEN 1024
#define DIM 768 // TinyLlama dim

// Simple Text Store
typedef struct {
    char** lines;
    int count;
    int capacity;
    Arena* arena;
} TextStore;

void TextStore_Init(TextStore* ts, int capacity, Arena* arena) {
    ts->lines = (char**)Arena__Push(arena, capacity * sizeof(char*));
    ts->count = 0;
    ts->capacity = capacity;
    ts->arena = arena;
}

u64 TextStore_Add(TextStore* ts, const char* text) {
    if (ts->count >= ts->capacity) return -1;
    
    int len = strlen(text);
    char* str = (char*)Arena__Push(ts->arena, len + 1);
    strcpy(str, text);
    
    ts->lines[ts->count] = str;
    return ts->count++; // ID is the index
}

void TextStore_Save(TextStore* ts, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return;
    fwrite(&ts->count, sizeof(int), 1, f);
    for (int i = 0; i < ts->count; i++) {
        int len = strlen(ts->lines[i]);
        fwrite(&len, sizeof(int), 1, f);
        fwrite(ts->lines[i], 1, len, f);
    }
    fclose(f);
}

void TextStore_Load(TextStore* ts, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return;
    int count;
    if (fread(&count, sizeof(int), 1, f) != 1) return;
    
    for (int i = 0; i < count; i++) {
        int len;
        fread(&len, sizeof(int), 1, f);
        char* str = (char*)Arena__Push(ts->arena, len + 1);
        fread(str, 1, len, f);
        str[len] = '\0';
        ts->lines[ts->count++] = str;
    }
    fclose(f);
}

// Dummy Embedding (Random)
void embed_text(const char* text, f32* out_vec, int dim) {
    // Deterministic-ish random based on text content to make it slightly consistent?
    // No, just random for now.
    // Actually, let's make it deterministic so "recall" works if we type the same thing.
    u32 seed = 0;
    for (const char* c = text; *c; c++) seed = seed * 31 + *c;
    srand(seed);
    
    float norm = 0;
    for (int i = 0; i < dim; i++) {
        out_vec[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        norm += out_vec[i] * out_vec[i];
    }
    // Normalize
    norm = sqrtf(norm);
    for (int i = 0; i < dim; i++) out_vec[i] /= norm;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    
    printf("C99 VectorDB CLI\n");
    printf("Commands: load <file>, save <file>, memo <text>, recall <k> <text>, exit\n");
    
    // Init
    Arena arena;
    Arena__Init(&arena, 1024 * 1024 * 256); // 256MB
    
    VDB_Context ctx;
    VDB_Init(&ctx, VDB_BACKEND_CPU, &arena);
    
    VDB_Index* index = VDB_Index_Create(&ctx, DIM, VDB_METRIC_COSINE, 10000);
    
    TextStore ts;
    TextStore_Init(&ts, 10000, &arena);
    
    char cmd_buf[MAX_CMD_LEN];
    
    while (1) {
        printf("> ");
        if (!fgets(cmd_buf, sizeof(cmd_buf), stdin)) break;
        
        // Remove newline
        cmd_buf[strcspn(cmd_buf, "\n")] = 0;
        
        if (strncmp(cmd_buf, "exit", 4) == 0) break;
        
        if (strncmp(cmd_buf, "memo ", 5) == 0) {
            char* text = cmd_buf + 5;
            f32 vec[DIM];
            embed_text(text, vec, DIM);
            
            u64 id = TextStore_Add(&ts, text);
            VDB_Index_Add(index, id, vec);
            printf("Memorized: '%s' (ID: %lu)\n", text, id);
        }
        else if (strncmp(cmd_buf, "recall ", 7) == 0) {
            // Parse k
            char* ptr = cmd_buf + 7;
            int k = strtol(ptr, &ptr, 10);
            while (*ptr == ' ') ptr++; // skip spaces
            char* text = ptr;
            
            f32 vec[DIM];
            embed_text(text, vec, DIM);
            
            VDB_Result results[100]; // max k=100
            if (k > 100) k = 100;
            
            VDB_Search(index, vec, k, results);
            
            printf("Top %d results for '%s':\n", k, text);
            for (int i = 0; i < k; i++) {
                if (results[i].score < -0.9f) continue; // Empty
                u64 id = results[i].id;
                if (id < (u64)ts.count) {
                    printf("  [%d] Score: %.4f | %s\n", i+1, results[i].score, ts.lines[id]);
                }
            }
        }
        else if (strncmp(cmd_buf, "save ", 5) == 0) {
            char* fname = cmd_buf + 5;
            char vdb_fname[256]; snprintf(vdb_fname, 256, "%s.vdb", fname);
            char txt_fname[256]; snprintf(txt_fname, 256, "%s.txt", fname);
            
            VDB_Index_Save(index, vdb_fname);
            TextStore_Save(&ts, txt_fname);
        }
        else if (strncmp(cmd_buf, "load ", 5) == 0) {
            char* fname = cmd_buf + 5;
            char vdb_fname[256]; snprintf(vdb_fname, 256, "%s.vdb", fname);
            char txt_fname[256]; snprintf(txt_fname, 256, "%s.txt", fname);
            
            // Reset arena? No, just load new ones. 
            // For prototype, let's just overwrite pointers (leak old memory in arena, it's fine)
            index = VDB_Index_Load(&ctx, vdb_fname);
            if (index) {
                // Re-init text store
                ts.count = 0; // Reset count
                TextStore_Load(&ts, txt_fname);
            } else {
                // Restore if failed
                printf("Failed to load.\n");
                // Re-create empty
                index = VDB_Index_Create(&ctx, DIM, VDB_METRIC_COSINE, 10000);
            }
        }
    }
    
    Arena__Free(&arena);
    return 0;
}
