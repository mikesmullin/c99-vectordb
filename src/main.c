#include "vectordb.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <limits.h>

// Arena Declarations (implemented in memory.c)
void Arena__Init(Arena* arena, size_t size);
void* Arena__Push(Arena* arena, size_t size);
void Arena__Free(Arena* arena);

// LLM Includes
#include "vulkan_ctx.c"
#include "llm_model.c"
#include "tokenizer.c"
#include "inference.c"

#define MAX_CMD_LEN 1024
#define DIM 768 // TinyLlama dim

// Global LLM State
Config llm_config;
TransformerWeights llm_weights;
RunState llm_state;
LLM_VulkanCtx llm_vk_ctx;
Tokenizer tokenizer;
int llm_initialized = 0;

static int Runtime__ChdirToWorkspaceRoot(void) {
    char exe_path[PATH_MAX];
    ssize_t n = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (n <= 0 || n >= (ssize_t)sizeof(exe_path)) {
        return 0;
    }
    exe_path[n] = '\0';

    char resolved[PATH_MAX];
    if (!realpath(exe_path, resolved)) {
        strncpy(resolved, exe_path, sizeof(resolved) - 1);
        resolved[sizeof(resolved) - 1] = '\0';
    }

    char* slash = strrchr(resolved, '/');
    if (!slash) return 0;
    *slash = '\0';

    slash = strrchr(resolved, '/');
    if (!slash) return 0;
    *slash = '\0';

    if (chdir(resolved) != 0) {
        return 0;
    }
    return 1;
}

void Init_LLM(Arena* arena) {
    if (llm_initialized) return;
    printf("Initializing TinyLlama 110M...\n");
    LLM__Load("models/stories110M.bin", &llm_config, &llm_weights, arena);
    LLM_VulkanCtx__Init(&llm_vk_ctx);
    LLM_VulkanCtx__SetupPipeline(&llm_vk_ctx);
    LLM_VulkanCtx__UploadWeights(&llm_vk_ctx, arena->base, arena->used);
    LLM_VulkanCtx__PrepareBuffers(&llm_vk_ctx, 1024 * 1024, 1024 * 1024);
    LLM__InitState(&llm_state, &llm_config, arena);
    Tokenizer__Init(&tokenizer, "models/tokenizer.bin", llm_config.vocab_size, arena);
    llm_initialized = 1;
    printf("LLM Initialized.\n");
}

void embed_text_llm(const char* text, f32* out_vec, int dim, Arena* arena) {
    if (!llm_initialized) { fprintf(stderr, "LLM not initialized!\n"); return; }
    int* tokens = (int*)malloc((strlen(text) + 10) * 2 * sizeof(int)); 
    int n_tokens;
    Tokenizer__Encode(&tokenizer, text, tokens, &n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        LLM__Forward(&llm_state, &llm_config, &llm_weights, tokens[i], i, &llm_vk_ctx, arena);
    }
    f32 norm = 0.0f;
    for (int i = 0; i < dim; i++) norm += llm_state.x[i] * llm_state.x[i];
    norm = sqrtf(norm);
    if (norm > 1e-5f) { for (int i = 0; i < dim; i++) out_vec[i] = llm_state.x[i] / norm; }
    else { memset(out_vec, 0, dim * sizeof(f32)); }
    free(tokens);
}

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



int main(int argc, char** argv) {
    (void)argc; (void)argv;

    if (!Runtime__ChdirToWorkspaceRoot()) {
        fprintf(stderr, "Warning: Failed to auto-locate workspace root; using current directory.\n");
    }
    
    printf("C99 VectorDB CLI (TinyLlama 110M Embeddings)\n");
    printf("Commands: load <file>, save <file>, memo <text>, recall <k> <text>, exit\n");
    
    // Init
    Arena arena;
    Arena__Init(&arena, 1024 * 1024 * 1024); // 1GB
    
    // Init LLM
    Init_LLM(&arena);
    
    VDB_Context ctx;
    VDB_Init(&ctx, &arena);
    
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
            embed_text_llm(text, vec, DIM, &arena);
            
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
            embed_text_llm(text, vec, DIM, &arena);
            
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
