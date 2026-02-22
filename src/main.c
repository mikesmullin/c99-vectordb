#include "vectordb.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <limits.h>
#include <errno.h>
#include <sys/stat.h>
#include <inttypes.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

// Arena Declarations (implemented in memory.c)
void Arena__Init(Arena* arena, size_t size);
void* Arena__Push(Arena* arena, size_t size);
void Arena__Free(Arena* arena);

// LLM Includes
#include "vulkan_ctx.c"
#include "llm_model.c"
#include "tokenizer.c"
#include "inference.c"

#define DIM 768 // TinyLlama dim

int g_memo_verbose = 0;

// Global LLM State
Config llm_config;
TransformerWeights llm_weights;
RunState llm_state;
LLM_VulkanCtx llm_vk_ctx;
Tokenizer tokenizer;
int llm_initialized = 0;

static int dir_has_workspace_layout(const char* dir) {
    char path_a[PATH_MAX];
    char path_b[PATH_MAX];
    char path_c[PATH_MAX];
    char path_d[PATH_MAX];

    snprintf(path_a, sizeof(path_a), "%s/models/stories110M.bin", dir);
    snprintf(path_b, sizeof(path_b), "%s/models/tokenizer.bin", dir);
    snprintf(path_c, sizeof(path_c), "%s/build/memo_search.spv", dir);
    snprintf(path_d, sizeof(path_d), "%s/build/headless.spv", dir);

    return access(path_a, F_OK) == 0 &&
           access(path_b, F_OK) == 0 &&
           access(path_c, F_OK) == 0 &&
           access(path_d, F_OK) == 0;
}

static int find_workspace_root_from(const char* start_path, char* out_root, size_t out_size) {
    char current[PATH_MAX];
    if (!realpath(start_path, current)) {
        return 0;
    }

    while (1) {
        if (dir_has_workspace_layout(current)) {
            strncpy(out_root, current, out_size - 1);
            out_root[out_size - 1] = '\0';
            return 1;
        }

        char* slash = strrchr(current, '/');
        if (!slash) return 0;
        if (slash == current) {
            current[1] = '\0';
            if (dir_has_workspace_layout(current)) {
                strncpy(out_root, current, out_size - 1);
                out_root[out_size - 1] = '\0';
                return 1;
            }
            return 0;
        }
        *slash = '\0';
    }
}

static int Runtime__ChdirToWorkspaceRoot(void) {
    char root[PATH_MAX];
    if (find_workspace_root_from(".", root, sizeof(root))) {
        return chdir(root) == 0;
    }

    char exe_path[PATH_MAX];
#ifdef __APPLE__
    uint32_t size = (uint32_t)sizeof(exe_path);
    if (_NSGetExecutablePath(exe_path, &size) != 0) {
        return 0;
    }
#else
    ssize_t n = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (n <= 0 || n >= (ssize_t)sizeof(exe_path)) {
        return 0;
    }
    exe_path[n] = '\0';
#endif

    char resolved[PATH_MAX];
    if (!realpath(exe_path, resolved)) {
        strncpy(resolved, exe_path, sizeof(resolved) - 1);
        resolved[sizeof(resolved) - 1] = '\0';
    }

    char* slash = strrchr(resolved, '/');
    if (!slash) return 0;
    *slash = '\0';

    if (!find_workspace_root_from(resolved, root, sizeof(root))) {
        return 0;
    }
    return chdir(root) == 0;
}

static int file_exists(const char* path) {
    return access(path, F_OK) == 0;
}

static int ensure_llm_assets_present(void) {
    if (!file_exists("models/stories110M.bin")) {
        fprintf(stderr, "Error: missing model file models/stories110M.bin\n");
        fprintf(stderr, "Hint: run ./models/download.sh from the project root.\n");
        return 0;
    }
    if (!file_exists("models/tokenizer.bin")) {
        fprintf(stderr, "Error: missing tokenizer file models/tokenizer.bin\n");
        fprintf(stderr, "Hint: run ./models/download.sh from the project root.\n");
        return 0;
    }
    return 1;
}

void Init_LLM(Arena* arena) {
    if (llm_initialized) return;
    MEMO_VLOG("Initializing TinyLlama 110M...\n");
    LLM__Load("models/stories110M.bin", &llm_config, &llm_weights, arena);
    LLM_VulkanCtx__Init(&llm_vk_ctx);
    LLM_VulkanCtx__SetupPipeline(&llm_vk_ctx);
    LLM_VulkanCtx__UploadWeights(&llm_vk_ctx, arena->base, arena->used);
    LLM_VulkanCtx__PrepareBuffers(&llm_vk_ctx, 1024 * 1024, 1024 * 1024);
    LLM__InitState(&llm_state, &llm_config, arena);
    Tokenizer__Init(&tokenizer, "models/tokenizer.bin", llm_config.vocab_size, arena);
    llm_initialized = 1;
    MEMO_VLOG("LLM Initialized.\n");
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

static int TextStore_Set(TextStore* ts, int id, const char* text) {
    if (id < 0 || id >= ts->count) return 0;
    int len = (int)strlen(text);
    char* str = (char*)Arena__Push(ts->arena, (size_t)len + 1);
    strcpy(str, text);
    ts->lines[id] = str;
    return 1;
}

static int has_path_separator(const char* s) {
    return (s && strchr(s, '/')) ? 1 : 0;
}

static int ensure_db_dir(void) {
    struct stat st;
    if (stat("db", &st) == 0) {
        if (S_ISDIR(st.st_mode)) return 1;
        fprintf(stderr, "Error: db exists but is not a directory\n");
        return 0;
    }
    if (mkdir("db", 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "Error: failed to create db directory: %s\n", strerror(errno));
        return 0;
    }
    return 1;
}

static int ensure_parent_dir_for_file(const char* file_path) {
    if (!file_path) return 0;

    char dir[PATH_MAX];
    strncpy(dir, file_path, sizeof(dir) - 1);
    dir[sizeof(dir) - 1] = '\0';

    char* slash = strrchr(dir, '/');
    if (!slash) return 1;
    if (slash == dir) {
        return 1;
    }
    *slash = '\0';

    for (char* p = dir + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
                fprintf(stderr, "Error: failed to create directory %s: %s\n", dir, strerror(errno));
                return 0;
            }
            *p = '/';
        }
    }

    if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "Error: failed to create directory %s: %s\n", dir, strerror(errno));
        return 0;
    }
    return 1;
}

static void build_db_paths(const char* base, char* index_path, size_t index_size, char* txt_path, size_t txt_size) {
    if (has_path_separator(base)) {
        snprintf(index_path, index_size, "%s.memo", base);
        snprintf(txt_path, txt_size, "%s.txt", base);
    } else {
        snprintf(index_path, index_size, "db/%s.memo", base);
        snprintf(txt_path, txt_size, "db/%s.txt", base);
    }
}

static char* join_args(int start, int argc, char** argv) {
    size_t len = 0;
    for (int i = start; i < argc; i++) {
        len += strlen(argv[i]);
        if (i + 1 < argc) len += 1;
    }
    char* out = (char*)malloc(len + 1);
    if (!out) return NULL;
    out[0] = '\0';
    for (int i = start; i < argc; i++) {
        strcat(out, argv[i]);
        if (i + 1 < argc) strcat(out, " ");
    }
    return out;
}

static int is_integer(const char* s) {
    if (!s || !*s) return 0;
    if (*s == '+' || *s == '-') s++;
    if (!*s) return 0;
    while (*s) {
        if (*s < '0' || *s > '9') return 0;
        s++;
    }
    return 1;
}

static void print_help(void) {
    printf("Usage:\n");
    printf("  memo [--help] [-v] [-f <file>]\n");
    printf("  memo save [-f <file>] [-v] [<id>] <note>\n");
    printf("  memo recall [-f <file>] [-v] [-k <N>] <query>\n");
    printf("  memo clear [-f <file>] [-v]\n\n");
    printf("Options:\n");
    printf("  [-f <file>] Optional DB basename (default: db/memo)\n");
    printf("  -v          Verbose logs to stderr\n");
    printf("  --help      Show this help\n");
}

static int load_state(VDB_Context* ctx, TextStore* ts, const char* db_base, VDB_Index** out_index) {
    char index_path[PATH_MAX];
    char txt_path[PATH_MAX];
    build_db_paths(db_base, index_path, sizeof(index_path), txt_path, sizeof(txt_path));

    VDB_Index* index = VDB_Index_Create(ctx, DIM, VDB_METRIC_COSINE, 10000);
    if (access(index_path, F_OK) == 0) {
        VDB_Index* loaded = VDB_Index_Load(ctx, index_path);
        if (loaded) {
            index = loaded;
            ts->count = 0;
            if (access(txt_path, F_OK) == 0) {
                TextStore_Load(ts, txt_path);
            }
        }
    }

    *out_index = index;
    return 1;
}

static int save_state(VDB_Index* index, TextStore* ts, const char* db_base) {
    if (!has_path_separator(db_base) && !ensure_db_dir()) {
        return 0;
    }
    char index_path[PATH_MAX];
    char txt_path[PATH_MAX];
    build_db_paths(db_base, index_path, sizeof(index_path), txt_path, sizeof(txt_path));
    if (!ensure_parent_dir_for_file(index_path) || !ensure_parent_dir_for_file(txt_path)) {
        return 0;
    }
    VDB_Index_Save(index, index_path);
    TextStore_Save(ts, txt_path);
    return 1;
}

static int clear_state(const char* db_base) {
    char index_path[PATH_MAX];
    char txt_path[PATH_MAX];
    int removed_any = 0;

    build_db_paths(db_base, index_path, sizeof(index_path), txt_path, sizeof(txt_path));

    if (unlink(index_path) == 0) removed_any = 1;
    else if (errno != ENOENT) {
        fprintf(stderr, "Error: failed to remove %s: %s\n", index_path, strerror(errno));
        return 0;
    }

    if (unlink(txt_path) == 0) removed_any = 1;
    else if (errno != ENOENT) {
        fprintf(stderr, "Error: failed to remove %s: %s\n", txt_path, strerror(errno));
        return 0;
    }

    if (removed_any) {
        printf("Cleared memory database (%s, %s)\n", index_path, txt_path);
    } else {
        printf("Database already empty (%s, %s)\n", index_path, txt_path);
    }
    return 1;
}



int main(int argc, char** argv) {
    const char* db_base = "memo";
    char* positional[64];
    int positional_count = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            g_memo_verbose = 1;
            continue;
        }
        if (strcmp(argv[i], "-f") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: -f requires a value\n");
                return 1;
            }
            db_base = argv[++i];
            continue;
        }
        positional[positional_count++] = argv[i];
    }

    if (positional_count == 0 || strcmp(positional[0], "--help") == 0 || strcmp(positional[0], "help") == 0) {
        print_help();
        return 0;
    }

    if (!Runtime__ChdirToWorkspaceRoot()) {
        MEMO_VLOG("Warning: Failed to auto-locate workspace root; using current directory.\n");
    }

    const char* command = positional[0];
    if (strcmp(command, "clear") == 0) {
        if (positional_count != 1) {
            fprintf(stderr, "Error: clear does not accept extra arguments\n");
            return 1;
        }
        return clear_state(db_base) ? 0 : 1;
    }

    if (strcmp(command, "save") == 0 || strcmp(command, "recall") == 0) {
        if (!ensure_llm_assets_present()) {
            return 1;
        }
    }

    Arena arena;
    Arena__Init(&arena, 1024ULL * 1024ULL * 1024ULL);

    Init_LLM(&arena);

    VDB_Context ctx;
    VDB_Init(&ctx, &arena);

    TextStore ts;
    TextStore_Init(&ts, 10000, &arena);

    VDB_Index* index = NULL;
    load_state(&ctx, &ts, db_base, &index);

    if (strcmp(command, "save") == 0) {
        if (positional_count < 2) {
            fprintf(stderr, "Error: save requires <note> or [<id>] <note>\n");
            Arena__Free(&arena);
            return 1;
        }

        int note_start = 1;
        int override_id = -1;
        if (positional_count >= 3 && is_integer(positional[1])) {
            override_id = (int)strtol(positional[1], NULL, 10);
            note_start = 2;
        }

        char* note = join_args(note_start, positional_count, positional);
        if (!note || note[0] == '\0') {
            fprintf(stderr, "Error: save requires non-empty note text\n");
            free(note);
            Arena__Free(&arena);
            return 1;
        }

        f32 vec[DIM];
        embed_text_llm(note, vec, DIM, &arena);

        u64 id;
        if (override_id >= 0) {
            if (override_id >= ts.count || override_id >= index->count) {
                fprintf(stderr, "Error: override id %d does not exist\n", override_id);
                free(note);
                Arena__Free(&arena);
                return 1;
            }
            TextStore_Set(&ts, override_id, note);
            for (int j = 0; j < DIM; j++) {
                index->vectors[override_id * DIM + j] = vec[j];
            }
            id = (u64)override_id;
        } else {
            id = TextStore_Add(&ts, note);
            if (id == (u64)-1) {
                fprintf(stderr, "Error: text store is full\n");
                free(note);
                Arena__Free(&arena);
                return 1;
            }
            VDB_Index_Add(index, id, vec);
        }

        if (!save_state(index, &ts, db_base)) {
            free(note);
            Arena__Free(&arena);
            return 1;
        }
        printf("Memorized: '%s' (ID: %" PRIu64 ")\n", note, id);
        free(note);
    } else if (strcmp(command, "recall") == 0) {
        int k = 2;
        int query_start = 1;
        if (positional_count >= 4 && strcmp(positional[1], "-k") == 0) {
            if (!is_integer(positional[2])) {
                fprintf(stderr, "Error: -k requires an integer\n");
                Arena__Free(&arena);
                return 1;
            }
            k = (int)strtol(positional[2], NULL, 10);
            query_start = 3;
        }

        if (query_start >= positional_count) {
            fprintf(stderr, "Error: recall requires <query>\n");
            Arena__Free(&arena);
            return 1;
        }
        if (k < 1) k = 1;
        if (k > 100) k = 100;

        char* query = join_args(query_start, positional_count, positional);
        if (!query || query[0] == '\0') {
            fprintf(stderr, "Error: recall requires non-empty query\n");
            free(query);
            Arena__Free(&arena);
            return 1;
        }

        printf("Top %d results for '%s':\n", k, query);
        if (index->count > 0) {
            f32 vec[DIM];
            embed_text_llm(query, vec, DIM, &arena);

            VDB_Result results[100];
            VDB_Search(index, vec, k, results);
            for (int i = 0; i < k; i++) {
                if (results[i].score < -0.9f) continue;
                u64 id = results[i].id;
                if (id < (u64)ts.count) {
                    printf("  [%d] Score: %.4f | %s\n", i + 1, results[i].score, ts.lines[id]);
                }
            }
        }
        free(query);
    } else {
        fprintf(stderr, "Error: unknown command '%s'\n", command);
        print_help();
        Arena__Free(&arena);
        return 1;
    }

    Arena__Free(&arena);
    return 0;
}
