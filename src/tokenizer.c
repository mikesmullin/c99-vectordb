#include "app_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int id;
    char* str;
} TokenIndex;

typedef struct {
    char** vocab;
    f32* vocab_scores;
    TokenIndex* sorted_vocab;
    int vocab_size;
    int max_token_length;
} Tokenizer;

int compare_token_index(const void* a, const void* b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void Tokenizer__Init(Tokenizer* t, const char* filename, int vocab_size, Arena* arena) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open tokenizer file: %s\n", filename);
        __builtin_trap();
    }

    t->vocab_size = vocab_size;
    t->vocab = (char**)Arena__Push(arena, vocab_size * sizeof(char*));
    t->vocab_scores = (f32*)Arena__Push(arena, vocab_size * sizeof(f32));
    t->sorted_vocab = (TokenIndex*)Arena__Push(arena, vocab_size * sizeof(TokenIndex));

    // Read max_token_length
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Failed to read max_token_length\n");
        __builtin_trap();
    }

    for (int i = 0; i < vocab_size; i++) {
        if (fread(&t->vocab_scores[i], sizeof(f32), 1, file) != 1) {
            fprintf(stderr, "Failed to read vocab score %d\n", i);
            __builtin_trap();
        }
        
        int len;
        if (fread(&len, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "Failed to read vocab len %d\n", i);
            __builtin_trap();
        }
        
        t->vocab[i] = (char*)Arena__Push(arena, len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) {
            fprintf(stderr, "Failed to read vocab string %d\n", i);
            __builtin_trap();
        }
        t->vocab[i][len] = '\0';
        
        t->sorted_vocab[i].id = i;
        t->sorted_vocab[i].str = t->vocab[i];
    }
    
    qsort(t->sorted_vocab, vocab_size, sizeof(TokenIndex), compare_token_index);
    
    fclose(file);
    MEMO_VLOG("Tokenizer Initialized. Vocab Size: %d\n", vocab_size);
}

int Tokenizer__FindToken(Tokenizer* t, const char* str) {
    TokenIndex key;
    key.str = (char*)str;
    TokenIndex* res = bsearch(&key, t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_token_index);
    if (res) return res->id;
    return -1;
}

void Tokenizer__Encode(Tokenizer* t, const char* text, int* tokens, int* n_tokens) {
    // 1. Encode every character as a token
    int n = 0;
    for (const char* c = text; *c != '\0'; c++) {
        char buf[2] = {*c, '\0'};
        int id = Tokenizer__FindToken(t, buf);
        if (id != -1) {
            tokens[n++] = id;
        }
    }
    
    // 2. Merge
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;
        
        for (int i = 0; i < n - 1; i++) {
            // form pair
            char buf[512];
            snprintf(buf, sizeof(buf), "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            
            int id = Tokenizer__FindToken(t, buf);
            if (id != -1) {
                if (t->vocab_scores[id] > best_score) {
                    best_score = t->vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }
            }
        }
        
        if (best_idx == -1) break; // no more merges
        
        // merge
        tokens[best_idx] = best_id;
        // shift
        for (int i = best_idx + 1; i < n - 1; i++) {
            tokens[i] = tokens[i+1];
        }
        n--;
    }
    *n_tokens = n;
}


const char* Tokenizer__Decode(Tokenizer* t, int token, int prev_token) {
    (void)prev_token; // Handle spacing later
    if (token < 0 || token >= t->vocab_size) return "";
    return t->vocab[token];
}
