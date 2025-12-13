#include "app_types.h"
#include <math.h>
#include <string.h>
#include <stdint.h>

/*
 * Hybrid CPU/GPU Inference Implementation
 * 
 * This architecture offloads the heavy lifting (Matrix Multiplications) to the GPU via Vulkan,
 * while keeping lightweight element-wise operations (RMSNorm, Softmax, RoPE, SwiGLU) on the CPU.
 * 
 * Rationale:
 * 1. MatMul is O(N^2) or O(N^3) and benefits massively from GPU parallelism.
 * 2. Norm/Activation are O(N). For batch_size=1 (inference), the PCIe transfer latency and 
 *    kernel launch overhead would outweigh the compute speedup if we moved these to GPU.
 *    It is faster to keep the vector in CPU cache, modify it, and send it back for the next MatMul.
 */

// Kept on CPU: O(N) operation. Faster to execute locally than to incur GPU synchronization/transfer overhead.
void rmsnorm(f32* o, f32* x, f32* weight, int size) {
    // calculate sum of squares
    f32 ss = 0.0f;
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int i = 0; i < size; i++) {
        o[i] = weight[i] * (ss * x[i]);
    }
}

// Kept on CPU: Requires global reduction (max/sum). Implementing efficient reductions in shaders 
// is complex, and for vocab_size (~32k), the CPU handles this in microseconds.
void softmax(f32* x, int size) {
    // find max value (for numerical stability)
    f32 max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    f32 sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Helper macro for weight offset calculation
#define W_OFFSET(ptr) ((u32)(((uintptr_t)(ptr) - (uintptr_t)arena->base) / sizeof(f32)))

void LLM__Forward(RunState* s, Config* p, TransformerWeights* w, int token, int pos, LLM_VulkanCtx* vk_ctx, Arena* arena) {
    f32* x = s->x;
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    int kv_dim = (dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;

    // 1. Embedding
    f32* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(f32));

    // 2. Layers
    for (int l = 0; l < p->n_layers; l++) {
        // Attention RMSNorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // QKV Matmuls
        // wq: (dim, dim) @ xb -> q
        LLM_VulkanCtx__MatMul(vk_ctx, s->q, s->xb, W_OFFSET(w->wq + l * dim * dim), dim, dim);
        // wk: (kv_dim, dim) @ xb -> k
        LLM_VulkanCtx__MatMul(vk_ctx, s->k, s->xb, W_OFFSET(w->wk + l * kv_dim * dim), dim, kv_dim);
        // wv: (kv_dim, dim) @ xb -> v
        LLM_VulkanCtx__MatMul(vk_ctx, s->v, s->xb, W_OFFSET(w->wv + l * kv_dim * dim), dim, kv_dim);

        // RoPE (Rotary Positional Embeddings)
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            f32 freq = 1.0f / powf(10000.0f, head_dim / (f32)head_size);
            f32 val = pos * freq;
            f32 fcr = cosf(val);
            f32 fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? q, k
            for (int v = 0; v < rotn; v++) {
                f32* vec = v == 0 ? s->q : s->k;
                f32 v0 = vec[i];
                f32 v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // KV Cache update
        int loff = l * p->seq_len * kv_dim; // layer offset
        f32* key_cache_row = s->key_cache + loff + pos * kv_dim;
        f32* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(f32));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(f32));

        // Multihead Attention
        for (int h = 0; h < p->n_heads; h++) {
            // get query vector for this head
            f32* q = s->q + h * head_size;
            // attention scores
            f32* att = s->att + h * p->seq_len;
            
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get key vector for this head and timestep
                f32* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate score
                f32 score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            // softmax
            softmax(att, pos + 1);

            // weighted sum of values
            f32* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(f32));
            for (int t = 0; t <= pos; t++) {
                f32* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                f32 a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // Output projection
        // wo: (dim, dim) @ xb -> xb2
        LLM_VulkanCtx__MatMul(vk_ctx, s->xb2, s->xb, W_OFFSET(w->wo + l * dim * dim), dim, dim);

        // Residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // FFN RMSNorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // FFN
        // w1: (hidden_dim, dim) @ xb -> hb
        LLM_VulkanCtx__MatMul(vk_ctx, s->hb, s->xb, W_OFFSET(w->w1 + l * dim * hidden_dim), dim, hidden_dim);
        // w3: (hidden_dim, dim) @ xb -> hb2
        LLM_VulkanCtx__MatMul(vk_ctx, s->hb2, s->xb, W_OFFSET(w->w3 + l * dim * hidden_dim), dim, hidden_dim);

        // SwiGLU
        for (int i = 0; i < hidden_dim; i++) {
            f32 val = s->hb[i];
            // silu(x) = x * sigmoid(x)
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3 output
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // w2: (dim, hidden_dim) @ hb -> xb
        LLM_VulkanCtx__MatMul(vk_ctx, s->xb, s->hb, W_OFFSET(w->w2 + l * hidden_dim * dim), hidden_dim, dim);

        // Residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // Final RMSNorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // Classifier
    LLM_VulkanCtx__MatMul(vk_ctx, s->logits, x, W_OFFSET(w->wcls), dim, p->vocab_size);
}

int LLM__Sample(RunState* s, Config* p) {
    // Greedy argmax
    int max_i = 0;
    f32 max_p = s->logits[0];
    for (int i = 1; i < p->vocab_size; i++) {
        if (s->logits[i] > max_p) {
            max_i = i;
            max_p = s->logits[i];
        }
    }
    return max_i;
}
