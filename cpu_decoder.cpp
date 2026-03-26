#include "cpu_decoder.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ============================================================================
// RunState 分配/释放
// ============================================================================

void alloc_run_state(RunState& s, const Config& config) {
  int dim = config.dim;
  int kv_dim = (config.dim / config.n_heads) * config.n_kv_heads;
  int n_layers = config.n_layers;
  int n_heads = config.n_heads;
  int seq_len = config.seq_len;

  s.x = new float[dim];
  s.xb = new float[dim];
  s.xb2 = new float[dim];
  s.q = new float[dim];
  s.k = new float[kv_dim];
  s.v = new float[kv_dim];
  s.att = new float[n_heads * seq_len];
  s.hb = new float[config.hidden_dim];
  s.hb2 = new float[config.hidden_dim];
  s.logits = new float[abs(config.vocab_size)];
  s.k_cache = new float[n_layers * seq_len * kv_dim];
  s.v_cache = new float[n_layers * seq_len * kv_dim];
}

void free_run_state(RunState& s) {
  delete[] s.x;
  delete[] s.xb;
  delete[] s.xb2;
  delete[] s.q;
  delete[] s.k;
  delete[] s.v;
  delete[] s.att;
  delete[] s.hb;
  delete[] s.hb2;
  delete[] s.logits;
  delete[] s.k_cache;
  delete[] s.v_cache;
}

// ============================================================================
// 计算函数
// ============================================================================

void rmsnorm(float* out, const float* x, const float* weight, int dim) {
  float ss = 0.0f;
  for (int i = 0; i < dim; i++) ss += x[i] * x[i];
  ss = 1.0f / sqrtf(ss / dim + 1e-6f);
  for (int i = 0; i < dim; i++) out[i] = x[i] * ss * weight[i];
}

void matmul(float* out, const float* x, const float* w, int n, int d) {
  for (int i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) val += x[j] * w[i * n + j];
    out[i] = val;
  }
}

void rope(float* q, float* k, const float* freq_real, const float* freq_imag,
          int pos, int dim, int kv_dim, int head_dim) {
  for (int i = 0; i < dim; i += 2) {
    int head_pos = i % head_dim;
    float cos_val = freq_real[pos * head_dim / 2 + head_pos / 2];
    float sin_val = freq_imag[pos * head_dim / 2 + head_pos / 2];
    float q0 = q[i], q1 = q[i + 1];
    q[i] = q0 * cos_val - q1 * sin_val;
    q[i + 1] = q0 * sin_val + q1 * cos_val;
  }
  for (int i = 0; i < kv_dim; i += 2) {
    int head_pos = i % head_dim;
    float cos_val = freq_real[pos * head_dim / 2 + head_pos / 2];
    float sin_val = freq_imag[pos * head_dim / 2 + head_pos / 2];
    float k0 = k[i], k1 = k[i + 1];
    k[i] = k0 * cos_val - k1 * sin_val;
    k[i + 1] = k0 * sin_val + k1 * cos_val;
  }
}

void attention_cpu(RunState& s, const Config& config, int layer, int pos,
                   int kv_dim, int head_dim) {
  int kv_mul = config.n_heads / config.n_kv_heads;
  float scale = 1.0f / sqrtf((float)head_dim);
  float* k_cache_layer = s.k_cache + layer * config.seq_len * kv_dim;
  float* v_cache_layer = s.v_cache + layer * config.seq_len * kv_dim;

  for (int h = 0; h < config.n_heads; h++) {
    float* q_head = s.q + h * head_dim;
    float* att_head = s.att + h * config.seq_len;

    for (int t = 0; t <= pos; t++) {
      float* k_head = k_cache_layer + t * kv_dim + (h / kv_mul) * head_dim;
      float score = 0.0f;
      for (int d = 0; d < head_dim; d++) score += q_head[d] * k_head[d];
      att_head[t] = score * scale;
    }

    float max_val = att_head[0];
    for (int t = 1; t <= pos; t++) max_val = fmaxf(max_val, att_head[t]);
    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
      att_head[t] = expf(att_head[t] - max_val);
      sum += att_head[t];
    }
    for (int t = 0; t <= pos; t++) att_head[t] /= sum;

    float* out_head = s.xb + h * head_dim;
    memset(out_head, 0, head_dim * sizeof(float));
    for (int t = 0; t <= pos; t++) {
      float* v_head = v_cache_layer + t * kv_dim + (h / kv_mul) * head_dim;
      for (int d = 0; d < head_dim; d++) out_head[d] += att_head[t] * v_head[d];
    }
  }
}

float silu(float x) { return x * (1.0f / (1.0f + expf(-x))); }

// ============================================================================
// CPUDecoder 实现
// ============================================================================

CPUDecoder::CPUDecoder(const std::string& model_file) {
  if (load_config(config, model_file) != 0) {
    fprintf(stderr, "failed to load config\n");
    exit(1);
  }
  if (open_model(model_file, mf) != 0) {
    fprintf(stderr, "failed to open model\n");
    exit(1);
  }
  float* data = (float*)((char*)mf.data + sizeof(Config));
  load_weights(w, config, data, mf);
  alloc_run_state(state, config);
}

CPUDecoder::~CPUDecoder() {
  free_run_state(state);
  close_model(mf);
}

void CPUDecoder::forward(int token, int pos) {
  int dim = config.dim;
  int head_dim = config.dim / config.n_heads;
  int kv_dim = config.n_kv_heads * head_dim;

  // 1. Embedding lookup
  float* emb = w.token_embedding + token * dim;
  memcpy(state.x, emb, dim * sizeof(float));

  for (int l = 0; l < config.n_layers; l++) {
    // 2. Attention 前 RMSNorm
    rmsnorm(state.xb, state.x, w.rms_att + l * dim, dim);

    // 3. QKV 投影
    matmul(state.q, state.xb, w.wq + l * dim * dim, dim, dim);
    matmul(state.k, state.xb, w.wk + l * kv_dim * dim, dim, kv_dim);
    matmul(state.v, state.xb, w.wv + l * kv_dim * dim, dim, kv_dim);

    // 4. RoPE
    rope(state.q, state.k, w.freq_cis_real, w.freq_cis_imag, pos, dim, kv_dim,
         head_dim);

    // 5. KV Cache 写入
    float* k_cache_layer = state.k_cache + l * config.seq_len * kv_dim;
    float* v_cache_layer = state.v_cache + l * config.seq_len * kv_dim;
    memcpy(k_cache_layer + pos * kv_dim, state.k, kv_dim * sizeof(float));
    memcpy(v_cache_layer + pos * kv_dim, state.v, kv_dim * sizeof(float));

    // 6. Attention
    attention_cpu(state, config, l, pos, kv_dim, head_dim);

    // 7. Attention 输出投影 + 残差
    matmul(state.xb2, state.xb, w.wo + l * dim * dim, dim, dim);
    for (int i = 0; i < dim; i++) state.x[i] += state.xb2[i];

    // 8. FFN 前 RMSNorm
    rmsnorm(state.xb, state.x, w.rms_ffn + l * dim, dim);

    // 9. SwiGLU FFN
    matmul(state.hb, state.xb, w.w1 + l * config.hidden_dim * dim, dim,
           config.hidden_dim);
    matmul(state.hb2, state.xb, w.w3 + l * config.hidden_dim * dim, dim,
           config.hidden_dim);
    for (int i = 0; i < config.hidden_dim; i++)
      state.hb[i] = silu(state.hb[i]) * state.hb2[i];

    // 10. FFN 输出投影 + 残差
    matmul(state.xb2, state.hb, w.w2 + l * dim * config.hidden_dim,
           config.hidden_dim, dim);
    for (int i = 0; i < dim; i++) state.x[i] += state.xb2[i];
  }

  // 11. 最终 RMSNorm
  rmsnorm(state.xb, state.x, w.rms_final, dim);

  // 12. 输出 logits
  matmul(state.logits, state.xb, w.wcls, dim, abs(config.vocab_size));
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <model_file> <tokenizer_file>\n", argv[0]);
    return 1;
  }

  std::string model_file = argv[1];
  std::string tokenizer_file = argv[2];

  Config tmp_config;
  load_config(tmp_config, model_file);

  Tokenizer tokenizer;
  load_tokenizer(tokenizer, tokenizer_file, abs(tmp_config.vocab_size));

  std::mt19937 rng(time(nullptr));
  CPUDecoder decoder(model_file);

  std::string prompt;
  printf("Enter prompt: ");
  std::getline(std::cin, prompt);

  decoder.generate(tokenizer, prompt, tmp_config.seq_len, 0.8f, 40, rng);

  free_tokenizer(tokenizer);
  return 0;
}