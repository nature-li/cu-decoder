
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <string>

#include "common.h"

#define CHECK_CUDA(call)                                                      \
  {                                                                           \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), \
              __LINE__);                                                      \
      exit(1);                                                                \
    }                                                                         \
  }

struct GPUWeights {
  float* token_embedding;  // [vocab_size, dim]
  float* rms_att;          // [n_layers, dim]
  float* wq;               // [n_layers, dim, dim]
  float* wk;               // [n_layers, n_kv_heads*head_dim, dim]
  float* wv;               // [n_layers, n_kv_heads*head_dim, dim]
  float* wo;               // [n_layers, dim, dim]
  float* rms_ffn;          // [n_layers, dim]
  float* w1;               // [n_layers, hidden_dim, dim]
  float* w2;               // [n_layers, dim, hidden_dim]
  float* w3;               // [n_layers, hidden_dim, dim]
  float* rms_final;        // [dim]
  float* freq_cis_real;    // [seq_len, head_dim/2]
  float* freq_cis_imag;    // [seq_len, head_dim/2]
  float* wcls;             // [vocab_size, dim]
};

struct RunState {
  // [dim] 当前 token 的隐藏状态，每层 attention/FFN 都在这上边做 in-place 更新
  float* x;
  float* xb;   // [dim] RSMNorm 输出缓冲区，避免覆盖 x
  float* xb2;  // [dim] attention/FFN 输出投影的临时缓冲区
  float* q;    // [dim] 当前 token 的 Query 向量
  float* k;    // [kv_dim] 当前 token 的 Key 向量
  float* v;    // [kv_dim] 当前 token 的 Value 向量
  //[n_heads, seq_len] 每个 head 对所有历史 token 的 attention score
  float* att;
  float* hb;      // [hidden_dim] FFN 中间结果: SiLU(w1(x))
  float* hb2;     // [hidden_dim] FFN 中间结果: w3(x)
  float* logits;  // [vocab_size] 最终输出的 logits, 用来采样下一个 token
  // [n_layers, seq_len, kv_dim] 所有层的 Key Cache, 避免重复计算历史 token
  float* k_cache;
  // [n_layers, seq_len, kv_dim] 所有层的 Value Cache，避免重复计算历史 token
  float* v_cache;
};

void upload_weights(GPUWeights& gw, const Weights& w, const Config& config) {
  int head_dim = config.dim / config.n_heads;
  int kv_dim = head_dim * config.n_kv_heads;
  int vocab_size = abs(config.vocab_size);

  auto upload = [](float** dst, const float* src, size_t n) {
    CHECK_CUDA(cudaMalloc(dst, n * sizeof(float)));
    CHECK_CUDA(
        cudaMemcpy(*dst, src, n * sizeof(float), cudaMemcpyHostToDevice));
  };

  upload(&gw.token_embedding, w.token_embedding, vocab_size * config.dim);
  upload(&gw.rms_att, w.rms_att, config.n_layers * config.dim);
  upload(&gw.wq, w.wq, config.n_layers * config.dim * config.dim);
  upload(&gw.wk, w.wk, config.n_layers * kv_dim * config.dim);
  upload(&gw.wv, w.wv, config.n_layers * kv_dim * config.dim);
  upload(&gw.wo, w.wo, config.n_layers * config.dim * config.dim);
  upload(&gw.rms_ffn, w.rms_ffn, config.n_layers * config.dim);
  upload(&gw.w1, w.w1, config.n_layers * config.hidden_dim * config.dim);
  upload(&gw.w2, w.w2, config.n_layers * config.dim * config.hidden_dim);
  upload(&gw.w3, w.w3, config.n_layers * config.hidden_dim * config.dim);
  upload(&gw.rms_final, w.rms_final, config.dim);
  upload(&gw.freq_cis_real, w.freq_cis_real, config.seq_len * head_dim / 2);
  upload(&gw.freq_cis_imag, w.freq_cis_imag, config.seq_len * head_dim / 2);

  // wcls 共享时指向 token_embedding
  if (w.wcls == w.token_embedding) {
    gw.wcls = gw.token_embedding;
  } else {
    upload(&gw.wcls, w.wcls, vocab_size * config.dim);
  }
}

void free_gpu_weights(GPUWeights& gw, const Weights& w) {
  cudaFree(gw.token_embedding);
  cudaFree(gw.rms_att);
  cudaFree(gw.wq);
  cudaFree(gw.wk);
  cudaFree(gw.wv);
  cudaFree(gw.wo);
  cudaFree(gw.rms_ffn);
  cudaFree(gw.w1);
  cudaFree(gw.w2);
  cudaFree(gw.w3);
  cudaFree(gw.rms_final);
  cudaFree(gw.freq_cis_real);
  cudaFree(gw.freq_cis_imag);
  // wcls 共享时不重复释放
  if (w.wcls != w.token_embedding) {
    cudaFree(gw.wcls);
  }
}

int alloc_run_state(RunState& s, const Config& config) {
  int dim = config.dim;
  int kv_dim = (config.dim / config.n_heads) * config.n_kv_heads;
  int n_layers = config.n_layers;
  int n_heads = config.n_heads;
  int seq_len = config.seq_len;

  s.x = new float[dim];                  // 隐藏状态
  s.xb = new float[dim];                 // RUMNorm 输出
  s.xb2 = new float[dim];                // attention/FFN 输出投影的临时缓冲区
  s.q = new float[dim];                  // Query
  s.k = new float[kv_dim];               // Key
  s.v = new float[kv_dim];               // Value
  s.att = new float[n_heads * seq_len];  // attention scores
  s.hb = new float[config.hidden_dim];   // FFN 中间结果: SiLU(w1(x))
  s.hb2 = new float[config.hidden_dim];  // FFN 中间结果: w3(x)
  s.logits = new float[abs(config.vocab_size)];        // 输出 logits
  s.k_cache = new float[n_layers * seq_len * kv_dim];  // KV Cache: Key
  s.v_cache = new float[n_layers * seq_len * kv_dim];  // KV Cache: Value

  return 0;
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

/**
 * rms(x) = sqrt(mean(x^2) + 1e-6)
 * RMSNorm(x) = x / rms(x) * weight
 */
void rmsnorm(float* out, const float* x, const float* weight, int dim) {
  // 计算 x 的均方根
  float ss = 0.0f;
  for (int i = 0; i < dim; i++) {
    ss += x[i] * x[i];
  }
  ss = 1.0f / sqrtf(ss / dim + 1e-6f);

  // 归一化
  for (int i = 0; i < dim; i++) {
    out[i] = x[i] * ss * weight[i];
  }
}

/**
 * q = xb @ wq
 * k = xb @ wk
 * v = xb @ wv
 *
 * 参数
 * - n: 输入维度
 * - d: 输出维度
 * - w: [d, n]
 */
void matmul(float* out, const float* x, const float* w, int n, int d) {
  for (int i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += x[j] * w[i * n + j];
    }
    out[i] = val;
  }
}

/**
 * [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
 *
 * 参数:
 * - freq_real: [seq_len, head_dim/2]
 * - freq_imag: [seq_len, head_dim/2]
 */
void rope(float* q, float* k, const float* freq_real, const float* freq_imag,
          int pos, int dim, int kv_dim, int head_dim) {
  // 每两个元素做一次旋转, Q 和 K 都要做
  for (int i = 0; i < dim; i += 2) {
    int head_pos = i % head_dim;
    float cos_val = freq_real[pos * head_dim / 2 + head_pos / 2];
    float sin_val = freq_imag[pos * head_dim / 2 + head_pos / 2];

    float q0 = q[i];
    float q1 = q[i + 1];
    q[i] = q0 * cos_val - q1 * sin_val;
    q[i + 1] = q0 * sin_val + q1 * cos_val;
  }

  for (int i = 0; i < kv_dim; i += 2) {
    int head_pos = i % head_dim;
    float cos_val = freq_real[pos * head_dim / 2 + head_pos / 2];
    float sin_val = freq_imag[pos * head_dim / 2 + head_pos / 2];

    float k0 = k[i];
    float k1 = k[i + 1];
    k[i] = k0 * cos_val - k1 * sin_val;
    k[i + 1] = k0 * sin_val + k1 * cos_val;
  }
}

void attention(RunState& s, const Config& config, int layer, int pos,
               int kv_dim, int head_dim) {
  // GQA 时每个 KV 头服务几个 Q 头
  int kv_mul = config.n_heads / config.n_kv_heads;
  float scale = 1.0f / sqrtf((float)head_dim);

  float* k_cache_layer = s.k_cache + layer * config.seq_len * kv_dim;
  float* v_cache_layer = s.v_cache + layer * config.seq_len * kv_dim;

  for (int h = 0; h < config.n_heads; h++) {
    // 当前 head 的 Q
    float* q_head = s.q + h * head_dim;
    // 存储当前 head 的 scores
    float* att_head = s.att + h * config.seq_len;

    // Q @ K^T -> scores
    for (int t = 0; t <= pos; t++) {
      float* k_head = k_cache_layer + t * kv_dim + (h / kv_mul) * head_dim;
      float score = 0.0f;
      for (int d = 0; d < head_dim; d++) {
        score += q_head[d] * k_head[d];
      }
      att_head[t] = score * scale;
    }

    // softmax
    float max_val = att_head[0];
    for (int t = 1; t <= pos; t++) {
      max_val = fmaxf(max_val, att_head[t]);
    }
    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
      att_head[t] = expf(att_head[t] - max_val);
      sum += att_head[t];
    }
    for (int t = 0; t <= pos; t++) {
      att_head[t] /= sum;
    }

    // score @ V -> 输出
    float* out_head = s.xb + h * head_dim;
    memset(out_head, 0, head_dim * sizeof(float));
    for (int t = 0; t <= pos; t++) {
      float* v_head = v_cache_layer + t * kv_dim + (h / kv_mul) * head_dim;
      for (int d = 0; d < head_dim; d++) {
        out_head[d] += att_head[t] * v_head[d];
      }
    }
  }
}

/**
 * 激活函数 SiLU: x * sigmoid(x)
 */
float silu(float x) { return x * (1.0f / (1.0f + expf(-x))); }

void forward(Config& config, Weights& w, RunState& s, int token, int pos) {
  int dim = config.dim;

  // 1.Embedding lookup
  // 从 token embedding 表里取出第 token 行，作为初始隐藏状态
  float* emb = w.token_embedding + token * dim;
  memcpy(s.x, emb, dim * sizeof(float));

  // 2.过第一层
  for (int l = 0; l < config.n_layers; l++) {
    int head_dim = config.dim / config.n_heads;
    // K/V 向量的总长度, 等于 n_kv_heads * head_dim
    int kv_dim = config.n_kv_heads * head_dim;

    // attention 前的 RMSNorm
    rmsnorm(s.xb, s.x, w.rms_att + l * dim, dim);

    // QKV 投影
    matmul(s.q, s.xb, w.wq + l * dim * dim, dim, dim);
    matmul(s.k, s.xb, w.wk + l * kv_dim * dim, dim, kv_dim);
    matmul(s.v, s.xb, w.wv + l * kv_dim * dim, dim, kv_dim);

    // RoPE
    rope(s.q, s.k, w.freq_cis_real, w.freq_cis_imag, pos, dim, kv_dim,
         head_dim);

    // 写入 KV Cache
    // 当前层 KV Cache 的起始指针
    float* k_cache_layer = s.k_cache + l * config.seq_len * kv_dim;
    float* v_cache_layer = s.v_cache + l * config.seq_len * kv_dim;
    // 把当前 pos 的 K/V 写入 cache
    memcpy(k_cache_layer + pos * kv_dim, s.k, kv_dim * sizeof(float));
    memcpy(v_cache_layer + pos * kv_dim, s.v, kv_dim * sizeof(float));

    // attention
    attention(s, config, l, pos, kv_dim, head_dim);
    // attention 输出投影: xb = xb @ wo
    matmul(s.xb2, s.xb, w.wo + l * dim * dim, dim, dim);
    for (int i = 0; i < dim; i++) {
      s.x[i] += s.xb2[i];
    }

    // FFN 前的 RMSNorm
    rmsnorm(s.xb, s.x, w.rms_ffn + l * dim, dim);

    /**
     * FFN(x) = w2(SiLU(w1(x)) * w3(x))
     */
    matmul(s.hb, s.xb, w.w1 + l * config.hidden_dim * dim, dim,
           config.hidden_dim);
    matmul(s.hb2, s.xb, w.w3 + l * config.hidden_dim * dim, dim,
           config.hidden_dim);
    for (int i = 0; i < config.hidden_dim; i++) {
      s.hb[i] = silu(s.hb[i]) * s.hb2[i];
    }

    // FFN 输出投影 + 残差连接
    matmul(s.xb2, s.hb, w.w2 + l * dim * config.hidden_dim, config.hidden_dim,
           dim);
    for (int i = 0; i < dim; i++) {
      s.x[i] += s.xb2[i];
    }
  }

  // 最终 RMSNorm
  rmsnorm(s.xb, s.x, w.rms_final, dim);

  // 输出 logits
  matmul(s.logits, s.xb, w.wcls, dim, abs(config.vocab_size));
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <model_file>\n", argv[0]);
    return 1;
  }
  std::string model_file = argv[1];

  // 加载 CPU 权重
  Config config;
  load_config(config, model_file);

  ModelFile mf;
  open_model(model_file, mf);

  float* data = (float*)((char*)mf.data + sizeof(Config));
  Weights w;
  load_weights(w, config, data, mf);

  // 上传到 GPU
  GPUWeights gw;
  upload_weights(gw, w, config);

  // 读回来对比
  float cpu_val, gpu_val;

  // 验证 token_embedding[0]
  cpu_val = w.token_embedding[0];
  CHECK_CUDA(cudaMemcpy(&gpu_val, gw.token_embedding, sizeof(float),
                        cudaMemcpyDeviceToHost));
  printf("token_embedding[0]: cpu=%f gpu=%f match=%d\n", cpu_val, gpu_val,
         cpu_val == gpu_val);

  // 验证 rms_final[0]
  cpu_val = w.rms_final[0];
  CHECK_CUDA(cudaMemcpy(&gpu_val, gw.rms_final, sizeof(float),
                        cudaMemcpyDeviceToHost));
  printf("rms_final[0]:       cpu=%f gpu=%f match=%d\n", cpu_val, gpu_val,
         cpu_val == gpu_val);

  // 验证 wq 最后一个元素
  int wq_size = config.n_layers * config.dim * config.dim;
  cpu_val = w.wq[wq_size - 1];
  CHECK_CUDA(cudaMemcpy(&gpu_val, gw.wq + wq_size - 1, sizeof(float),
                        cudaMemcpyDeviceToHost));
  printf("wq[last]:           cpu=%f gpu=%f match=%d\n", cpu_val, gpu_val,
         cpu_val == gpu_val);

  free_gpu_weights(gw, w);
  close_model(mf);
  return 0;
}