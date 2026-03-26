
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

struct GPURunState {
  float* x;    // [dim]
  float* xb;   // [dim]
  float* xb2;  // [dim]
  float* q;    // [dim]
  float* k;    // [kv_dim]
  float* v;    // [kv_dim]
  float* att;  // [n_heads, seq_len]
  float* hb;   // [hidden_dim]
  float* hb2;  // [hidden_dim]
  // [vocab_size] 这个需要 CPU 能读到，用 cudaMallocHost 或普通 malloc
  float* logits;
  float* k_cache;  // [n_layers, seq_len, kv_dim]
  float* v_cache;  // [n_layers, seq_len, kv_dim]
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

void alloc_gpu_run_state(GPURunState& s, const Config& config) {
  int dim = config.dim;
  int kv_dim = config.n_kv_heads * (config.dim / config.n_heads);
  int n_layers = config.n_layers;
  int n_heads = config.n_heads;
  int seq_len = config.seq_len;
  int vocab_size = abs(config.vocab_size);

  CHECK_CUDA(cudaMalloc(&s.x, dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&s.xb, dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&s.xb2, dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&s.q, dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&s.k, kv_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&s.v, kv_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&s.att, n_heads * seq_len * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&s.hb, config.hidden_dim * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&s.hb2, config.hidden_dim * sizeof(float)));
  CHECK_CUDA(
      cudaMalloc(&s.k_cache, n_layers * seq_len * kv_dim * sizeof(float)));
  CHECK_CUDA(
      cudaMalloc(&s.v_cache, n_layers * seq_len * kv_dim * sizeof(float)));

  // logits 需要频繁从 GPU 读回 CPU 做采样，用 pinned memory 更快
  CHECK_CUDA(cudaMallocHost(&s.logits, vocab_size * sizeof(float)));
}

void free_gpu_run_state(GPURunState& s) {
  cudaFree(s.x);
  cudaFree(s.xb);
  cudaFree(s.xb2);
  cudaFree(s.q);
  cudaFree(s.k);
  cudaFree(s.v);
  cudaFree(s.att);
  cudaFree(s.hb);
  cudaFree(s.hb2);
  cudaFree(s.k_cache);
  cudaFree(s.v_cache);
  cudaFreeHost(s.logits);
}

/**
 * Embedding lookup kernel
 *
 * 每个线程负责取出 embedding 表中第 token 行的一个元素
 *
 * Grid: (dim + 255) / 256 个 block
 * Block: 256 个线程
 * 线程 i: out[i] = table[token * dim + i]
 *
 * 例:
 * dim=288, 启动 2 个 block, 共 512 个线程
 * 线程 0~287 各取一个元素, 线程 288~511 越界直接返回
 */
__global__ void embedding_kernel(float* out, const float* table, int token,
                                 int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim) {
    out[i] = table[token * dim + i];
  }
}

/**
 * RMSNorm kernel
 *
 * 公式:
 * rms(x) = sqrt(mean(x^2) + 1e-6)
 * out[i] = x[i] / rms(x) * weight[i]
 *
 *
 * Grid: 1 个 block (因为需要对整个 dim 做全局归约求和)
 * Block: 256 个线程
 * 每个线程负责: 跨步累加自己负责的元素的平方和，最后参与归一化
 *
 * 例:
 * dim=288, 256 个线程
 * 线程 0:   处理 x[0], x[256]        (跨步 256)
 * 线程 1:   处理 x[1], x[257]
 * ...
 * 线程 31:  处理 x[31], x[287]
 * 线程 32~255: 只处理 x[32]~x[255]   (第二步越界)
 *
 * 归约方式: 每个线程算局部平方和，atomicAdd 到 shared memory 汇总
 */

__global__ void rmsnorm_kernel(float* out, const float* x, const float* weight,
                               int dim) {
  __shared__ float shared_sum;
  if (threadIdx.x == 0) {
    shared_sum = 0.0f;
  }
  __syncthreads();

  // 1. 每个线程计算自己负责元素的平方和
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    local_sum += x[i] * x[i];
  }

  // 2. 原子加到 shared memory 汇总
  atomicAdd(&shared_sum, local_sum);
  __syncthreads();

  // 3. 计算归一化因子
  float norm = rsqrtf(shared_sum / dim + 1e-6f);

  // 4. 每个线程归一化自己负责的元素
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    out[i] = x[i] * norm * weight[i];
  }
}

/**
 * MatMul kernel: out = x @ w^T
 *
 * x: [n]      输入向量
 * w: [d, n]   权重矩阵
 * out: [d]    输出向量
 *
 * Grid:  (d + 255) / 256 个 block
 * Block: 256 个线程
 * 线程 i: 计算 out[i] = x 和 w 第 i 行的点积
 */
__global__ void matmul_kernel(float* out, const float* x, const float* w, int n,
                              int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < d) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += x[j] * w[i * n + j];
    }
    out[i] = val;
  }
}

/**
 * RoPE kernel
 *
 * 对 Q 和 K 做旋转位置编码
 * 每两个元素一组做旋转: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
 *
 * Grid:  (dim/2 + 255) / 256 个 block
 * Block: 256 个线程
 * 线程 i: 负责处理第 i 对元素的旋转
 * - Q 的第 i 对: q[2i], q[2i+1]
 * - K 的第 i 对: k[2i], k[2i+1] (如果 2i < kv_dim)
 */
__global__ void rope_kernel(float* q, float* k, const float* freq_real,
                            const float* freq_imag, int pos, int dim,
                            int kv_dim, int head_dim) {
  // 第 i 对
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // 对应元素下标
  int idx = i * 2;

  if (idx < dim) {
    int head_pos = idx % head_dim;
    float cos_val = freq_real[pos * head_dim / 2 + head_pos / 2];
    float sin_val = freq_imag[pos * head_dim / 2 + head_pos / 2];

    float q0 = q[idx], q1 = q[idx + 1];
    q[idx] = q0 * cos_val - q1 * sin_val;
    q[idx + 1] = q0 * sin_val + q1 * cos_val;
  }

  if (idx < head_dim) {
    int head_pos = idx % head_dim;
    float cos_val = freq_real[pos * head_dim / 2 + head_pos / 2];
    float sin_val = freq_imag[pos * head_dim / 2 + head_pos / 2];

    float k0 = k[idx], k1 = k[idx + 1];
    k[idx] = k0 * cos_val - k1 * sin_val;
    k[idx + 1] = k0 * sin_val + k1 * cos_val;
  }
}

/**
 * KV Cache 写入 kernel
 *
 * 把当前 pos 的 K/V 写入对应层的 cache
 *
 * Grid:  (kv_dim + 255) / 256 个 block
 * Block: 256 个线程
 * 线程 i:
 * - k_cache[layer * seq_len * kv_dim + pos * kv_dim + i] = k[i]
 * - v_cache[layer * seq_len * kv_dim + pos * kv_dim + i] = v[i]
 */
__global__ void kvcache_write_kernel(float* k_cache, float* v_cache,
                                     const float* k, const float* v, int layer,
                                     int pos, int seq_len, int kv_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kv_dim) {
    int offset = layer * seq_len * kv_dim + pos * kv_dim + i;
    k_cache[offset] = k[i];
    v_cache[offset] = v[i];
  }
}

/**
 * Attention kernel
 *
 * Grid:  n_heads 个 block，每个 block 负责一个 head
 * Block: 256 个线程
 * 每个 block:
 * - 1. 计算当前 head 的 Q @ K^T -> scores
 * - 2. softmax
 * - 3. scores @ V -> 输出
 *
 * shared memory: [seq_len] 存 attention scores
 */
__global__ void attention_kernel(const float* q, const float* k_cache,
                                 const float* v_cache, float* out, int pos,
                                 int seq_len, int kv_dim, int head_dim,
                                 int kv_mul) {
  int h = blockIdx.x;  // 当前 head
  int tid = threadIdx.x;
  float scale = rsqrtf((float)head_dim);

  extern __shared__ float scores[];  // seq_len

  const float* q_head = q + h * head_dim;
  const float* k_layer = k_cache;
  const float* v_layer = v_cache;

  // 1. Q @ K^T -> scores，跨步循环处理所有历史 token
  for (int t = tid; t <= pos; t += blockDim.x) {
    const float* k_head = k_layer + t * kv_dim + (h / kv_mul) * head_dim;
    float score = 0.0f;
    for (int d = 0; d < head_dim; d++) {
      score += q_head[d] * k_head[d];
    }
    scores[t] = score * scale;
  }
  __syncthreads();

  // 2. softmax，0 号线程串行做
  if (tid == 0) {
    float max_val = scores[0];
    for (int t = 1; t <= pos; t++) {
      max_val = fmaxf(max_val, scores[t]);
    }

    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
      scores[t] = expf(scores[t] - max_val);
      sum += scores[t];
    }

    for (int t = 0; t <= pos; t++) {
      scores[t] /= sum;
    }
  }
  __syncthreads();

  // 3. scores @ V -> 输出，跨步循环处理 head_dim
  float* out_head = out + h * head_dim;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    float val = 0.0f;
    for (int t = 0; t <= pos; t++) {
      const float* v_head = v_layer + t * kv_dim + (h / kv_mul) * head_dim;
      val += scores[t] * v_head[d];
    }
    out_head[d] = val;
  }
}

/**
 * SwiGLU kernel
 *
 * FFN(x) = w2(SiLU(w1(x)) * w3(x))
 * 这个 kernel 负责中间那步: hb[i] = SiLU(hb[i]) * hb2[i]
 *
 * Grid:  (hidden_dim + 255) / 256 个 block
 * Block: 256 个线程
 * 线程 i: hb[i] = silu(hb[i]) * hb2[i]
 */
__global__ void swiglu_kernel(float* hb, const float* hb2, int hidden_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < hidden_dim) {
    float x = hb[i];
    hb[i] = x * (1.0f / (1.0f + expf(-x))) * hb2[i];
  }
}

/**
 * Residual add kernel
 *
 * x[i] += delta[i]
 *
 * Grid:  (dim + 255) / 256 个 block
 * Block: 256 个线程
 * 线程 i: x[i] += delta[i]
 */
__global__ void residual_kernel(float* x, const float* delta, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim) {
    x[i] += delta[i];
  }
}

void forward_gpu(const Config& config, const GPUWeights& gw, GPURunState& s,
                 int token, int pos) {
  int dim = config.dim;
  int head_dim = config.dim / config.n_heads;
  int kv_dim = config.n_kv_heads * head_dim;
  int kv_mul = config.n_heads / config.n_kv_heads;
  int vocab_size = abs(config.vocab_size);
  int threads = 256;

  // 1. Embedding lookup
  {
    int blocks = (dim + threads - 1) / threads;
    embedding_kernel<<<blocks, threads>>>(s.x, gw.token_embedding, token, dim);
  }

  for (int l = 0; l < config.n_layers; l++) {
    // 2. Attention 前的 RMSNorm
    rmsnorm_kernel<<<1, threads>>>(s.xb, s.x, gw.rms_att + l * dim, dim);

    // 3. QKV 投影
    {
      matmul_kernel<<<(dim + threads - 1) / threads, threads>>>(
          s.q, s.xb, gw.wq + l * dim * dim, dim, dim);
      matmul_kernel<<<(kv_dim + threads - 1) / threads, threads>>>(
          s.k, s.xb, gw.wk + l * kv_dim * dim, dim, kv_dim);
      matmul_kernel<<<(kv_dim + threads - 1) / threads, threads>>>(
          s.v, s.xb, gw.wv + l * kv_dim * dim, dim, kv_dim);
    }

    // 4. RoPE
    {
      int blocks = (dim / 2 + threads - 1) / threads;
      rope_kernel<<<blocks, threads>>>(s.q, s.k, gw.freq_cis_real,
                                       gw.freq_cis_imag, pos, dim, kv_dim,
                                       head_dim);
    }

    // 5. KV Cache 写入
    {
      int blocks = (kv_dim + threads - 1) / threads;
      kvcache_write_kernel<<<blocks, threads>>>(s.k_cache, s.v_cache, s.k, s.v,
                                                l, pos, config.seq_len, kv_dim);
    }

    // 6. Attention
    {
      size_t shared_mem = config.seq_len * sizeof(float);
      attention_kernel<<<config.n_heads, 256, shared_mem>>>(
          s.q, s.k_cache + l * config.seq_len * kv_dim,
          s.v_cache + l * config.seq_len * kv_dim, s.xb, pos, config.seq_len,
          kv_dim, head_dim, kv_mul);
    }

    // 7. Attention 输出投影 + 残差连接
    {
      matmul_kernel<<<(dim + threads - 1) / threads, threads>>>(
          s.xb2, s.xb, gw.wo + l * dim * dim, dim, dim);
      residual_kernel<<<(dim + threads - 1) / threads, threads>>>(s.x, s.xb2,
                                                                  dim);
    }

    // 8. FFN 前的 RMSNorm
    rmsnorm_kernel<<<1, 256>>>(s.xb, s.x, gw.rms_ffn + l * dim, dim);

    // 9. FFN: SwiGLU
    {
      matmul_kernel<<<(config.hidden_dim + threads - 1) / threads, threads>>>(
          s.hb, s.xb, gw.w1 + l * config.hidden_dim * dim, dim,
          config.hidden_dim);
      matmul_kernel<<<(config.hidden_dim + threads - 1) / threads, threads>>>(
          s.hb2, s.xb, gw.w3 + l * config.hidden_dim * dim, dim,
          config.hidden_dim);
      swiglu_kernel<<<(config.hidden_dim + threads - 1) / threads, threads>>>(
          s.hb, s.hb2, config.hidden_dim);
    }

    // 10. FFN 输出投影 + 残差连接
    {
      matmul_kernel<<<(dim + threads - 1) / threads, threads>>>(
          s.xb2, s.hb, gw.w2 + l * dim * config.hidden_dim, config.hidden_dim,
          dim);
      residual_kernel<<<(dim + threads - 1) / threads, threads>>>(s.x, s.xb2,
                                                                  dim);
    }
  }

  // 11. 最终 RMSNorm
  rmsnorm_kernel<<<1, threads>>>(s.xb, s.x, gw.rms_final, dim);

  // 12.输出 logits，结果写到 pinned memory
  {
    matmul_kernel<<<(vocab_size + threads - 1) / threads, threads>>>(
        s.logits, s.xb, gw.wcls, dim, vocab_size);
  }
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

  GPURunState s;
  alloc_gpu_run_state(s, config);

  free_gpu_run_state(s);
  free_gpu_weights(gw, w);
  close_model(mf);
  return 0;
}