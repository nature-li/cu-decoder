#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "gpu_decoder.h"

#define CHECK_CUDA(call)                                                      \
  {                                                                           \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), \
              __LINE__);                                                      \
      exit(1);                                                                \
    }                                                                         \
  }

// ============================================================================
// Kernels
// ============================================================================

/**
 * Embedding lookup kernel
 * Grid:  (dim + 255) / 256 个 block
 * Block: 256 个线程
 * 线程 i: out[i] = table[token * dim + i]
 */
__global__ void embedding_kernel(float* out, const float* table, int token,
                                 int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim) out[i] = table[token * dim + i];
}

/**
 * warp 内归约求和
 * 调用后 warp 内 lane 0 持有所有线程的 val 之和
 */
__device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    /**
     * 让当前线程去获取排在它后面第 offset 个线程里的数据
     * 返回值 = 线程(i + offset) 的 val
     */
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

/**
 * RMSNorm kernel
 *
 * 每个 thread 负责跨步累加自己的平方和
 * warp 内用 __shfl_down_sync 归约，无 atomicAdd
 * 各 warp 结果写入 shared memory，再做最终归约
 *
 * Grid:  1 个 block
 * Block: 256 个线程（8 个 warp）
 * 每个 thread: 跨步累加 x[tid], x[tid+256], x[tid+512]... 的平方
 */
__global__ void rmsnorm_kernel(float* out, const float* x, const float* weight,
                               int dim) {
  // 8 个 warp，每个 warp 的归约结果存在这里
  __shared__ float warp_sums[8];

  int tid = threadIdx.x;
  int warp_id = tid / 32;  // 属于第几个 warp
  int lane_id = tid % 32;  // 在 warp 内的位置

  // 1. 每个线程跨步累加平方和
  float local_sum = 0.0f;
  for (int i = tid; i < dim; i += blockDim.x) {
    local_sum += x[i] * x[i];
  }

  // 2. warp 内归约
  local_sum = warp_reduce_sum(local_sum);

  // 3. 每个 warp 的 lane 0 把结果写入 shared memory
  if (lane_id == 0) {
    warp_sums[warp_id] = local_sum;
  }
  __syncthreads();

  // 4. 用第一个 warp 对 8 个 warp 结果做最终归约
  if (warp_id == 0) {
    float val = (lane_id < 8) ? warp_sums[lane_id] : 0.0f;
    val = warp_reduce_sum(val);

    // lane 0 写入最终结果
    if (lane_id == 0) {
      warp_sums[0] = val;
    }
  }
  __syncthreads();

  // 5. 归一化
  float norm = rsqrtf(warp_sums[0] / dim + 1e-6f);
  for (int i = tid; i < dim; i += blockDim.x) {
    out[i] = x[i] * norm * weight[i];
  }
}

/**
 * 用 cublasSgemv 替换手写的 matmul_kernel
 * 计算 out = x @ w^T
 * w: [d, n] 行优先
 * x: [n]
 * out: [d]
 *
 * 行主序下
 * out = x @ w^T = w @ x
 *
 * 代入 cublas 后的参数:
 * A=w, op=CUBLAS_OP_T
 * x=x
 * m=n
 * n=d
 * lda=n
 */
void matmul_cublas(cublasHandle_t handle, float* out, float* x, const float* w,
                   int n, int d) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  /**
   * 计算公式:
   * y = alpha * op(A) * x + beta * y
   *
   * 参数:
   * - handle:  cuBLAS 句柄
   * - trans:   是否对 A 做转置
   * - m:       A 的行数
   * - n:       A 的列数
   * - alpha:   标量 alpha
   * - A:       矩阵 A, shape: [m, n]
   * - lda:     A 的 leading demension, 列优先下是行数
   * - x:       输入向量 x
   * - incx:    x 的元素间距, 1 表示连续
   * - beta:    标量 beta
   * - y:       输出向量 y
   * - incy:    y 的元素间距, 1 表示连续
   */
  cublasSgemv(handle, CUBLAS_OP_T, n, d, &alpha, w, n, x, 1, &beta, out, 1);
}

/**
 * RoPE kernel
 * Grid:  (dim/2 + 255) / 256 个 block
 * Block: 256 个线程
 * 线程 i: 负责第 i 对元素的旋转
 */
__global__ void rope_kernel(float* q, float* k, const float* freq_real,
                            const float* freq_imag, int pos, int dim,
                            int kv_dim, int head_dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = i * 2;

  if (idx < dim) {
    int head_pos = idx % head_dim;
    float cos_val = freq_real[pos * head_dim / 2 + head_pos / 2];
    float sin_val = freq_imag[pos * head_dim / 2 + head_pos / 2];
    float q0 = q[idx], q1 = q[idx + 1];
    q[idx] = q0 * cos_val - q1 * sin_val;
    q[idx + 1] = q0 * sin_val + q1 * cos_val;
  }

  if (idx < kv_dim) {
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
 * Grid:  (kv_dim + 255) / 256 个 block
 * Block: 256 个线程
 * 线程 i: 写入 k_cache 和 v_cache 的第 i 个元素
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
 * Grid:  n_heads 个 block，每个 block 负责一个 head
 * Block: 256 个线程
 * shared memory: [seq_len] 存 attention scores
 */
__global__ void attention_kernel(const float* q, const float* k_cache,
                                 const float* v_cache, float* out, int pos,
                                 int seq_len, int kv_dim, int head_dim,
                                 int kv_mul) {
  int h = blockIdx.x;
  int tid = threadIdx.x;
  float scale = rsqrtf((float)head_dim);

  extern __shared__ float scores[];

  const float* q_head = q + h * head_dim;

  // 1. Q @ K^T -> scores
  for (int t = tid; t <= pos; t += blockDim.x) {
    const float* k_head = k_cache + t * kv_dim + (h / kv_mul) * head_dim;
    float score = 0.0f;
    for (int d = 0; d < head_dim; d++) score += q_head[d] * k_head[d];
    scores[t] = score * scale;
  }
  __syncthreads();

  // 2. softmax
  if (tid == 0) {
    float max_val = scores[0];
    for (int t = 1; t <= pos; t++) max_val = fmaxf(max_val, scores[t]);
    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {
      scores[t] = expf(scores[t] - max_val);
      sum += scores[t];
    }
    for (int t = 0; t <= pos; t++) scores[t] /= sum;
  }
  __syncthreads();

  // 3. scores @ V -> 输出
  float* out_head = out + h * head_dim;
  for (int d = tid; d < head_dim; d += blockDim.x) {
    float val = 0.0f;
    for (int t = 0; t <= pos; t++) {
      const float* v_head = v_cache + t * kv_dim + (h / kv_mul) * head_dim;
      val += scores[t] * v_head[d];
    }
    out_head[d] = val;
  }
}

/**
 * SwiGLU kernel
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
 * Grid:  (dim + 255) / 256 个 block
 * Block: 256 个线程
 * 线程 i: x[i] += delta[i]
 */
__global__ void residual_kernel(float* x, const float* delta, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < dim) x[i] += delta[i];
}

// ============================================================================
// GPUWeights 上传/释放
// ============================================================================

void upload_weights(GPUWeights& gw, const Weights& w, const Config& config) {
  int head_dim = config.dim / config.n_heads;
  int kv_dim = config.n_kv_heads * head_dim;
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
  if (w.wcls != w.token_embedding) cudaFree(gw.wcls);
}

// ============================================================================
// GPURunState 分配/释放
// ============================================================================

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

// ============================================================================
// GPUDecoder 实现
// ============================================================================

GPUDecoder::GPUDecoder(const std::string& model_file) {
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
  upload_weights(gw, w, config);
  alloc_gpu_run_state(gpu_state, config);

  cublasCreate(&cublas_handle);
}

GPUDecoder::~GPUDecoder() {
  cublasDestroy(cublas_handle);
  free_gpu_run_state(gpu_state);
  free_gpu_weights(gw, w);
  close_model(mf);
}

float* GPUDecoder::get_logits() {
  cudaDeviceSynchronize();
  return gpu_state.logits;
}

void GPUDecoder::forward(int token, int pos) {
  int dim = config.dim;
  int head_dim = config.dim / config.n_heads;
  int kv_dim = config.n_kv_heads * head_dim;
  int kv_mul = config.n_heads / config.n_kv_heads;
  int vocab_size = abs(config.vocab_size);
  int threads = 256;

  // 1. Embedding lookup
  embedding_kernel<<<(dim + threads - 1) / threads, threads>>>(
      gpu_state.x, gw.token_embedding, token, dim);

  for (int l = 0; l < config.n_layers; l++) {
    // 2. Attention 前 RMSNorm
    rmsnorm_kernel<<<1, threads>>>(gpu_state.xb, gpu_state.x,
                                   gw.rms_att + l * dim, dim);

    // 3. QKV 投影
    matmul_cublas(cublas_handle, gpu_state.q, gpu_state.xb,
                  gw.wq + l * dim * dim, dim, dim);
    matmul_cublas(cublas_handle, gpu_state.k, gpu_state.xb,
                  gw.wk + l * kv_dim * dim, dim, kv_dim);
    matmul_cublas(cublas_handle, gpu_state.v, gpu_state.xb,
                  gw.wv + l * kv_dim * dim, dim, kv_dim);

    // 4. RoPE
    rope_kernel<<<(dim / 2 + threads - 1) / threads, threads>>>(
        gpu_state.q, gpu_state.k, gw.freq_cis_real, gw.freq_cis_imag, pos, dim,
        kv_dim, head_dim);

    // 5. KV Cache 写入
    kvcache_write_kernel<<<(kv_dim + threads - 1) / threads, threads>>>(
        gpu_state.k_cache, gpu_state.v_cache, gpu_state.k, gpu_state.v, l, pos,
        config.seq_len, kv_dim);

    // 6. Attention
    size_t shared_mem = config.seq_len * sizeof(float);
    attention_kernel<<<config.n_heads, threads, shared_mem>>>(
        gpu_state.q, gpu_state.k_cache + l * config.seq_len * kv_dim,
        gpu_state.v_cache + l * config.seq_len * kv_dim, gpu_state.xb, pos,
        config.seq_len, kv_dim, head_dim, kv_mul);

    // 7. Attention 输出投影 + 残差
    matmul_cublas(cublas_handle, gpu_state.xb2, gpu_state.xb,
                  gw.wo + l * dim * dim, dim, dim);
    residual_kernel<<<(dim + threads - 1) / threads, threads>>>(
        gpu_state.x, gpu_state.xb2, dim);

    // 8. FFN 前 RMSNorm
    rmsnorm_kernel<<<1, threads>>>(gpu_state.xb, gpu_state.x,
                                   gw.rms_ffn + l * dim, dim);

    // 9. SwiGLU FFN
    matmul_cublas(cublas_handle, gpu_state.hb, gpu_state.xb,
                  gw.w1 + l * config.hidden_dim * dim, dim, config.hidden_dim);
    matmul_cublas(cublas_handle, gpu_state.hb2, gpu_state.xb,
                  gw.w3 + l * config.hidden_dim * dim, dim, config.hidden_dim);
    swiglu_kernel<<<(config.hidden_dim + threads - 1) / threads, threads>>>(
        gpu_state.hb, gpu_state.hb2, config.hidden_dim);

    // 10. FFN 输出投影 + 残差
    matmul_cublas(cublas_handle, gpu_state.xb2, gpu_state.hb,
                  gw.w2 + l * dim * config.hidden_dim, config.hidden_dim, dim);
    residual_kernel<<<(dim + threads - 1) / threads, threads>>>(
        gpu_state.x, gpu_state.xb2, dim);
  }

  // 11. 最终 RMSNorm
  rmsnorm_kernel<<<1, threads>>>(gpu_state.xb, gpu_state.x, gw.rms_final, dim);

  // 12. 输出 logits 到 pinned memory
  matmul_cublas(cublas_handle, gpu_state.logits, gpu_state.xb, gw.wcls, dim,
                vocab_size);
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
  GPUDecoder decoder(model_file);

  std::string prompt;
  printf("Enter prompt: ");
  std::getline(std::cin, prompt);

  decoder.generate(tokenizer, prompt, tmp_config.seq_len, 0.8f, 40, rng);

  free_tokenizer(tokenizer);
  return 0;
}