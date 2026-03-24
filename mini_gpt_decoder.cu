#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>

// ============================================================================
// 1. 结构与宏
// ============================================================================
#define CHECK_CUDA(call)                                                 \
  {                                                                      \
    cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                            \
      fprintf(stderr, "CUDA Error: %s at %d\n", cudaGetErrorString(err), \
              __LINE__);                                                 \
      exit(1);                                                           \
    }                                                                    \
  }

struct GPTConfig {
  int vocab_size;
  int d_model;
  int num_heads;
  int head_dim;
  int ffn_hidden;
  int max_seq_len;
  int num_layers;
};

// ============================================================================
// 2. 所有 Kernels (无省略)
// ============================================================================

/**
 * out += bias
 */
__global__ void add_bias_kernel(float* out, const float* bias, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] += bias[i];
  }
}

/**
 * 高性能线性层 (Linear Layer) 实现 - PyTorch 标准对齐版
 * 计算 C = X @ W.T + bias
 *
 * @param handle    cuBLAS 句柄
 * @param m         Batch Size (m)
 * @param n         输出维度 (out_features)
 * @param k         输入维度 (in_features)
 * @param d_x       输入向量: [m, k] (行优先)
 * @param d_w       权重矩阵: [n, k] (行优先, PyTorch 标准)
 * @param d_bias    偏置项:   [n]
 * @param d_out     输出向量: [m, n] (行优先)
 */
void cublas_linear(cublasHandle_t handle, int m, int n, int k,
                   const float* d_x,     // [m, k]
                   const float* d_w,     // [n, k]
                   const float* d_bias,  // [1, n]
                   float* d_out          // [m, n]
) {
  float alpha = 1.0f;
  float beta = 0.0f;

  /**
   * C = A * B (在 cuBLAS 眼中)
   * 推理时 x 是 (1, k), W 是 (k, n), out 是 (1, n)
   * 因为 cuBLAS 是列优先，我们通过调换参数实现行优先计算
   *
   * 官方标准调用 (列优先计算 C[m,n] = A[m,k] * B[k,n])
   * cublasSgemm(handle, transa, transb, m, n, k, ... A, lda, B, ldb, ... C,
   * ldc);
   *
   * C_T: shape[n, m]
   * A_T: shape[k, m]
   * B_T: shape[n, k]
   *
   * C_T = B_T * A_T = [n, m]
   *
   * 目标(Row-Major): C[m, n] = X[m, k] @ W^T[k, n]
   * cuBLAS(Col-Major): C_col[n, m] = W_col^T[n, k] @ X_col[k, m]
   *
   * W 在内存中是 [n, k] 行优先 -> W_col 是 [k, n] 列优先
   * X 在内存中是 [m, k] 行优先 -> X_col 是 [k, m] 列优先
   * 我们需要 W_col^T (变成 n x k) 去乘 X_col (k x m)
   */
  cublasSgemm(handle,       // cuBLAS 上下文句柄
              CUBLAS_OP_T,  // 矩阵 A 是否转置
              CUBLAS_OP_N,  // 矩阵 B 是否转置
              n,            // 结果矩阵 C_col 的行数 (out_features)
              m,            // 结果矩阵 C_col 的列数 (batch_size)
              k,            // 公共维度 (in_features)
              &alpha,       // 缩放系数 alpha (通常指向 1.0f)
              d_w,          // A 矩阵 (Weight)
              k,            // lda: W_col 的行数，即 W 行优先时的列数 k
              d_x,          // B 矩阵 (Input)
              k,            // ldb: X_col 的行数，即 X 行优先时的列数 k
              &beta,        // 缩放系数 beta (通常指向 0.0f)
              d_out,        // C 矩阵 (Output)
              n);           // ldc: C_col 的行数 n

  /**
   * cuBLAS 不支持 Bias，我们需要补一个简单的 kernel
   */
  if (d_bias != nullptr) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads>>>(d_out, d_bias, n);
    CHECK_CUDA(cudaGetLastError());
  }
}

/**
 * out = emb_table[token_id]
 */
__global__ void embedding_kernel(float* out, int token_id,
                                 const float* emb_table, int d_model) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = tid; i < d_model; i += blockDim.x * gridDim.x)
    out[i] = emb_table[token_id * d_model + i];
}

__global__ void rms_norm_kernel(float* out, const float* inv,
                                const float* weight, int d_model) {
  int tid = threadIdx.x;
  __shared__ float s_variance;
  float sum = 0.0f;
  for (int i = tid; i < d_model; i += blockDim.x) sum += inv[i] * inv[i];
  if (tid == 0) s_variance = 0;
  __syncthreads();
  atomicAdd(&s_variance, sum);
  __syncthreads();
  float norm_factor = rsqrtf(s_variance / d_model + 1e-6f);
  for (int i = tid; i < d_model; i += blockDim.x)
    out[i] = inv[i] * norm_factor * weight[i];
}

__global__ void split_qkv_kernel(const float* qkv, float* q, float* k_new,
                                 float* v_new, int d_model) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < d_model) {
    q[tid] = qkv[tid];
    k_new[tid] = qkv[tid + d_model];
    v_new[tid] = qkv[tid + 2 * d_model];
  }
}

__global__ void update_cache_kernel(float* k_cache, float* v_cache,
                                    const float* k_new, const float* v_new,
                                    int step, int num_heads, int head_dim,
                                    int max_seq_len) {
  int head_idx = blockIdx.x;
  int dim_idx = threadIdx.x;
  if (head_idx < num_heads && dim_idx < head_dim) {
    int cache_offset = (head_idx * max_seq_len + step) * head_dim + dim_idx;
    int new_offset = head_idx * head_dim + dim_idx;
    k_cache[cache_offset] = k_new[new_offset];
    v_cache[cache_offset] = v_new[new_offset];
  }
}

__global__ void attention_kernel(const float* q, const float* k_cache,
                                 const float* v_cache, float* out, int step,
                                 int num_heads, int head_dim, int max_seq_len) {
  int head_idx = blockIdx.x;
  int tid = threadIdx.x;
  float scale = 1.0f / sqrtf((float)head_dim);
  extern __shared__ float scores[];

  /**
   * 阶段 1：计算 Q @ K^T
   * 使用跨步循环：256个线程一起上，如果 step > 256，线程会继续处理下一个批次的
   * token
   */
  for (int i = tid; i <= step; i += blockDim.x) {
    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++) {
      sum += q[head_idx * head_dim + d] *
             k_cache[(head_idx * max_seq_len + i) * head_dim + d];
    }
    scores[i] = sum * scale;
  }

  // 必须同步，等待所有线程把各自负责的 token 算完，填满 scores 数组
  __syncthreads();

  /**
   * 阶段 2：Softmax 计算
   * 暂时保留用 0 号线程串行计算的逻辑。虽然慢一点，但在演示代码中能保证 100%
   * 正确。 (后续如果要追求极致性能，这里可以替换为 Block 并行归约 Reduce)
   */
  if (tid == 0) {
    float max_v = -1e20f;
    for (int i = 0; i <= step; i++) {
      max_v = fmaxf(max_v, scores[i]);
    }

    float exp_sum = 0.0f;
    for (int i = 0; i <= step; i++) {
      scores[i] = expf(scores[i] - max_v);
      exp_sum += scores[i];
    }

    for (int i = 0; i <= step; i++) {
      scores[i] /= exp_sum;
    }
  }

  // 必须同步，等待 0 号线程把归一化后的概率写回 scores
  __syncthreads();

  /**
   * 阶段 3：计算 Scores @ V
   * 同样使用跨步循环来遍历特征维度 head_dim (通常是 64 或 128)
   */
  for (int d = tid; d < head_dim; d += blockDim.x) {
    float res = 0.0f;
    for (int i = 0; i <= step; i++) {
      res += scores[i] * v_cache[(head_idx * max_seq_len + i) * head_dim + d];
    }
    out[head_idx * head_dim + d] = res;
  }
}

__global__ void apply_relu_kernel(float* out, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    out[tid] = fmaxf(0.0f, out[tid]);
  }
}

/**
 * out = ReLU(x @ weight.T)
 */
void ffn_proj_kernel(cublasHandle_t handle, int batch, float* d_output,
                     float* d_input, float* d_weight, int in_dim, int out_dim,
                     bool is_relu) {
  const float alpha = 1.0f;
  const float beta = 0.0f;

  /**
   * 行主序
   * A = d_input
   * B = d_weight
   * C = A @ B.T
   * A: [d_model] -> [batch][in_dim]
   * B: [ffn_hidden, d_model] -> [out_dim][in_dim]
   * B.T: [d_model, ffn_hidden] -> [in_dim][out_dim]
   * C: [ffn_hidden] -> [batch][out_dim]
   *
   *
   * 列主序
   * C.T = B @ A.T
   */
  cublasSgemm(handle,
              CUBLAS_OP_T,  // x @ weight -> weight @ x
              CUBLAS_OP_N,  // 输入矩阵不转置
              out_dim,      // op(A): [out_dim][in_dim]
              batch,        // B: [in_dim][batch]
              in_dim,       // in_dim
              &alpha,
              d_weight,  // <A>, shape: [in_dim][out_dim]
              in_dim,
              d_input,  // <B>, shape: [in_dim][batch]
              in_dim, &beta,
              d_output,  // C = output, shape: [out_dim][batch]
              out_dim);

  // 2. 如果需要 ReLU，启动一个简单的元素级 Kernel
  if (is_relu) {
    int total_elements = batch * out_dim;
    int threads_per_block = 256;
    int blocks_per_grid =
        (total_elements + threads_per_block - 1) / threads_per_block;

    apply_relu_kernel<<<blocks_per_grid, threads_per_block>>>(d_output,
                                                              total_elements);
    CHECK_CUDA(cudaGetLastError());
  }
}

__global__ void residual_kernel(float* x, const float* delta, int d_model) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < d_model) x[tid] += delta[tid];
}

// ============================================================================
// 3. 辅助函数：初始化、采样与核心 Forward
// ============================================================================
void init_w(float* d_ptr, size_t size, int seed_offset) {
  std::vector<float> h(size);
  std::mt19937 gen(1337 + seed_offset);
  std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
  for (size_t i = 0; i < size; ++i) h[i] = dis(gen);
  CHECK_CUDA(cudaMemcpy(d_ptr, h.data(), size * sizeof(float),
                        cudaMemcpyHostToDevice));
}

// 核心前向传播函数
void gpt_forward(cublasHandle_t cb_handle, int current_id, int step,
                 GPTConfig& config, float* d_x, float* d_tmp, float* d_attn,
                 float* d_ffn_i, float* d_ffn_o, float* d_logits, float* d_qkv,
                 float* d_q, float* d_kn, float* d_vn, float* d_emb,
                 float* d_wlm, float** d_rw1, float** d_rw2, float** d_wup,
                 float** d_wdn, float** d_wqkv, float** d_kc, float** d_vc,
                 float* d_norm_f) {
  // Embedding
  embedding_kernel<<<(config.d_model + 255) / 256, 256>>>(
      d_x, current_id, d_emb, config.d_model);
  CHECK_CUDA(cudaGetLastError());

  // 多层循环
  for (int l = 0; l < config.num_layers; l++) {
    // Attention
    rms_norm_kernel<<<1, 256>>>(d_tmp, d_x, d_rw1[l], config.d_model);
    CHECK_CUDA(cudaGetLastError());
    /**
     * QKV 投影计算 (d_qkv = d_tmp @ W_qkv)
     * m = batch_size = 1
     * n = out_features = 3 * d_model
     * k = in_features = d_model
     */
    cublas_linear(cb_handle,
                  1,                   // m (batch)
                  3 * config.d_model,  // n (out_dim)
                  config.d_model,      // k (in_dim)
                  d_tmp,               // 输入: d_tmp (RMSNorm 的输出)
                  d_wqkv[l],           // 权重: 这一层的 QKV 权重
                  nullptr,             // bias: 现代模型通常 QKV 不加 Bias
                  d_qkv);              // 输出: 存入 d_qkv buffer

    split_qkv_kernel<<<(config.d_model + 255) / 256, 256>>>(
        d_qkv, d_q, d_kn, d_vn, config.d_model);
    CHECK_CUDA(cudaGetLastError());
    update_cache_kernel<<<config.num_heads, config.head_dim>>>(
        d_kc[l], d_vc[l], d_kn, d_vn, step, config.num_heads, config.head_dim,
        config.max_seq_len);
    CHECK_CUDA(cudaGetLastError());

    int attention_threads = 256;
    // 共享内存用来存 scores，最大 2048 个 float (8KB，完全没问题)
    size_t shared_mem_size = config.max_seq_len * sizeof(float);
    attention_kernel<<<config.num_heads, attention_threads, shared_mem_size>>>(
        d_q, d_kc[l], d_vc[l], d_attn, step, config.num_heads, config.head_dim,
        config.max_seq_len);
    CHECK_CUDA(cudaGetLastError());
    residual_kernel<<<(config.d_model + 255) / 256, 256>>>(d_x, d_attn,
                                                           config.d_model);
    CHECK_CUDA(cudaGetLastError());

    // FFN
    rms_norm_kernel<<<1, 256>>>(d_tmp, d_x, d_rw2[l], config.d_model);
    CHECK_CUDA(cudaGetLastError());
    ffn_proj_kernel(cb_handle, 1, d_ffn_i, d_tmp, d_wup[l], config.d_model,
                    config.ffn_hidden, true);
    ffn_proj_kernel(cb_handle, 1, d_ffn_o, d_ffn_i, d_wdn[l], config.ffn_hidden,
                    config.d_model, false);
    residual_kernel<<<(config.d_model + 255) / 256, 256>>>(d_x, d_ffn_o,
                                                           config.d_model);
    CHECK_CUDA(cudaGetLastError());
  }

  // Final RMSNorm
  rms_norm_kernel<<<1, 256>>>(d_tmp, d_x, d_norm_f, config.d_model);
  CHECK_CUDA(cudaGetLastError());

  // Output Logits
  ffn_proj_kernel(cb_handle, 1, d_logits, d_tmp, d_wlm, config.d_model,
                  config.vocab_size, false);
}

int sample_token(const std::vector<float>& logits, float temp) {
  std::vector<float> p(logits.size());
  float sum = 0;
  for (size_t i = 0; i < logits.size(); ++i) {
    p[i] = expf(logits[i] / temp);
    sum += p[i];
  }
  float r = (float)rand() / RAND_MAX, cur = 0;
  for (size_t i = 0; i < p.size(); ++i) {
    cur += p[i] / sum;
    if (r <= cur) return i;
  }
  return logits.size() - 1;
}

/**
 * Top-K 采样函数
 * @param k 取前 k 个最高分的词，通常设为 40 或 50
 */
int sample_token_topk(const std::vector<float>& logits, float temp, int k,
                      std::mt19937& gen) {
  int vocab_size = logits.size();
  // 确保 k 不超过词表大小
  k = std::min(k, vocab_size);

  // 1. 复制一份 logits 用来找阈值
  std::vector<float> sorted_logits = logits;

  // 2. 使用 nth_element 把前 k 大的数放到数组前面（不完全排序，效率高）
  std::nth_element(sorted_logits.begin(), sorted_logits.begin() + k - 1,
                   sorted_logits.end(), std::greater<float>());

  // 第 k 个最大的值作为门槛
  float threshold = sorted_logits[k - 1];

  // 3. 计算 expf，只保留大于等于阈值的词
  std::vector<float> p(vocab_size);
  float max_logit = *std::max_element(logits.begin(), logits.end());
  float sum = 0;

  for (int i = 0; i < vocab_size; ++i) {
    if (logits[i] >= threshold) {
      p[i] = expf((logits[i] - max_logit) / temp);
    } else {
      p[i] = 0.0f;  // 被淘汰的词概率直接归零
    }
    sum += p[i];
  }

  // 4. 轮盘赌抽奖
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  float r = dis(gen);
  float cur = 0;
  for (int i = 0; i < vocab_size; ++i) {
    cur += p[i] / sum;
    if (r <= cur) return i;
  }
  return vocab_size - 1;
}

// ============================================================================
// 4. MAIN 函数
// ============================================================================
int main() {
  std::mt19937 gen(time(NULL));
  //   std::mt19937 gen(42);

  std::vector<char> vocab;
  for (char c = 'a'; c <= 'z'; ++c) vocab.push_back(c);
  for (char c = 'A'; c <= 'Z'; ++c) vocab.push_back(c);
  for (char c = '0'; c <= '9'; ++c) vocab.push_back(c);

  GPTConfig config = {62, 512, 8, 64, 2048, 1024, 12};
  int current_id = 0;
  int top_k = 10;
  float temperature = 0.8f;

  // 初始化 cuBLAS 句柄
  cublasHandle_t cb_handle;
  cublasCreate(&cb_handle);

  // 指针与分配
  float *d_x, *d_tmp, *d_attn, *d_ffn_i, *d_ffn_o, *d_logits, *d_qkv, *d_q,
      *d_kn, *d_vn, *d_emb, *d_wlm, *d_norm_f;
  CHECK_CUDA(cudaMalloc(&d_x, config.d_model * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_tmp, config.d_model * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_attn, config.d_model * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_ffn_i, config.ffn_hidden * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_ffn_o, config.d_model * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_logits, config.vocab_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_qkv, 3 * config.d_model * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_q, config.d_model * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_kn, config.d_model * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_vn, config.d_model * sizeof(float)));
  CHECK_CUDA(
      cudaMalloc(&d_emb, config.vocab_size * config.d_model * sizeof(float)));
  CHECK_CUDA(
      cudaMalloc(&d_wlm, config.vocab_size * config.d_model * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_norm_f, config.d_model * sizeof(float)));

  init_w(d_emb, config.vocab_size * config.d_model, __LINE__);
  init_w(d_wlm, config.vocab_size * config.d_model, __LINE__);
  init_w(d_norm_f, config.d_model, __LINE__);

  float** d_rw1 = new float*[config.num_layers];
  float** d_rw2 = new float*[config.num_layers];
  float** d_wup = new float*[config.num_layers];
  float** d_wdn = new float*[config.num_layers];
  float** d_wqkv = new float*[config.num_layers];
  float** d_kc = new float*[config.num_layers];
  float** d_vc = new float*[config.num_layers];

  for (int l = 0; l < config.num_layers; l++) {
    // 为每一层的每一个权重矩阵提供唯一的种子偏移
    CHECK_CUDA(cudaMalloc(&d_rw1[l], config.d_model * sizeof(float)));
    init_w(d_rw1[l], config.d_model, l * 10 + 1);

    CHECK_CUDA(cudaMalloc(&d_rw2[l], config.d_model * sizeof(float)));
    init_w(d_rw2[l], config.d_model, l * 10 + 2);

    CHECK_CUDA(cudaMalloc(&d_wup[l],
                          config.ffn_hidden * config.d_model * sizeof(float)));
    init_w(d_wup[l], config.ffn_hidden * config.d_model, l * 10 + 3);

    CHECK_CUDA(cudaMalloc(&d_wdn[l],
                          config.d_model * config.ffn_hidden * sizeof(float)));
    init_w(d_wdn[l], config.d_model * config.ffn_hidden, l * 10 + 4);

    CHECK_CUDA(cudaMalloc(&d_wqkv[l],
                          config.d_model * 3 * config.d_model * sizeof(float)));
    init_w(d_wqkv[l], config.d_model * 3 * config.d_model, l * 10 + 5);

    // KV Cache 清零即可
    size_t c_sz =
        config.num_heads * config.max_seq_len * config.head_dim * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_kc[l], c_sz));
    CHECK_CUDA(cudaMalloc(&d_vc[l], c_sz));
    cudaMemset(d_kc[l], 0, c_sz);
    cudaMemset(d_vc[l], 0, c_sz);
  }

  std::cout << "Starting Generation: " << vocab[current_id];
  std::vector<float> h_logits(config.vocab_size);

  for (int step = 0; step < 1024; step++) {
    // 调用封装好的 forward
    gpt_forward(cb_handle, current_id, step, config, d_x, d_tmp, d_attn,
                d_ffn_i, d_ffn_o, d_logits, d_qkv, d_q, d_kn, d_vn, d_emb,
                d_wlm, d_rw1, d_rw2, d_wup, d_wdn, d_wqkv, d_kc, d_vc,
                d_norm_f);

    CHECK_CUDA(cudaMemcpy(h_logits.data(), d_logits,
                          config.vocab_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    current_id = sample_token_topk(h_logits, temperature, top_k, gen);
    std::cout << vocab[current_id] << std::flush;
  }

  // 释放资源
  cudaFree(d_x);
  cudaFree(d_tmp);
  cudaFree(d_attn);
  cudaFree(d_ffn_i);
  cudaFree(d_ffn_o);
  cudaFree(d_logits);
  cudaFree(d_qkv);
  cudaFree(d_q);
  cudaFree(d_kn);
  cudaFree(d_vn);
  cudaFree(d_emb);
  cudaFree(d_wlm);
  cudaFree(d_norm_f);
  for (int l = 0; l < config.num_layers; l++) {
    cudaFree(d_rw1[l]);
    cudaFree(d_rw2[l]);
    cudaFree(d_wup[l]);
    cudaFree(d_wdn[l]);
    cudaFree(d_wqkv[l]);
    cudaFree(d_kc[l]);
    cudaFree(d_vc[l]);
  }
  delete[] d_rw1;
  delete[] d_rw2;
  delete[] d_wup;
  delete[] d_wdn;
  delete[] d_wqkv;
  delete[] d_kc;
  delete[] d_vc;

  cublasDestroy(cb_handle);

  std::cout << "\nDone." << std::endl;
  return 0;
}