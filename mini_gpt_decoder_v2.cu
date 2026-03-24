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
  if (tid <= step) {
    float sum = 0.0f;
    for (int d = 0; d < head_dim; d++)
      sum += q[head_idx * head_dim + d] *
             k_cache[(head_idx * max_seq_len + tid) * head_dim + d];
    scores[tid] = sum * scale;
  }
  __syncthreads();
  if (tid == 0) {
    float max_v = -1e20f;
    for (int i = 0; i <= step; i++) max_v = fmaxf(max_v, scores[i]);
    float exp_sum = 0.0f;
    for (int i = 0; i <= step; i++) {
      scores[i] = expf(scores[i] - max_v);
      exp_sum += scores[i];
    }
    for (int i = 0; i <= step; i++) scores[i] /= exp_sum;
  }
  __syncthreads();
  if (tid < head_dim) {
    float res = 0.0f;
    for (int i = 0; i <= step; i++)
      res += scores[i] * v_cache[(head_idx * max_seq_len + i) * head_dim + tid];
    out[head_idx * head_dim + tid] = res;
  }
}

__global__ void ffn_proj_kernel(float* out, const float* x, const float* weight,
                                int in_dim, int out_dim, bool is_relu) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  extern __shared__ float s_reduce[];
  float sum = 0.0f;
  for (int i = tid; i < in_dim; i += blockDim.x)
    sum += x[i] * weight[row * in_dim + i];
  s_reduce[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) s_reduce[tid] += s_reduce[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[row] = is_relu ? fmaxf(0.0f, s_reduce[0]) : s_reduce[0];
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
void gpt_forward(int current_id, int step, GPTConfig& config, float* d_x,
                 float* d_tmp, float* d_attn, float* d_ffn_i, float* d_ffn_o,
                 float* d_logits, float* d_qkv, float* d_q, float* d_kn,
                 float* d_vn, float* d_emb, float* d_wlm, float** d_rw1,
                 float** d_rw2, float** d_wup, float** d_wdn, float** d_kc,
                 float** d_vc) {
  // Embedding
  embedding_kernel<<<(config.d_model + 255) / 256, 256>>>(
      d_x, current_id, d_emb, config.d_model);

  // 多层循环
  for (int l = 0; l < config.num_layers; l++) {
    // Attention
    rms_norm_kernel<<<1, 256>>>(d_tmp, d_x, d_rw1[l], config.d_model);
    init_w(d_qkv, 3 * config.d_model, l * 10 + 1);  // 模拟投影
    split_qkv_kernel<<<1, 256>>>(d_qkv, d_q, d_kn, d_vn, config.d_model);
    update_cache_kernel<<<config.num_heads, config.head_dim>>>(
        d_kc[l], d_vc[l], d_kn, d_vn, step, config.num_heads, config.head_dim,
        config.max_seq_len);
    attention_kernel<<<config.num_heads, config.max_seq_len,
                       config.max_seq_len * sizeof(float)>>>(
        d_q, d_kc[l], d_vc[l], d_attn, step, config.num_heads, config.head_dim,
        config.max_seq_len);
    residual_kernel<<<(config.d_model + 255) / 256, 256>>>(d_x, d_attn,
                                                           config.d_model);

    // FFN
    rms_norm_kernel<<<1, 256>>>(d_tmp, d_x, d_rw2[l], config.d_model);
    ffn_proj_kernel<<<config.ffn_hidden, 256, 256 * sizeof(float)>>>(
        d_ffn_i, d_tmp, d_wup[l], config.d_model, config.ffn_hidden, true);
    ffn_proj_kernel<<<config.d_model, 256, 256 * sizeof(float)>>>(
        d_ffn_o, d_ffn_i, d_wdn[l], config.ffn_hidden, config.d_model, false);
    residual_kernel<<<(config.d_model + 255) / 256, 256>>>(d_x, d_ffn_o,
                                                           config.d_model);
  }

  // Output Logits
  ffn_proj_kernel<<<config.vocab_size, 256, 256 * sizeof(float)>>>(
      d_logits, d_x, d_wlm, config.d_model, config.vocab_size, false);
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

  std::vector<char> vocab;
  for (char c = 'a'; c <= 'z'; ++c) vocab.push_back(c);
  for (char c = 'A'; c <= 'Z'; ++c) vocab.push_back(c);
  for (char c = '0'; c <= '9'; ++c) vocab.push_back(c);

  GPTConfig config = {62, 512, 8, 64, 2048, 1024, 12};
  int current_id = 0;
  int top_k = 10;
  float temperature = 0.8f;

  // 指针与分配
  float *d_x, *d_tmp, *d_attn, *d_ffn_i, *d_ffn_o, *d_logits, *d_qkv, *d_q,
      *d_kn, *d_vn, *d_emb, *d_wlm;
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

  init_w(d_emb, config.vocab_size * config.d_model, 0);
  init_w(d_wlm, config.vocab_size * config.d_model, 0);

  float** d_rw1 = new float*[config.num_layers];
  float** d_rw2 = new float*[config.num_layers];
  float** d_wup = new float*[config.num_layers];
  float** d_wdn = new float*[config.num_layers];
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

    // KV Cache 依然清零即可
    size_t c_sz =
        config.num_heads * config.max_seq_len * config.head_dim * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_kc[l], c_sz));
    CHECK_CUDA(cudaMalloc(&d_vc[l], c_sz));
    cudaMemset(d_kc[l], 0, c_sz);
    cudaMemset(d_vc[l], 0, c_sz);
  }

  std::cout << "Starting Generation: " << vocab[current_id];
  std::vector<float> h_logits(config.vocab_size);

  for (int step = 0; step < 30; step++) {
    // 调用封装好的 forward
    gpt_forward(current_id, step, config, d_x, d_tmp, d_attn, d_ffn_i, d_ffn_o,
                d_logits, d_qkv, d_q, d_kn, d_vn, d_emb, d_wlm, d_rw1, d_rw2,
                d_wup, d_wdn, d_kc, d_vc);

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
  for (int l = 0; l < config.num_layers; l++) {
    cudaFree(d_rw1[l]);
    cudaFree(d_rw2[l]);
    cudaFree(d_wup[l]);
    cudaFree(d_wdn[l]);
    cudaFree(d_kc[l]);
    cudaFree(d_vc[l]);
  }
  delete[] d_rw1;
  delete[] d_rw2;
  delete[] d_wup;
  delete[] d_wdn;
  delete[] d_kc;
  delete[] d_vc;

  std::cout << "\nDone." << std::endl;
  return 0;
}