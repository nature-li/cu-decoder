#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "comm.h"

// 超参数
const int B = 1;
const int NUM_HEADS = 4;
const int HEAD_DIM = 8;  // 32 / 4 = 8
const int D_MODEL = 32;
const int MAX_SEQ_LEN = 128;

// kernels

/**
 * 简单的线性层实现（这里只为了演示只做 Q/K/V 的拆分逻辑）
 */
__global__ void split_qkv_kernel(const float* qkv, float* q, float* k_new,
                                 float* v_new) {
  /**
   * D_MODEL * 3 = 96
   * tid 范围: [0, 95]
   */
  int tid = threadIdx.x;
  if (tid < D_MODEL) {
    q[tid] = qkv[tid];
  } else if (tid < D_MODEL * 2) {
    k_new[tid - D_MODEL] = qkv[tid];
  } else {
    v_new[tid - D_MODEL * 2] = qkv[tid];
  }
}

/**
 * 将新的 k v 写入 Cache
 */
__global__ void update_cache_kernel(float* k_cache, float* v_cache,
                                    const float* k_new, const float* v_new,
                                    int step) {
  int head_idx = blockIdx.x;
  int dim_idx = threadIdx.x;
  if (head_idx < NUM_HEADS && dim_idx < HEAD_DIM) {
    /**
     * 逻辑等价于: k_cache[head_idx][step][dim_idx]
     * * 物理寻址推导:
     * 1. 要跳过前面的 heads: head_idx * (单个 head 的总容量)
     * 其中 单个 head 容量 = MAX_SEQ_LEN * HEAD_DIM
     * 2. 在当前 head 内跳过前面的 steps: step * (单个 step 的容量)
     * 其中 单个 step 容量 = HEAD_DIM
     * 3. 在当前 step 内定位具体维度: + dim_idx
     *
     * 最终公式:
     * cache_offset = head_idx * (MAX_SEQ_LEN * HEAD_DIM) + step * HEAD_DIM +
     * dim_idx 提取公因子后即为: (head_idx * MAX_SEQ_LEN + step) * HEAD_DIM +
     * dim_idx
     * * 约束条件 (Boundary Conditions):
     * 0 <= head_idx < NUM_HEADS
     * 0 <= step     < MAX_SEQ_LEN  // 这里的 step 就是当前 Token 的位置索引
     * 0 <= dim_idx  < HEAD_DIM
     */
    int cache_offset = (head_idx * MAX_SEQ_LEN + step) * HEAD_DIM + dim_idx;
    int new_offset = head_idx * HEAD_DIM + dim_idx;
    k_cache[cache_offset] = k_new[new_offset];
    v_cache[cache_offset] = v_new[new_offset];
  }
}

/**
 * Attention Kernel (Q * K_cache^T -> Softmax -> V_cache)
 */
__global__ void attention_kernel(const float* q, const float* k_cache,
                                 const float* v_cache, float* out, int step) {
  int head_idx = blockIdx.x;
  int tid = threadIdx.x;
  float scale = 1.0f / sqrtf((float)HEAD_DIM);

  // 存储当前 head 对历史所有 token 的 score
  __shared__ float scores[MAX_SEQ_LEN];

  /**
   * 计算 Score: Q @ K_cache.T
   *
   * 每个线程负责处理一个历史 Token
   * for tid in range(step):
   *    for dim_idx in range(HEAD_DIM):
   *        data[tid * HEAD_DIM + dim_idx]
   */
  if (tid <= step) {
    float sum = 0.0f;
    for (int dim_idx = 0; dim_idx < HEAD_DIM; dim_idx++) {
      // 读取第 head_idx 个头的第 dim_idx 个维度
      float q_val = q[head_idx * HEAD_DIM + dim_idx];
      float k_val =
          k_cache[(head_idx * MAX_SEQ_LEN + tid) * HEAD_DIM + dim_idx];
      sum += q_val * k_val;
    }
    scores[tid] = sum * scale;
  }
  __syncthreads();

  // 简化的 Softmax (仅由线程 0 计算并写回 shared memory)
  if (tid == 0) {
    float max_v = -1e20f;
    for (int i = 0; i <= step; i++) {
      max_v = max(max_v, scores[i]);
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
  __syncthreads();

  /**
   * Score @ V_cache
   *
   * 每个线程负责计算结果向量中的一个维度（Dimension）。
   *
   * for dim_idx in range(HEAD_DIM):
   *    for tid in range(step):
   *        data[i * HEAD_DIM + dim_idx]
   */
  if (tid < HEAD_DIM) {
    float res = 0.0f;
    for (int i = 0; i <= step; i++) {
      float v_val = v_cache[(head_idx * MAX_SEQ_LEN + i) * HEAD_DIM + tid];
      res += scores[i] * v_val;
    }
    out[head_idx * HEAD_DIM + tid] = res;
  }
}

int main() {
  // 显存分配
  float* d_qkv = nullptr;
  float* d_q = nullptr;
  float* d_k_new = nullptr;
  float* d_v_new = nullptr;
  float* d_k_cache = nullptr;
  float* d_v_cache = nullptr;
  float* d_attn_out = nullptr;

  CHECK_CUDA(cudaMalloc(&d_qkv, sizeof(float) * D_MODEL * 3));
  CHECK_CUDA(cudaMalloc(&d_q, sizeof(float) * D_MODEL));
  CHECK_CUDA(cudaMalloc(&d_k_new, sizeof(float) * D_MODEL));
  CHECK_CUDA(cudaMalloc(&d_v_new, sizeof(float) * D_MODEL));
  CHECK_CUDA(cudaMalloc(&d_k_cache,
                        sizeof(float) * NUM_HEADS * MAX_SEQ_LEN * HEAD_DIM));
  CHECK_CUDA(cudaMalloc(&d_v_cache,
                        sizeof(float) * NUM_HEADS * MAX_SEQ_LEN * HEAD_DIM));
  CHECK_CUDA(cudaMalloc(&d_attn_out, sizeof(float) * D_MODEL));

  // 模拟初始化数据
  std::vector<float> h_qkv(D_MODEL * 3, 1.0f);
  CHECK_CUDA(cudaMemcpy(d_qkv, h_qkv.data(), sizeof(float) * D_MODEL * 3,
                        cudaMemcpyHostToDevice));

  std::cout << "Starting Mini GPT Decoder inference..." << std::endl;
  // --- 推理循环 (Step-by-step) ---
  for (int step = 0; step < 5; ++step) {
    // 1. Split Q, K, V
    split_qkv_kernel<<<1, D_MODEL * 3>>>(d_qkv, d_q, d_k_new, d_v_new);

    // 2. Update KV Cache
    update_cache_kernel<<<NUM_HEADS, HEAD_DIM>>>(d_k_cache, d_v_cache, d_k_new,
                                                 d_v_new, step);

    // 3. Attention
    attention_kernel<<<NUM_HEADS, HEAD_DIM>>>(d_q, d_k_cache, d_v_cache,
                                              d_attn_out, step);

    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "Step " << step << " completed." << std::endl;
  }

  cudaFree(d_qkv);
  cudaFree(d_q);
  cudaFree(d_k_new);
  cudaFree(d_v_new);
  cudaFree(d_k_cache);
  cudaFree(d_v_cache);
  cudaFree(d_attn_out);
  std::cout << "Done." << std::endl;
  return 0;
}