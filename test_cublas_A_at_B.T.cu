#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

/**
 * 按 row-major 方式打印矩阵
 *
 * 功能:
 *   将一维数组按 row-major 的二维矩阵形式打印出来，便于观察结果。
 *
 * 参数:
 *   mat:  保存矩阵数据的一维数组
 *   rows: 矩阵行数
 *   cols: 矩阵列数
 *   name: 矩阵名字
 */
static void print_matrix_row_major(const std::vector<float>& mat, int rows,
                                   int cols, const char* name) {
  std::cout << name << " (" << rows << "x" << cols << "):" << std::endl;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << std::setw(8) << mat[i * cols + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  const int m = 2;
  const int k = 3;
  const int n = 2;

  // A: m x k，row-major
  std::vector<float> h_A = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // B: n x k，row-major
  std::vector<float> h_B = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  // C: m x n，row-major
  std::vector<float> h_C(m * n, 0.0f);

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  cudaMalloc(&d_A, h_A.size() * sizeof(float));
  cudaMalloc(&d_B, h_B.size() * sizeof(float));
  cudaMalloc(&d_C, h_C.size() * sizeof(float));

  cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, h_C.data(), h_C.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;

  /**
   * A: [m][k]
   * B: [n][k]
   * B.T: [k][n]
   * C: [m][n]
   * 行主序
   * C = A @ B.T
   *
   * 列主邓
   * C.T = B @ A.T
   */
  cublasSgemm(
      handle, CUBLAS_OP_T, CUBLAS_OP_N,
      n,  // 列主序下 op(A) 的 row
      m,  // 列主序下 op(B) 的 col
      k,  // k
      &alpha,
      d_B,  // A 传入 B (不要转置), 所以要设置 CUBLAS_OP_T, 序主序下 d_B 的 shape: [k][n]
      k,    // 参数 <A> 的 row
      d_A,  // B 传入 A 的转置，所以设置为 CUBLAS_OP_N, 列序下 shape: [k][m]
      k,    // 参数 <B> 的 row
      &beta,
      d_C,  // C 传入 C 的转置, 列序下 shape 为 [n][m]
      n);   // 参数 <C> 的 row

  cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  print_matrix_row_major(h_A, m, k, "A");
  print_matrix_row_major(h_B, n, k, "B");
  print_matrix_row_major(h_C, m, n, "C = A @ B.T");

  cublasDestroy(handle);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}