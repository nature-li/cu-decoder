#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

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
  // A: m x k (2 x 3), Row-Major
  const int m = 2;
  const int k = 3;
  // B: m x n (2 x 2), Row-Major
  const int n = 2;
  // C = A.T @ B -> (3 x 2) @ (2 x 2) = 3 x 2

  std::vector<float> h_A = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};  // 2x3

  std::vector<float> h_B = {7.0f, 8.0f, 9.0f, 10.0f};  // 2x2

  std::vector<float> h_C(k * n, 0.0f);  // 3x2

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  cudaMalloc(&d_A, h_A.size() * sizeof(float));
  cudaMalloc(&d_B, h_B.size() * sizeof(float));
  cudaMalloc(&d_C, h_C.size() * sizeof(float));

  cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  cublasHandle_t handle;
  cublasCreate(&handle);

  float alpha = 1.0f;
  float beta = 0.0f;

  /**
   * A: [m][k]
   * B: [m][n]
   * A.T: [k][m]
   * C: [k][n]
   *
   * 行主序
   * C = A.T @ B
   *
   * 列主序
   * C.T = B.T @ A
   */
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
              n,  // 列主序下 op(A) 的 row
              k,  // 列主序下 op(B) 的 col
              m,  // k
              &alpha,
              d_B,  // <A> 传入 B的转置, 列主序下 <A> 的 shape: [n][m]
              n,    // lda: 列主序下参数 <A> 的 row
              d_A,  // <B> 参数传入 A，A 的 shape 为 [k][m],
              k,    // ldb: 列主序下参数 <B> 的 row
              &beta,
              d_C,  // <C> 传入 C 的转置，列主序下 shape: [n][k]
              n     // ldc: 列主序下参数 <C> 的 row
  );

  cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  print_matrix_row_major(h_A, m, k, "A");
  print_matrix_row_major(h_B, m, n, "B");
  print_matrix_row_major(h_C, k, n, "C = A.T @ B");

  cublasDestroy(handle);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}