#include <cublas_v2.h>
#include <cuda_runtime.h>

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
  // A = [[1,2,3],
  //      [4,5,6]]
  const int m = 2;
  const int k = 3;
  std::vector<float> h_A = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // B = [[7, 8],
  //      [9,10],
  //      [11,12]]
  const int n = 2;
  std::vector<float> h_B = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

  // C = A @ B -> 2x2
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
   * C = A @ B => C.T = B.T @ A.T
   *
   * A 传入 B.T
   * B 传入 A.T
   * D 传入 C.T
   */
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              n,  // 列主序下 A 的 row
              m,  // 列主序下 B 的 col
              k,  // k
              &alpha,
              d_B,  // A 传入 B.T, 列主序下 A 的 shape: [n][k]
              n,    // 列主序下参数 <A> 的 row
              d_A,  // B 传入 A.T, 列主序下 B 的 shape: [k][m]
              k,    // 列主序下参数 <B> 的 row
              &beta,
              d_C,  // C 传入 C.T, 列主序下 C 的 shape: [n][m]
              n);   // 序主序下参数 <C> 的 row

  cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float),
             cudaMemcpyDeviceToHost);

  print_matrix_row_major(h_A, m, k, "A");
  print_matrix_row_major(h_B, k, n, "B");
  print_matrix_row_major(h_C, m, n, "C = A @ B");

  std::vector<float> expected = {58.0f, 64.0f, 139.0f, 154.0f};
  print_matrix_row_major(expected, m, n, "Expected");

  bool ok = true;
  for (size_t i = 0; i < h_C.size(); ++i) {
    if (std::abs(h_C[i] - expected[i]) > 1e-5f) {
      ok = false;
      break;
    }
  }

  std::cout << "match = " << (ok ? "true" : "false") << std::endl;

  cublasDestroy(handle);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}