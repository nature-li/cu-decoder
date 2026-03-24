import numpy as np

# 目标: C = A.T @ B
# A: m x k (2 x 3)
# B: m x n (2 x 2)
# C: k x n (3 x 2)


# 定义 A: 2 x 3
A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

# 定义 B: 2 x 2
# 为了演示，我们定义一个 2x2 的矩阵
B = np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32)

# 计算 A.T @ B
# A.T 是 3x2, B 是 2x2, 结果是 3x2
C = A.T @ B

print("A.T.shape =", A.T.shape)
print("A.T:\n", A.T)
print()

print("B.shape =", B.shape)
print("B:\n", B)
print()

print("C = A.T @ B")
print("C.shape =", C.shape)
print(C)
print()
