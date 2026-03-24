import numpy as np


# A: m x k
A = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
], dtype=np.float32)  # 2 x 3

# B: n x k
B = np.array([
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
], dtype=np.float32)  # 2 x 3

# 目标: C = A @ B.T
C = A @ B.T

print("A.shape =", A.shape)
print(A)
print()

print("B.shape =", B.shape)
print(B)
print()

print("C = A @ B.T")
print("C.shape =", C.shape)
print(C)
print()