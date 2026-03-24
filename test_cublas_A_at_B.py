import numpy as np

A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
B = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32)
C = A @ B

print("A.shape:", A.shape)
print(A)

print("B.shape:", B.shape)
print(B)

print("C.shape:", C.shape)
print(C)
