import numpy as np


def dot(a, b):
    row = a.shape[0]
    col = b.shape[1]
    p = a.shape[1]
    if p != b.shape[0]:
        raise ValueError(
            "The number of columns of a must be equal to the number of rows of b")

    c = np.zeros((row, col), dtype=int)
    for i in range(row):
        for j in range(col):
            for k in range(p):
                c[i][j] += a[i][k] * b[k][j]
    return c


def dot_from_numpy(a, b):
    return np.dot(a, b)


if __name__ == '__main__':
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = np.array([[9, 10], [11, 12]])

    print(A)
    print(B)
    print(C)
    print("")
    print(f"dot(A, B) = \n {dot(A, B)}")
    print(f"dot_from_numpy(A, B) = \n {dot_from_numpy(A, B)}")

    # 定义了一些矩阵运算，并展示了分配律和结合律的示例
    print("-"*10 + " 分配律 " + "-"*10)
    print(f"dot(A, B + C) = \n{dot(A, B + C)}")  # 展示 A 与 (B + C) 的点乘结果
    # 展示 A 与 B 的点乘结果加上 A 与 C 的点乘结果
    print(f"dot(A, B) + dot(A, C) = \n{dot(A, B) + dot(A, C)}")

    print("-"*10 + " 结合律 " + "-"*10)
    # 展示 (A 与 B 的点乘) 与 C 的点乘结果
    print(f"dot(dot(A, B), C) = \n{dot(dot(A, B), C)}")
    # 展示 A 与 (B 与 C 的点乘) 的结果
    print(f"dot(A, dot(B, C)) = \n{dot(A, dot(B, C))}")

    print("-"*10 + " (AB)⊤=B⊤A⊤ " + "-"*10)
    # 定义矩阵A和B
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    # 计算 (AB)⊤
    AB_transpose = (A @ B).T

    # 计算 B⊤A⊤
    B_transpose_A_transpose = B.T @ A.T

    # 验证是否相等
    print("(AB)⊤ =\n", AB_transpose)
    print("B⊤A⊤ =\n", B_transpose_A_transpose)
