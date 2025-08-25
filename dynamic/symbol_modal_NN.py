from sympy import *

if __name__ == "__main__":

    ################################
    # 定义符号
    ################################
    W = Matrix(symbols([" ".join(f"w_{i}{j}" for i in range(4) for j in range(4))])).reshape(4, 4)
    q = Matrix([symbols(f"q_{i}") for i in range(4)])
    q_inf = Matrix([symbols(f"q_inf{i}") for i in range(4)])
    dq = Matrix([symbols(f"dq{i}") for i in range(4)])

    print("=" * 10)
    print(W.__repr__())
    print(q.__repr__())
    print(q_inf.__repr__())
    print(dq.__repr__())

    ################################
    # 定义约束方程
    ################################
    phi = Matrix(
        [
            (W.T * q).applyfunc(exp),
        ]
    )

    print("=" * 10)
    print("phi =")
    print(phi.__repr__())

    ################################
    # 计算相关变量
    ################################
    phi_q = phi.jacobian(q)
    phi_q_dq = phi_q * dq
    phi_q_dq_q = phi_q_dq.jacobian(q)
    phi_q_dq_q_dq = phi_q_dq_q * dq

    print("=" * 10 + "\n phi_q = dphi/dq = \n {}".format(simplify(phi_q).__repr__()))
    # print("=" * 10 + "\n phi_q_dq = \n {}".format(simplify(phi_q_dq).__repr__()))
    # print("=" * 10 + "\n phi_q_dq_q = \n {}".format(simplify(phi_q_dq_q).__repr__()))
    # print("=" * 10 + "\n phi_q_dq_q_dq = \n {}".format(simplify(phi_q_dq_q_dq).__repr__()))

    ################################
    # 验证
    ################################
    # 验证 phi_q
    exp_qW = (W.T * q).applyfunc(exp)
    exp_mat = exp_qW * ones(1, 4)  # (n,1) @ (1,n) = (n,n)
    expr = W.T.multiply_elementwise(exp_mat)
    print("=" * 10 + "\n expr = \n {}".format(simplify(expr).__repr__()))
    print("=" * 10 + "\n valid = \n {}".format(simplify(expr - phi_q).__repr__()))

    # 验证 phi_q_dq_q_dq
    # dqW = dq.T * W  # 形状 (1,4)
    # sq = dqW.applyfunc(lambda x: x**2).T  # 形状 (4,1)
    # exp_part = (q.T * W).T.applyfunc(exp)  # 形状 (4,1)
    # expr = sq.multiply_elementwise(exp_part)
    # print("=" * 10 + "\n expr = \n {}".format(simplify(expr).__repr__()))
    # print("=" * 10 + "\n valid = \n {}".format(simplify(expr - phi_q_dq_q_dq).__repr__()))


# def abandoned():
#     ################################
#     # phi_q_dq_q_dq 矩阵分解
#     # 思路: A*X =B -> X = A-1 * B
#     ################################
#     print("=" * 10)
#     item = (W.T).inv() * phi_q_dq_q_dq
#     print("item = A-1 * B = \n {}".format(simplify(item).__repr__()))

#     ################################
#     # 验证
#     ################################
#     item_0 = W.T
#     item_1 = dq.applyfunc(lambda x: x**2)
#     item_2 = q.applyfunc(exp)
#     W_dq_q = item_0 * (item_1.multiply_elementwise(item_2))
#     print("=" * 10)
#     print("W.T * dq**2 * exp(q) = \n {}".format(simplify(W_dq_q).__repr__()))
#     print("valid = W_dq_q - phi_q_dq_q_dq = \n {}".format(simplify(W_dq_q - phi_q_dq_q_dq).__repr__()))
