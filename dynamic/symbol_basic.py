from sympy import *

if __name__ == "__main__":

    # l = Matrix(symbols(r"self.l[0:2]"))
    # q = Matrix(symbols(r"q[\:\,0:4]"))
    # dq = Matrix(symbols(r"dq[\:\,0:4]"))

    l = Matrix(symbols("l_0:2"))
    q = Matrix(symbols("q_0:4"))
    dq = Matrix(symbols("dq_0:4"))

    # *号矩阵乘法
    # 元素逐一相乘 dq.multiply_elementwise(dq)
    dqdq = dq.T * dq
    print(dqdq.__repr__())

    dqdq = dq * dq.T
    print(dqdq.__repr__())

    dqdq = dq.multiply_elementwise(dq)
    print(dqdq.__repr__())
