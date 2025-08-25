from sympy import *

if __name__ == "__main__":

    w = Matrix(symbols("w_0:16")).reshape(4, 4)
    q_rgb = Matrix(symbols("q_rgb_0:4"))
    q_inf = Matrix(symbols("q_inf_0:4"))
    dq_rgb = Matrix(symbols("dq_0:4"))

    # phi = Matrix(
    #     [
    #         w.T * q_rgb - w.T * q_inf,
    #     ]
    # )
    # phi = phi.applyfunc(tanh)
    phi = w.T * q_rgb.applyfunc(exp)
    # phi = phi_linear.applyfunc(tanh)

    print("phi =")
    print(phi.__repr__())
    # print(simplify(phi).__repr__())

    phi_q = phi.jacobian(q_rgb)
    print("\nphi_q = dphi/dq =")
    print(phi_q.__repr__())

    phi_q_dq = phi_q * dq_rgb
    print("\nphi_q * dq =")
    print(phi_q_dq.__repr__())

    phi_q_dq_q = phi_q_dq.jacobian(q_rgb)
    print("\nJacobian(phi_q * dq, q) =")
    print(phi_q_dq_q.__repr__())

    phi_q_dq_q_dq = phi_q_dq_q * dq_rgb
    print("\nSimplified (phi_q_dq_q * dq) =")
    print(phi_q_dq_q_dq.__repr__())

    print("\nphi_q_dq_q * dq =")
    print(phi_q_dq_q_dq.__repr__())
    # print(simplify(phi_q_dq_q_dq).__repr__())
