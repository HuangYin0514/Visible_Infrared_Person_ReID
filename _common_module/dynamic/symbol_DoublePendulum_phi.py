from sympy import *

if __name__ == "__main__":

    l = Matrix(symbols("l_0:2"))
    q = Matrix(symbols("q_0:4"))
    dq = Matrix(symbols("dq_0:4"))
    phi = Matrix(
        [
            q[0] ** 2 + q[1] ** 2 - l[0] ** 2,
            (q[0] - q[2]) ** 2 + (q[1] - q[3]) ** 2 - l[1] ** 2,
        ]
    )

    print("phi =")
    print(phi.__repr__())

    phi_q = phi.jacobian(q)
    print("\nphi_q = dphi/dq =")
    print(phi_q.__repr__())

    phi_q_dq = phi_q * dq
    print("\nphi_q * dq =")
    print(phi_q_dq.__repr__())

    phi_q_dq_q = phi_q_dq.jacobian(q)
    print("\nJacobian(phi_q * dq, q) =")
    print(phi_q_dq_q.__repr__())

    phi_q_dq_q_dq = phi_q_dq_q * dq
    print("\nSimplified (phi_q_dq_q * dq) =")
    print(phi_q_dq_q_dq.__repr__())

    print("\nphi_q_dq_q * dq =")
    print(simplify(phi_q_dq_q_dq).__repr__())
