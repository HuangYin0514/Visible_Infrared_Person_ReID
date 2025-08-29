import torch


def DE_func(t, q, dq):
    """
    二阶系统右端项: \ddot{q} = f(q, dq, t)
    示例: M \ddot{q} + C dq + K q = 0
    """
    K = torch.eye(4, device=q.device)  # 刚度矩阵
    C = 0.1 * torch.eye(4, device=q.device)  # 阻尼矩阵
    ddq = -q @ K.T - dq @ C.T  # 加速度
    return ddq


def dynamics(t, y):
    """
    一阶系统形式
    输入 y: (batch, N), 其中前N/2维是 q, 后N/2维是 dq
    输出 dy: (batch, N)
    """
    batch = y.shape[0]
    q, dq = y[:, :4], y[:, 4:]
    ddq = DE_func(t, q, dq)  # (batch, 4)
    d_q_dq = torch.cat([dq, ddq], dim=1)
    return d_q_dq


def euler_step(t, y, h):
    """
    一步欧拉法
    """
    k1 = dynamics(t, y)
    return y + h * k1


def rk4_step(t, y, h):
    """
    一步 RK4
    """
    k1 = dynamics(t, y)
    k2 = dynamics(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = dynamics(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = dynamics(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve(q0, dq0, t0, h, steps):
    """
    积分器
    输入:
        q0: (batch, 4)
        dq0: (batch, 4)
    输出:
        traj_q: (batch, steps+1, 4)
        traj_v: (batch, steps+1, 4)
    """
    y = torch.cat([q0, dq0], dim=1)  # (batch, 8)
    traj_q = [q0]
    traj_v = [dq0]
    t = t0
    for _ in range(steps):
        y = euler_step(t, y, h)
        traj_q.append(y[:, :4])
        traj_v.append(y[:, 4:])
        t += h
    return torch.stack(traj_q, dim=1), torch.stack(traj_v, dim=1)


batch_size = 2
q0 = torch.randn(batch_size, 4)  # 初始位置
dq0 = torch.zeros(batch_size, 4)  # 初始速度
t0, h, steps = 0.0, 0.01, 100

traj_q, traj_v = solve(q0, dq0, t0, h, steps)

print(traj_q.shape)  # (2, 101, 4)
print(traj_v.shape)  # (2, 101, 4)
