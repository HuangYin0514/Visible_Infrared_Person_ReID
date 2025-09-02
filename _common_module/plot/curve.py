import matplotlib.pyplot as plt
import numpy as np

# 定义函数
x_min = 0.001
x_max = 10
x = np.linspace(x_min, x_max, 1000)
y = 1 / (1 + x)

# 绘制曲线
plt.figure(figsize=(6, 6))
plt.plot(x, y)

plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
