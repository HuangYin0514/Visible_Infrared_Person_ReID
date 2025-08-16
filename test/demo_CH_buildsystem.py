import numpy as np
import pychrono as chrono

# ------------------------------------------------------------------------------
# 1. Create the Chrono physical system
# ------------------------------------------------------------------------------
sys = chrono.ChSystemSMC()
sys.SetGravitationalAcceleration(chrono.ChVector3d(0, -9.81, 0))

# ------------------------------------------------------------------------------
# 2. Create the ground body
# ------------------------------------------------------------------------------
ground = chrono.ChBody()
ground.SetFixed(True)
sys.Add(ground)

# ------------------------------------------------------------------------------
# 3. Create the pendulum body
# ------------------------------------------------------------------------------
pend1 = chrono.ChBody()
pend1.SetMass(0.79)
pend1.SetInertiaXX(chrono.ChVector3d(0.01, 0.01, 0.0658))

# 设置初始位置：长度 0.5 m，初始偏角 0.2 rad
pend_length = 0.5
initial_angle = 0.2  # 弧度
pend1.SetPos(chrono.ChVector3d(pend_length * np.sin(initial_angle), -pend_length * np.cos(initial_angle), 0))
sys.Add(pend1)

# ------------------------------------------------------------------------------
# 4. Revolute joint at pendulum top (origin)
# ------------------------------------------------------------------------------
joint1 = chrono.ChLinkRevolute()
# Initialize: ground, pendulum, pivot位置（原点），旋转轴(0,0,1)
joint1.Initialize(ground, pend1, chrono.ChVectorD(0, 0, 0), chrono.ChVectorD(0, 0, 1))
sys.Add(joint1)

# ------------------------------------------------------------------------------
# 5. Simulation parameters
# ------------------------------------------------------------------------------
end_time = 10.0
step_size = 0.01
time = 0.0

# ------------------------------------------------------------------------------
# 6. Run simulation and print
# ------------------------------------------------------------------------------
print("time pend1_x pend1_y pend1_z")
while time < end_time:
    sys.DoStepDynamics(step_size)
    pos = pend1.GetPos()
    print(f"{time:.4f} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    time += step_size
