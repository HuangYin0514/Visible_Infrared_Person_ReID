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
ground.SetPos(chrono.ChVector3d(0, 0, 0))
sys.Add(ground)

# ------------------------------------------------------------------------------
# 3. Create the pendulum body
# ------------------------------------------------------------------------------
pend1 = chrono.ChBody()
pend1.SetPos(chrono.ChVector3d(0, -0.5, 0))
pend1.SetMass(0.79)
pend1.SetInertiaXX(chrono.ChVector3d(0.01, 0.01, 0.0658))
pend1.SetAngVelParent(chrono.ChVector3d(0.02, 0.03, 0.01))
sys.Add(pend1)

# ------------------------------------------------------------------------------
# 4. Revolute joint
# ------------------------------------------------------------------------------
joint_frame = chrono.ChFramed(chrono.ChVector3d(0, 0, 0))
joint1 = chrono.ChLinkRevolute()
joint1.Initialize(ground, pend1, joint_frame)
sys.Add(joint1)

# ------------------------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------------------------
end_time = 50.0
step_size = 0.01
time = 0.0

# Run simulation and print directly
print("time pend1_x pend1_y pend1_z")
while time < end_time:
    sys.DoStepDynamics(step_size)
    pos1 = pend1.GetPos()
    print(f"{time:.4f} {pos1.x:.6f} {pos1.y:.6f} {pos1.z:.6f}")
    time += step_size
