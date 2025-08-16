import numpy as np
import pychrono as chrono

# ------------------------------------------------------------------------------
# 1. Create the Chrono physical system
# ------------------------------------------------------------------------------

sys = chrono.ChSystemSMC()
sys.SetGravitationalAcceleration(chrono.ChVector3d(0, -9.81, 0))  # Global gravity

# ------------------------------------------------------------------------------
# 2. Create the ground body (fixed, at global origin)
# ------------------------------------------------------------------------------

ground = chrono.ChBody()
ground.SetFixed(True)
ground.SetPos(chrono.ChVector3d(0, 0, 0))  # Explicit in global frame
sys.Add(ground)

# ------------------------------------------------------------------------------
# 3. Create the pendulum body (initial position, orientation, velocity all in global)
# ------------------------------------------------------------------------------

pend1 = chrono.ChBody()

# Set global position of pendulum CG
pend1.SetPos(chrono.ChVector3d(0, -0.5, 0))  # 0.5m below joint in Y (gravity) direction

# Set mass and inertia
pend1.SetMass(0.79)
pend1.SetInertiaXX(chrono.ChVector3d(0.01, 0.01, 0.0658))  # principal inertias
# Put dummy inertia in X and Y direction. Otherwise it gives Inf solution

# Set angular velocity in global frame (around Z-axis)
pend1.SetAngVelParent(chrono.ChVector3d(0, 0, 0.01))  # radians/sec in global Z
sys.Add(pend1)

# ------------------------------------------------------------------------------
# 4. Revolute joint between ground and pend1 (at global origin, around global Z)
# ------------------------------------------------------------------------------

# Frame at joint location (global coordinates)
joint_frame = chrono.ChFramed(chrono.ChVector3d(0, 0, 0))  # Revolute joint at origin
joint1 = chrono.ChLinkRevolute()
joint1.Initialize(ground, pend1, joint_frame)

sys.Add(joint1)

# Simulation parameters
end_time = 50.0  # seconds
step_size = 0.01
time = 0.0

# Initialize storage lists
time_data = []
pend1_x, pend1_y, pend1_z = [], [], []
pend2_x, pend2_y, pend2_z = [], [], []

# Run the simulation
while time < end_time:
    sys.DoStepDynamics(step_size)

    # Get positions
    pos1 = pend1.GetPos()

    # Append data to lists
    time_data.append(time)

    pend1_x.append(pos1.x)
    pend1_y.append(pos1.y)
    pend1_z.append(pos1.z)

    time += step_size

# Prepare data dictionary
mat_data = {
    "time": np.array(time_data, dtype=np.float64),
    "pend1_x": np.array(pend1_x, dtype=np.float64),
    "pend1_y": np.array(pend1_y, dtype=np.float64),
    "pend1_z": np.array(pend1_z, dtype=np.float64),
    "pend2_x": np.array(pend2_x, dtype=np.float64),
    "pend2_y": np.array(pend2_y, dtype=np.float64),
    "pend2_z": np.array(pend2_z, dtype=np.float64),
}

print(mat_data)
