import pychrono as chrono

# Create system

sys = chrono.ChSystemNSC()
sys.SetGravitationalAcceleration(chrono.ChVector3d(0, -9.81, 0))

# 1. Ground body (fixed)
ground = chrono.ChBody()
ground.SetFixed(True)
ground.SetPos(chrono.ChVector3d(0, 0, 0))
sys.Add(ground)

# 2. First pendulum body
pend1 = chrono.ChBody()
pend1.SetMass(0.79)
pend1.SetInertiaXX(chrono.ChVector3d(0.01, 0.01, 0.0658))
pend1.SetPos(chrono.ChVector3d(0, -0.5, 0))  # CG in center of bar
pend1.SetAngVelParent(chrono.ChVector3d(0, 0, 10))
sys.Add(pend1)

# Visual: 1m tall, half-dim = 0.5, shift visual down by 0.5
shape1 = chrono.ChVisualShapeBox(chrono.ChVector3d(0.01, 1, 0.01))
shape1.SetColor(chrono.ChColor(0.5, 0.5, 0.8))
pend1.AddVisualShape(shape1)

# 3. Second pendulum body
pend2 = chrono.ChBody()
pend2.SetMass(0.3)
pend2.SetInertiaXX(chrono.ChVector3d(0.01, 0.01, 0.0250))
pend2.SetPos(chrono.ChVector3d(0, -1.5, 0))  # CG in center of second bar
pend2.SetAngVelParent(chrono.ChVector3d(0, 0, 5))
sys.Add(pend2)

# Visual for pend2
shape2 = chrono.ChVisualShapeBox(chrono.ChVector3d(0.01, 1, 0.01))
shape2.SetColor(chrono.ChColor(0.8, 0.5, 0.5))
pend2.AddVisualShape(shape2)

# 4. Revolute joint between ground and pend1 at origin
joint1 = chrono.ChLinkRevolute()
frame1 = chrono.ChFramed(chrono.ChVector3d(0, 0, 0))
joint1.Initialize(ground, pend1, frame1)
sys.Add(joint1)

# 5. Revolute joint between pend1 and pend2 at (0, -1, 0)
joint2 = chrono.ChLinkRevolute()
frame2 = chrono.ChFramed(chrono.ChVector3d(0, -1, 0))
joint2.Initialize(pend1, pend2, frame2)
sys.Add(joint2)


# Simulation parameters
end_time = 50.0  # seconds
step_size = 0.01
time = 0.0

# Run the simulation
while time < end_time:
    sys.DoStepDynamics(step_size)

    # Get positions
    pos = pend1.GetPos()
    print(f"{time:.4f} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    time += step_size
