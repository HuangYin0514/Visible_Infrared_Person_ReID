# =============================================================================
# PROJECT CHRONO - http://projectchrono.org
# Python demo: 两个刚体发生碰撞，并报告接触点
# =============================================================================

import pychrono as chrono

print("演示：小方块掉落并与大方块发生碰撞")

# 创建物理系统
my_system = chrono.ChSystemNSC()

# 创建碰撞材质
material = chrono.ChContactMaterialNSC()
material.SetFriction(0.5)

# -----------------------
# 创建地面大方块（固定）
# -----------------------
bodyA = chrono.ChBody()
bodyA.SetName("GroundBox")
bodyA.SetPos(chrono.ChVector3d(0, -1, 0))
bodyA.SetFixed(True)  # 固定不动
bodyA.AddCollisionShape(chrono.ChCollisionShapeBox(material, 5, 1, 5))  # 半尺寸=5x1x5 => 10x2x10
bodyA.EnableCollision(True)
my_system.Add(bodyA)

# -----------------------
# 创建下落的小方块
# -----------------------
bodyB = chrono.ChBody()
bodyB.SetName("FallingBox")
bodyB.SetMass(1)
bodyB.SetInertiaXX(chrono.ChVector3d(0.1, 0.1, 0.1))
bodyB.SetPos(chrono.ChVector3d(0, 3, 0))  # 在上方
bodyB.AddCollisionShape(chrono.ChCollisionShapeBox(material, 0.5, 0.5, 0.5))  # 半尺寸=0.5 => 边长1
bodyB.EnableCollision(True)
my_system.Add(bodyB)


# -----------------------
# 定义接触报告回调
# -----------------------
class MyReportContactCallback(chrono.ReportContactCallback):
    def __init__(self):
        super().__init__()

    def OnReportContact(self, vA, vB, cA, dist, rad, force, torque, modA, modB, cnstr_offset):
        bodyUpA = chrono.CastContactableToChBody(modA)
        bodyUpB = chrono.CastContactableToChBody(modB)
        print(f"  接触点: A={vA}  B={vB}  dist={dist:.6f} " f"force={force}  " f"体A={bodyUpA.GetName()}  体B={bodyUpB.GetName()}")
        return True


my_rep = MyReportContactCallback()

# -----------------------
# 仿真循环
# -----------------------
step_size = 0.01
while my_system.GetChTime() < 1.0:
    my_system.DoStepDynamics(step_size)
    print(f"time={my_system.GetChTime():.2f}  FallingBox y={bodyB.GetPos().y:.3f}")

    # 打印接触信息
    my_system.GetContactContainer().ReportAllContacts(my_rep)

print("模拟结束。")
