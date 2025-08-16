# =============================================================================
# PROJECT CHRONO - http://projectchrono.org
#
# 示例：两个刚体碰撞检测（含重力）
# =============================================================================

print("Second tutorial: create and populate a physical system with collision")

import pychrono as chrono

# 1. 创建物理系统，并启用重力
my_system = chrono.ChSystemNSC()
my_system.SetGravitationalAcceleration(chrono.ChVector3d(0, -9.81, 0))
my_system.GetCollisionSystem().SetType(chrono.ChCollisionSystem.Type_BULLET)  # ✅ 启用 Bullet 碰撞

# 2. 创建接触材质
material = chrono.ChContactMaterialNSC()
material.SetFriction(0.3)
material.SetCompliance(0)

# 3. 创建地板（固定刚体）
bodyA = chrono.ChBody()
bodyA.SetName("BodyA")
bodyA.SetMass(20)
bodyA.SetPos(chrono.ChVector3d(0, -1, 0))  # 厚度1，顶面在 y=0
bodyA.AddCollisionShape(chrono.ChCollisionShapeBox(material, 10, 1, 10))
bodyA.SetFixed(True)
bodyA.EnableCollision(True)

# 4. 创建小方块（会下落）
bodyB = chrono.ChBody()
bodyB.SetName("BodyB")
bodyB.SetMass(5)
bodyB.SetPos(chrono.ChVector3d(0, 2, 0))  # 初始离地板有一定高度
bodyB.AddCollisionShape(chrono.ChCollisionShapeBox(material, 1, 1, 1))
bodyB.EnableCollision(True)

# 加入系统
my_system.Add(bodyA)
my_system.Add(bodyB)


# 5. 定义碰撞回调
class MyReportContactCallback(chrono.ReportContactCallback):
    def __init__(self):
        chrono.ReportContactCallback.__init__(self)

    def OnReportContact(self, vA, vB, cA, dist, rad, force, torque, modA, modB, cnstr_offset):
        bodyUpA = chrono.CastContactableToChBody(modA)
        bodyUpB = chrono.CastContactableToChBody(modB)
        print(f"  接触点A={vA}  距离={dist:.4f}  碰撞: {bodyUpA.GetName()} <-> {bodyUpB.GetName()}")
        return True  # 返回 False 会停止遍历


my_rep = MyReportContactCallback()


# 6. 仿真循环
my_system.SetChTime(0)
while my_system.GetChTime() < 1.2:

    my_system.DoStepDynamics(0.01)

    print("time=", round(my_system.GetChTime(), 2), " BodyB pos y=", round(bodyB.GetPos().y, 3))

    # 打印所有接触
    my_system.GetContactContainer().ReportAllContacts(my_rep)

print("----------")

# 7. 输出所有刚体的位置
print("Positions of all bodies in the system:")
for abody in my_system.GetBodies():
    print(" ", abody.GetName(), " pos =", abody.GetPos())

print("Done...")
