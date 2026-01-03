import time
import math
import pybullet as p
import pybullet_data

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

car_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.20, 0.10, 0.05])
car_visual = p.createVisualShape(
    p.GEOM_BOX, halfExtents=[0.20, 0.10, 0.05], rgbaColor=[0.2, 0.6, 1.0, 1.0]
)
car_id = p.createMultiBody(2.0, car_collision, car_visual, [0, 0, 0.1])

dt = 1 / 240
v = 1.0   # forward speed (tweak)
w = 0.6   # yaw rate / steering feel (tweak)

x, y, theta = 0.0, 0.0, 0.0

print("Driving... Ctrl+C to stop.")
while True:
    theta += w * dt
    x += v * math.cos(theta) * dt
    y += v * math.sin(theta) * dt

    p.resetBasePositionAndOrientation(
        car_id,
        [x, y, 0.1],
        p.getQuaternionFromEuler([0, 0, theta]),
    )

    p.stepSimulation()
    time.sleep(dt)
