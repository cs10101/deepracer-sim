import time
import pybullet as p
import pybullet_data

# GUI mode so you can see everything
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# A simple box "car"
car_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.20, 0.10, 0.05])
car_visual = p.createVisualShape(
    p.GEOM_BOX, halfExtents=[0.20, 0.10, 0.05], rgbaColor=[0.2, 0.6, 1.0, 1.0]
)

car_id = p.createMultiBody(
    baseMass=2.0,
    baseCollisionShapeIndex=car_collision,
    baseVisualShapeIndex=car_visual,
    basePosition=[0, 0, 0.1],
)

print("Car spawned. Close the window or Ctrl+C to stop.")

while True:
    p.stepSimulation()
    time.sleep(1 / 240)
