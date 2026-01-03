import pybullet as p
import pybullet_data
import time

# Start PyBullet with a GUI
p.connect(p.GUI)

# Tell PyBullet where its built-in assets are
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Gravity (Earth-like)
p.setGravity(0, 0, -9.81)

# Load a flat ground plane
plane_id = p.loadURDF("plane.urdf")

print("World running. Close the window to exit.")

while True:
    p.stepSimulation()
    time.sleep(1 / 240)
