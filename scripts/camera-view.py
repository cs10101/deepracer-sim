import time
import math
import numpy as np
import cv2
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

def get_camera_image(car_id, w=84, h=84):
    pos, orn = p.getBasePositionAndOrientation(car_id)
    rot = p.getMatrixFromQuaternion(orn)

    # Forward direction of the car (x-axis in body frame)
    forward = np.array([rot[0], rot[3], rot[6]])

    cam_pos = np.array(pos) + np.array([0, 0, 0.20])              # slightly above car
    cam_target = cam_pos + 1.0 * forward                           # look ahead

    view = p.computeViewMatrix(cam_pos.tolist(), cam_target.tolist(), [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(fov=90, aspect=1.0, nearVal=0.01, farVal=10.0)

    _, _, rgba, _, _ = p.getCameraImage(
        width=w, height=h, viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    rgba = np.array(rgba, dtype=np.uint8).reshape((h, w, 4))
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    return gray

dt = 1 / 240
v = 50.0 # forward speed of the car
w = 5.5 # turning rate of the car
x, y, theta = 0.0, 0.0, 0.0

print("Camera running. Press Q in the camera window to quit.")
while True:
    # Move the car in a curve (just for testing)
    theta += w * dt
    x += v * math.cos(theta) * dt
    y += v * math.sin(theta) * dt

    p.resetBasePositionAndOrientation(car_id, [x, y, 0.1], p.getQuaternionFromEuler([0, 0, theta]))
    p.stepSimulation()

    img = get_camera_image(car_id)
    cv2.imshow("Sim Camera (84x84)", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(dt)

cv2.destroyAllWindows()
