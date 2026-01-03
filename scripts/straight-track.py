import time
import math
import numpy as np
import cv2
import pybullet as p
import pybullet_data

# ---------- Config ----------
DT = 1 / 240

WALL_HEIGHT = 0.25
WALL_THICKNESS = 0.05
TRACK_LENGTH = 12.0
TRACK_HALF_WIDTH = 1.0   # try 0.7 for tighter

CAR_Z = 0.10

# Controls (Option A)
V_MAX = 5.0      # forward/back speed
W_MAX = 2.4      # turn rate

# Camera
CAM_W, CAM_H = 84, 84
CAM_FOV = 90
CAM_NEAR, CAM_FAR = 0.01, 10.0

ESC_KEY = 27  # ASCII code for Escape


# ---------- Helpers ----------
def make_wall(center, half_extents, rgba=(0.9, 0.9, 0.9, 1.0)):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
    wall_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=center
    )
    return wall_id


def make_car(start_pos):
    car_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.20, 0.10, 0.05])
    car_visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.20, 0.10, 0.05], rgbaColor=[0.2, 0.6, 1.0, 1.0]
    )
    car_id = p.createMultiBody(
        baseMass=2.0,
        baseCollisionShapeIndex=car_collision,
        baseVisualShapeIndex=car_visual,
        basePosition=list(start_pos)
    )
    return car_id


def get_camera_image(car_id, w=CAM_W, h=CAM_H):
    """Return an (h,w) grayscale image from a forward-facing camera on the car."""
    pos, orn = p.getBasePositionAndOrientation(car_id)
    rot = p.getMatrixFromQuaternion(orn)

    # Forward direction in world frame (car's x-axis)
    forward = np.array([rot[0], rot[3], rot[6]])

    cam_pos = np.array(pos) + np.array([0.0, 0.0, 0.20])   # slightly above car
    cam_target = cam_pos + 1.0 * forward                   # look ahead

    view = p.computeViewMatrix(cam_pos.tolist(), cam_target.tolist(), [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(
        fov=CAM_FOV, aspect=float(w) / float(h), nearVal=CAM_NEAR, farVal=CAM_FAR
    )

    _, _, rgba, _, _ = p.getCameraImage(
        width=w,
        height=h,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgba = np.array(rgba, dtype=np.uint8).reshape((h, w, 4))
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    return gray


# ---------- Main ----------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# Track walls
left_center  = [0,  TRACK_HALF_WIDTH + WALL_THICKNESS/2, WALL_HEIGHT/2]
right_center = [0, -TRACK_HALF_WIDTH - WALL_THICKNESS/2, WALL_HEIGHT/2]
wall_half_extents = [TRACK_LENGTH/2, WALL_THICKNESS/2, WALL_HEIGHT/2]

make_wall(left_center,  wall_half_extents)
make_wall(right_center, wall_half_extents)

# Gates (optional obstacles)
gate_half = [0.10, 0.25, 0.20]
make_wall([-2.0,  TRACK_HALF_WIDTH - 0.15, 0.20], gate_half, rgba=(1, 0.6, 0.2, 1))
make_wall([ 2.0, -TRACK_HALF_WIDTH + 0.15, 0.20], gate_half, rgba=(1, 0.6, 0.2, 1))

# Spawn ONE car
start_x = -TRACK_LENGTH/2 + 0.7
car_id = make_car((start_x, 0.0, CAR_Z))

# Follow camera in the PyBullet GUI (nice while driving)
p.resetDebugVisualizerCamera(
    cameraDistance=4.0, cameraYaw=30, cameraPitch=-25,
    cameraTargetPosition=[start_x, 0, 0]
)

# Kinematic state
x, y, theta = start_x, 0.0, 0.0
v, w = 0.0, 0.0

print("Controls: W/S = forward/back, A/D = steer, R = reset, ESC = quit")
print("Camera: press Q in the OpenCV window to quit (or ESC in PyBullet)")

while True:
    keys = p.getKeyboardEvents()

    # Quit via ESC
    if ESC_KEY in keys and keys[ESC_KEY] & p.KEY_WAS_TRIGGERED:
        break

    # Reset car
    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        x, y, theta = start_x, 0.0, 0.0
        p.resetBasePositionAndOrientation(car_id, [x, y, CAR_Z], p.getQuaternionFromEuler([0, 0, theta]))
        v, w = 0.0, 0.0

    # Controls
    steer_left  = (ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN)
    steer_right = (ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN)
    forward     = (ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN)
    backward    = (ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN)

    target_v = 0.0
    if forward:
        target_v = V_MAX
    if backward:
        target_v = -V_MAX

    target_w = 0.0
    if steer_left:
        target_w = W_MAX
    if steer_right:
        target_w = -W_MAX

    # Smooth it so driving feels nicer
    v += (target_v - v) * 0.15
    w += (target_w - w) * 0.20

    # Option A kinematics
    theta += w * DT
    x += v * math.cos(theta) * DT
    y += v * math.sin(theta) * DT

    p.resetBasePositionAndOrientation(
        car_id,
        [x, y, CAR_Z],
        p.getQuaternionFromEuler([0, 0, theta])
    )

    # Camera image
    img = get_camera_image(car_id)
    cv2.imshow("Sim Camera (84x84)", img)

    # Quit via OpenCV window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Keep the PyBullet camera following roughly
    p.resetDebugVisualizerCamera(
        cameraDistance=4.0, cameraYaw=30, cameraPitch=-25,
        cameraTargetPosition=[x, y, 0]
    )

    p.stepSimulation()
    time.sleep(DT)

cv2.destroyAllWindows()
p.disconnect()
