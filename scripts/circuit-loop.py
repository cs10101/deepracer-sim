import time
import math
import numpy as np
import cv2
import pybullet as p
import pybullet_data

DT = 1/120
ESC_KEY = 27

# --- Track parameters ---
WALL_H = 0.25
WALL_T = 0.06
TRACK_W = 2.0          # drivable width (between inner + outer wall)
HALF_W = TRACK_W / 2

# Circuit dimensions (centerline rectangle)
CIRCUIT_L = 10.0       # length in X
CIRCUIT_W = 6.0        # width in Y

# --- Car controls (Option A) ---
# Maximum linear and angular velocities
V_MAX = 15.0
# Max steering angular velocity
W_MAX = 10
# Car Z height
CAR_Z = 0.10

# --- Camera ---
CAM_W, CAM_H = 84, 84

def make_box_static(center, half_extents, rgba=(0.9, 0.9, 0.9, 1.0)):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
    return p.createMultiBody(0, col, vis, center)

def make_box_visual(center, half_extents, rgba=(1, 1, 1, 1)):
    """Visual-only box (no collision) for centerline markers."""
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
    return p.createMultiBody(0, -1, vis, center)

def make_car(start_pos):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.20, 0.10, 0.05])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.20, 0.10, 0.05], rgbaColor=[0.2, 0.6, 1.0, 1.0])
    return p.createMultiBody(2.0, col, vis, start_pos)

def get_camera_image(car_id, w=CAM_W, h=CAM_H):
    pos, orn = p.getBasePositionAndOrientation(car_id)
    rot = p.getMatrixFromQuaternion(orn)
    forward = np.array([rot[0], rot[3], rot[6]])

    cam_pos = np.array(pos) + np.array([0, 0, 0.20])
    cam_target = cam_pos + 1.0 * forward

    view = p.computeViewMatrix(cam_pos.tolist(), cam_target.tolist(), [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(90, float(w)/float(h), 0.01, 20.0)

    _, _, rgba, _, _ = p.getCameraImage(w, h, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgba = np.array(rgba, dtype=np.uint8).reshape((h, w, 4))
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    return gray

def build_rect_circuit():
    """
    Build a rectangular loop using inner + outer walls.
    Centerline rectangle has dimensions CIRCUIT_L x CIRCUIT_W.
    Track width is TRACK_W.
    """
    # Outer rectangle half-dims
    outer_L = CIRCUIT_L/2 + HALF_W
    outer_W = CIRCUIT_W/2 + HALF_W

    # Inner rectangle half-dims
    inner_L = CIRCUIT_L/2 - HALF_W
    inner_W = CIRCUIT_W/2 - HALF_W

    # Helper to place 4 walls around a rectangle
    def rectangle_walls(half_L, half_W, rgba):
        # Top / bottom (along X)
        make_box_static([0,  half_W + WALL_T/2, WALL_H/2], [half_L + WALL_T, WALL_T/2, WALL_H/2], rgba)
        make_box_static([0, -half_W - WALL_T/2, WALL_H/2], [half_L + WALL_T, WALL_T/2, WALL_H/2], rgba)
        # Left / right (along Y)
        make_box_static([ half_L + WALL_T/2, 0, WALL_H/2], [WALL_T/2, half_W + WALL_T, WALL_H/2], rgba)
        make_box_static([-half_L - WALL_T/2, 0, WALL_H/2], [WALL_T/2, half_W + WALL_T, WALL_H/2], rgba)

    # Outer boundary walls
    rectangle_walls(outer_L, outer_W, rgba=(0.85, 0.85, 0.85, 1.0))
    # Inner boundary walls
    rectangle_walls(inner_L, inner_W, rgba=(0.75, 0.75, 0.75, 1.0))

    # Centerline: draw a thin white strip around the center rectangle
    # We'll place small segments along the rectangle perimeter.
    seg_len = 0.40
    seg_thick = 0.03
    z = 0.01

    # Along top and bottom edges
    for sx in np.arange(-CIRCUIT_L/2, CIRCUIT_L/2 + 1e-6, seg_len):
        make_box_visual([sx,  CIRCUIT_W/2, z], [seg_len/2, seg_thick/2, 0.005], rgba=(1,1,1,1))
        make_box_visual([sx, -CIRCUIT_W/2, z], [seg_len/2, seg_thick/2, 0.005], rgba=(1,1,1,1))

    # Along left and right edges
    for sy in np.arange(-CIRCUIT_W/2, CIRCUIT_W/2 + 1e-6, seg_len):
        make_box_visual([ CIRCUIT_L/2, sy, z], [seg_thick/2, seg_len/2, 0.005], rgba=(1,1,1,1))
        make_box_visual([-CIRCUIT_L/2, sy, z], [seg_thick/2, seg_len/2, 0.005], rgba=(1,1,1,1))

# --- Main ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

build_rect_circuit()

# Start on bottom straight, facing +X
start_pos = [-CIRCUIT_L/2 + 0.8, -CIRCUIT_W/2 + 0.25, CAR_Z]
car_id = make_car(start_pos)

# Give the car a bit of friction so it doesn't skate
p.changeDynamics(car_id, -1, lateralFriction=1.2, rollingFriction=0.02, restitution=0.0)

theta = 0.0
x, y = start_pos[0], start_pos[1]
v, w = 0.0, 0.0

p.resetDebugVisualizerCamera(cameraDistance=8.0, cameraYaw=40, cameraPitch=-35, cameraTargetPosition=[0, 0, 0])

print("Rect circuit ready.")
print("Controls: W/S = forward/back, A/D = steer, R = reset, ESC = quit. Press Q in camera window to quit.")

while True:
    keys = p.getKeyboardEvents()

    if ESC_KEY in keys and keys[ESC_KEY] & p.KEY_WAS_TRIGGERED:
        break

    if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
        x, y, theta = start_pos[0], start_pos[1], 0.0
        p.resetBasePositionAndOrientation(car_id, [x, y, CAR_Z], p.getQuaternionFromEuler([0, 0, theta]))
        v, w = 0.0, 0.0

    steer_left  = (ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN)
    steer_right = (ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN)
    forward     = (ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN)
    backward    = (ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN)

    target_v = (V_MAX if forward else (-V_MAX if backward else 0.0))
    target_w = (W_MAX if steer_left else (-W_MAX if steer_right else 0.0))

    # --- Physics-based movement (walls are now solid) ---

    # Get current orientation from physics
    pos, orn = p.getBasePositionAndOrientation(car_id)
    yaw = p.getEulerFromQuaternion(orn)[2]

    # Smooth velocity + steering (same as before)
    v += (target_v - v) * 0.15
    w += (target_w - w) * 0.20

    # Convert forward speed into world-frame velocity
    vx = v * math.cos(yaw)
    vy = v * math.sin(yaw)

    # Let the physics engine move the car
    p.resetBaseVelocity(
        car_id,
        linearVelocity=[vx, vy, 0.0],
        angularVelocity=[0.0, 0.0, w]
    )

    img = get_camera_image(car_id)

    cv2.imshow("Sim Camera (84x84)", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    p.stepSimulation()

    contacts = p.getContactPoints(bodyA = car_id)

    if len(contacts) > 0:
        print("Wall has been hit!")
        
    time.sleep(DT)

cv2.destroyAllWindows()
p.disconnect()
