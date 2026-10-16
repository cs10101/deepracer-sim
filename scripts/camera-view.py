# importing dependancies which relate to the agents vision capabilities
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

# this function is designed to get the visual observation from the agent
def get_camera_image(car_id, w=84, h=84):

    # we asign two variables to track the agents position and oriontation while on the track
    pos, orn = p.getBasePositionAndOrientation(car_id)

    # we want to convert the agents oriontation to a rotation matrix
    rot = p.getMatrixFromQuaternion(orn)

    # Forward direction of the car (x-axis in body frame)
    forward = np.array([rot[0], rot[3], rot[6]])

    cam_pos = np.array(pos) + np.array([0, 0, 0.20]) # slightly above car
    cam_target = cam_pos + 1.0 * forward # look ahead

    # cam_pos and cam_target are both used to define the camera view
    # the list of [0,0,1] is the world up direction. this keeps the camera upright and pointed forward
    view = p.computeViewMatrix(cam_pos.tolist(), cam_target.tolist(), [0, 0, 1])

    # this line is defining the camera lens.
    # we have the feild of view set to 90 degrees, an aspect ration of 1.0, and a near and far val
    # all of these settings help to try and simulate a real camera
    proj = p.computeProjectionMatrixFOV(fov=90, aspect=1.0, nearVal=0.01, farVal=10.0)


    # p.getCameraImage(..) is called with the height and width, the computed view matrix, and projection matrix
    # and it returnes the rendered image data as RGBA values
    _, _, rgba, _, _ = p.getCameraImage(
        width=w, height=h, viewMatrix=view, projectionMatrix=proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    rgba = np.array(rgba, dtype=np.uint8).reshape((h, w, 4))
    gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY) # finally we convert the image back to greyscale as this makes training easier
    return gray

dt = 1 / 240 # simulation time step
v = 50.0 # forward speed of the car
w = 5.5 # turning rate of the car
x, y, theta = 0.0, 0.0, 0.0 # initial position and orientation of the car

print("Camera running. Press Q in the camera window to quit.")

# this while loop is the main loop of the simulation.
# it updates the car's position and orientation, steps the simulation, gets the camera image, and displays it in a window.
while True:
    # Move the car in a curve (just for testing)
    theta += w * dt
    x += v * math.cos(theta) * dt
    y += v * math.sin(theta) * dt

    # we update the car's position and orientation in the simulation using p.resetBasePositionAndOrientation
    p.resetBasePositionAndOrientation(car_id, [x, y, 0.1], p.getQuaternionFromEuler([0, 0, theta]))
    p.stepSimulation() # this line steps the simulation forward by one time step, updating the physics and rendering the scene

    # we call the get_camera_image function to get the current camera image from the agent's perspective
    img = get_camera_image(car_id)
    cv2.imshow("Sim Camera (84x84)", img)

    # we check if the user has pressed the "q" key to quit the simulation
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(dt) # this line adds a small delay to control the speed of the simulation and make it more visually appealing

cv2.destroyAllWindows() # this line closes the OpenCV window when the simulation is finished
