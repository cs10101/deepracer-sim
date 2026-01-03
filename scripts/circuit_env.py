# RectCircuitEnv: A PyBullet-based rectangular circuit environment for reinforcement learning.

# imported libraries
import math # for mathematical functions
import numpy as np # for numerical operations
import gymnasium as gym # for creating the gym environment
from gymnasium import spaces # for defining action and observation spaces
import pybullet as p # for physics simulation
import pybullet_data # for accessing pybullet data files

# imported configuration parameters
from config import TRACK_W, CIRCUIT_L, CIRCUIT_W, V_MAX, W_MAX, MAX_STEPS, DT

# Rectangular Circuit Environment class
class RectCircuitEnv(gym.Env):
    """
    Forward-only DeepRacer-style env (Option A):
    - Obs: 84x84 grayscale camera
    - Action: [throttle, steering] in [-1,1]^2
        throttle -> [0, V_MAX]
        steering -> [-W_MAX, +W_MAX]
    - Physics-based motion: resetBaseVelocity (walls are solid)
    - Reward: progress along centerline - distance penalty - wall penalty
    - Done: wall contact OR off-track OR max steps
    """
    # Gym metadata
    metadata = {"render_modes": ["human"]}

    # this is the constructor method for the RectCircuitEnv class.
    def __init__(
        self,
        render=False, # disable redering to speed up training and not kill my battery
        obs_w=84,
        obs_h=84,
        track_w = TRACK_W,
        circuit_l = CIRCUIT_L,
        circuit_w = CIRCUIT_W,
        v_max = V_MAX,
        w_max = W_MAX,
        max_steps = MAX_STEPS,
        seed=0,
        dt = DT
    ): # initialize the environment with various parameters
        super().__init__()
        self.render = render
        self.obs_w = obs_w
        self.obs_h = obs_h

        self.TRACK_W = track_w
        self.HALF_W = track_w / 2.0
        self.CIRCUIT_L = circuit_l
        self.CIRCUIT_W = circuit_w

        self.V_MAX = v_max
        self.W_MAX = w_max
        self.max_steps = max_steps

        # Bigger dt = faster training, still stable for this simple sim
        self.dt = float(dt)

        # Random number generator
        self.rng = np.random.default_rng(seed)

        # Forward-only throttle + steering
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Grayscale camera observation
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.obs_h, self.obs_w, 1),
            dtype=np.uint8
        )

        # PyBullet client and car ID
        self.client = None
        self.car_id = None

        # Episode state
        self.steps = 0
        self.v = 0.0
        self.w = 0.0
        self.last_s = 0.0
        self.last_pos = (0.0, 0.0)

    # ---------- Geometry / progress helpers ----------
    def _closest_point_on_segment(px, py, ax, ay, bx, by):
        # vector AB
        abx, aby = bx - ax, by - ay
        # vector AP
        apx, apy = px - ax, py - ay
        # project t = (AP·AB)/(AB·AB)
        denom = abx*abx + aby*aby
        if denom < 1e-9:
            return ax, ay, 0.0  # degenerate segment

        t = (apx*abx + apy*aby) / denom
        t_clamped = max(0.0, min(1.0, t))
        cx = ax + t_clamped * abx
        cy = ay + t_clamped * aby
        return cx, cy, t_clamped

    def _closest_s_and_dist(self, x, y):
        """
        Return:
          s: progress along rectangle perimeter (0..P)
          d: distance to rectangle centerline
        Centerline rectangle corners at (+/-L/2, +/-W/2).
        """
        L = self.CIRCUIT_L
        W = self.CIRCUIT_W
        hx, hy = L / 2, W / 2

        # Segments around perimeter (start bottom-left moving +X)
        segs = [
            ((-hx, -hy), ( hx, -hy), 0.0),        # bottom: length L
            (( hx, -hy), ( hx,  hy), L),          # right: length W
            (( hx,  hy), (-hx,  hy), L + W),      # top: length L
            ((-hx,  hy), (-hx, -hy), 2*L + W),    # left: length W
        ]
        P = 2*L + 2*W

        best_d2 = float("inf")
        best_s = 0.0

        # Find closest point on any segment
        # this section is used to find the closest point on the rectangular circuit to the current position of the car.
        # the for loop iterates through each segment of the rectangle and calculates the distance from the car's position to the closest point on that segment.
        for (ax, ay), (bx, by), s0 in segs:
            vx, vy = (bx - ax), (by - ay)
            wx, wy = (x - ax), (y - ay)
            vv = vx*vx + vy*vy
            t = 0.0 if vv == 0 else max(0.0, min(1.0, (wx*vx + wy*vy) / vv))
            px, py = (ax + t*vx), (ay + t*vy)
            dx, dy = (x - px), (y - py)
            d2 = dx*dx + dy*dy
            # if this distance is the best so far, update best_d2 and best_s
            if d2 < best_d2:
                best_d2 = d2
                seg_len = math.sqrt(vv)
                best_s = (s0 + t*seg_len) % P

        # returns the closest distance and progress along the rectangle perimeter
        return best_s, math.sqrt(best_d2)

    # ---------- PyBullet building ----------
    def _make_box(self, center, half_extents, rgba=(0.9, 0.9, 0.9, 1.0), collision=True):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents) if collision else -1
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
        return p.createMultiBody(0, col, vis, center)

    def _build_rect_circuit(self):
        WALL_H = 0.25
        WALL_T = 0.06

        outer_L = self.CIRCUIT_L/2 + self.HALF_W
        outer_W = self.CIRCUIT_W/2 + self.HALF_W
        inner_L = self.CIRCUIT_L/2 - self.HALF_W
        inner_W = self.CIRCUIT_W/2 - self.HALF_W

        def rectangle_walls(half_L, half_W, rgba):
            # Top/bottom
            self._make_box([0,  half_W + WALL_T/2, WALL_H/2], [half_L + WALL_T, WALL_T/2, WALL_H/2], rgba, collision=True)
            self._make_box([0, -half_W - WALL_T/2, WALL_H/2], [half_L + WALL_T, WALL_T/2, WALL_H/2], rgba, collision=True)
            # Left/right
            self._make_box([ half_L + WALL_T/2, 0, WALL_H/2], [WALL_T/2, half_W + WALL_T, WALL_H/2], rgba, collision=True)
            self._make_box([-half_L - WALL_T/2, 0, WALL_H/2], [WALL_T/2, half_W + WALL_T, WALL_H/2], rgba, collision=True)

        rectangle_walls(outer_L, outer_W, rgba=(0.85, 0.85, 0.85, 1.0))
        rectangle_walls(inner_L, inner_W, rgba=(0.75, 0.75, 0.75, 1.0))

        # Centerline (visual only) - keep segments inside corners (no overhang)
        seg_len = 0.40
        seg_thick = 0.03
        z = 0.01

        half_L = self.CIRCUIT_L / 2
        half_W = self.CIRCUIT_W / 2

        # Place segment centers so each segment stays within [-half_L, +half_L] (no overhang)
        xs = np.arange(-half_L + seg_len/2, half_L - seg_len/2 + 1e-6, seg_len)
        for sx in xs:
            self._make_box([sx,  half_W, z], [seg_len/2, seg_thick/2, 0.005], rgba=(1,1,1,1), collision=False)
            self._make_box([sx, -half_W, z], [seg_len/2, seg_thick/2, 0.005], rgba=(1,1,1,1), collision=False)

        ys = np.arange(-half_W + seg_len/2, half_W - seg_len/2 + 1e-6, seg_len)
        for sy in ys:
            self._make_box([ half_L, sy, z], [seg_thick/2, seg_len/2, 0.005], rgba=(1,1,1,1), collision=False)
            self._make_box([-half_L, sy, z], [seg_thick/2, seg_len/2, 0.005], rgba=(1,1,1,1), collision=False)

        # Optional: corner markers so the line looks continuous at turns
        corner_size = seg_thick / 2
        for cx, cy in [(half_L, half_W), (half_L, -half_W), (-half_L, half_W), (-half_L, -half_W)]:
            self._make_box([cx, cy, z], [corner_size, corner_size, 0.005], rgba=(1,1,1,1), collision=False)


    # this method is used to spawn the car in the circuit environment.
    def _spawn_car(self):
        CAR_Z = 0.10
        start_pos = [-self.CIRCUIT_L/2 + 0.8, -self.CIRCUIT_W/2 + 0.25, CAR_Z]

        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.20, 0.10, 0.05])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.20, 0.10, 0.05], rgbaColor=[0.2, 0.6, 1.0, 1.0])
        self.car_id = p.createMultiBody(2.0, col, vis, start_pos)

        # Friction so it behaves like a grippy car, not a hockey puck
        p.changeDynamics(self.car_id, -1, lateralFriction=1.2, rollingFriction=0.02, restitution=0.0)

        self.v = 0.0
        self.w = 0.0
        self.steps = 0

        self.last_pos = (start_pos[0], start_pos[1])
        self.last_s, _ = self._closest_s_and_dist(start_pos[0], start_pos[1])

    # this method is used to get the observations from the environment.
    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.car_id)
        rot = p.getMatrixFromQuaternion(orn)
        forward = np.array([rot[0], rot[3], rot[6]])

        cam_pos = np.array(pos) + np.array([0.0, 0.0, 0.20])
        cam_target = cam_pos + 1.8 * forward

        view = p.computeViewMatrix(cam_pos.tolist(), cam_target.tolist(), [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(110, self.obs_w / self.obs_h, 0.01, 20.0)

        _, _, rgba, _, _ = p.getCameraImage(
            self.obs_w, self.obs_h, view, proj,
            renderer=p.ER_TINY_RENDERER
        )
        rgba = np.array(rgba, dtype=np.uint8).reshape((self.obs_h, self.obs_w, 4))
        gray = (0.299 * rgba[:, :, 0] + 0.587 * rgba[:, :, 1] + 0.114 * rgba[:, :, 2]).astype(np.uint8)
        return gray[:, :, None]

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.client is None:
            self.client = p.connect(p.GUI if self.render else p.DIRECT)

        p.resetSimulation()

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)      # fixed-step sim (important for RL)
        p.setTimeStep(self.dt)          # makes self.dt real
        p.loadURDF("plane.urdf")


        self._build_rect_circuit()
        self._spawn_car()

        P = 2 * self.CIRCUIT_L + 2 * self.CIRCUIT_W

        self.num_checkpoints = 12
        self.cp_len = P / self.num_checkpoints
        self.visited_cps = set()

        if self.render:
            p.resetDebugVisualizerCamera(
                cameraDistance=8.0, cameraYaw=40, cameraPitch=-35,
                cameraTargetPosition=[0, 0, 0]
            )

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.steps += 1

        # ----- Forward-only throttle mapping -----
        throttle = float(np.clip(action[0], -1.0, 1.0))
        steer = float(np.clip(action[1], -1.0, 1.0))

        v_min = 0.25 * self.V_MAX # can be modified depending on track tightness

        # Map throttle [-1,1] -> [0, V_MAX]
        target_v = v_min + ((throttle + 1.0) / 2.0) * (self.V_MAX - v_min)

        # Map steer [-1,1] -> [-W_MAX, +W_MAX]
        target_w = steer * self.W_MAX

        # Reduce speed when steering hard (cornering safety)
        steer_mag = abs(steer)
        target_v *= (1.0 - 0.6 * steer_mag)

        # Smooth (keeps it stable and learnable)
        self.v += (target_v - self.v) * 0.40
        self.w += (target_w - self.w) * 0.60

        # Current yaw from physics
        # --- Physics state BEFORE stepping (used to compute desired velocity direction) ---
        pos, orn = p.getBasePositionAndOrientation(self.car_id)
        x, y = pos[0], pos[1]
        yaw = p.getEulerFromQuaternion(orn)[2]

        # Convert forward speed + yaw into world-frame velocity
        vx = self.v * math.cos(yaw)
        vy = self.v * math.sin(yaw)

        # --- Apply motion & step physics ---
        p.resetBaseVelocity(
            self.car_id,
            linearVelocity=[vx, vy, 0.0],
            angularVelocity=[0.0, 0.0, self.w]
        )
        p.stepSimulation()

        # --- Physics state AFTER stepping (actual outcome) ---
        lin_vel, ang_vel = p.getBaseVelocity(self.car_id)
        speed = math.sqrt(lin_vel[0]**2 + lin_vel[1]**2)
        yaw_rate = abs(ang_vel[2])

        pos, orn = p.getBasePositionAndOrientation(self.car_id)
        x, y = pos[0], pos[1]
        yaw = p.getEulerFromQuaternion(orn)[2]

        # --- Lookahead / preview distance (compute now, apply in reward later) ---
        lookahead = 0.8  # try 0.5–1.2
        fx = x + lookahead * math.cos(yaw)
        fy = y + lookahead * math.sin(yaw)
        _, dist_ahead = self._closest_s_and_dist(fx, fy)


        # ----- Reward -----
        P = 2*self.CIRCUIT_L + 2*self.CIRCUIT_W
        s, dist = self._closest_s_and_dist(x, y)

        # progress around loop (handle wrap)
        ds = s - self.last_s
        if ds < -P/2:
            ds += P
        elif ds > P/2:
            ds -= P
        self.last_s = s

        ds_fwd = max(0.0, ds) # ignore backwards movement

        no_progress_penalty = 0.0
        if ds_fwd < 1e-4:
            no_progress_penalty = 0.2

        cp_idx = int(s // self.cp_len)

        checkpoint_reward = 0.0

        if (cp_idx not in self.visited_cps) and (ds_fwd > 0.0):
            self.visited_cps.add(cp_idx)
            checkpoint_reward = 1.0

        # Contact with walls?
        contacts = p.getContactPoints(bodyA=self.car_id)

        hit_wall = False

        for c in contacts:
            other_body = c[2]
            normal_force = c[9]

            # ignore ground plane
            if other_body == 0:
                continue

            #ignore tiny numerical contacts
            if normal_force > 1.0:
                hit_wall = True
                break

        if hit_wall:
            ds_fwd = 0.0
            checkpoint_reward = 0.0

        # ----- reward shaping -----
        progress_reward = 3.0 * ds_fwd
        center_penalty = 0.4 * (dist ** 2)

        steering_penalty = 0.02 * (self.w ** 2)
        turn_at_speed_penalty = 0.2 * (speed / self.V_MAX) * (self.w ** 2)

        wall_penalty = 8.0 if hit_wall else 0.0

        edge = dist / self.HALF_W
        edge = max(0.0, min(1.5, edge))
        track_factor = max(0.0, 1.0 - edge)

        # Speed reward only helps when we're not near the edge
        speed_reward = 0.10 * (speed / self.V_MAX) * track_factor if ds_fwd > 0.0 else 0.0

        # Lookahead / preview penalty (encourages early turning)
        lookahead_penalty = (speed / self.V_MAX) * 0.8 * dist_ahead

        # Spin penalty (discourages spinning out of control)
        spin_penalty = 0.1 * yaw_rate * yaw_rate * (speed / self.V_MAX)

        margin = self.HALF_W - dist  # how much track you have left before off-track
        safe_margin = 0.25  # meters-ish; tune 0.20–0.35
        edge_penalty = 0.0
        if margin < safe_margin:
            edge_penalty = 8.0 * (safe_margin - margin) ** 2



        # the reward variable is used to store the total reward the agent gets for each step in the enviornment.
        # during training the agent will learn how to maximise this reward over time
        reward = (
            progress_reward # this reward encourages the agent to make forward progress along the track
            + checkpoint_reward # this reward encourages the agent to reach a new checkpoint on the track
            + speed_reward # this reward encourages the agent to maintain a high speed when driving on the track
            - center_penalty # this penalty discouraged the agent from straying too far from the centerline
            - steering_penalty # this penalty discourages the agent from making sharp turns
            - turn_at_speed_penalty # this penalty discourages  the agent from making sharp turns at high speeds, encouraging the model to slow down when turning
            - lookahead_penalty # this penalty encourages the agent to anticipate upcoming turns and adjust its steeping 
            - wall_penalty # this penalty discourages the agent from hitting the walls on the track
            - no_progress_penalty # this penalty discourages the agent from getting stuck and not making any forward progress
            - spin_penalty # this penalty discourages the agent from spinning out of control, especially at high speeds
            - edge_penalty # this penalty discourages the agent from getting too close to the edge of the track
            - 0.002 # small constant penalty to encourage the agent to finish the track faster
        )

        terminated = False
        truncated = False

        # --- Done conditions ---
        if hit_wall:
            terminated = True
            reward -= (3.0 + 5.0 * (speed / self.V_MAX))

        # Off-track if too far from centerline
        if dist > (self.HALF_W * 0.95):
            terminated = True
            reward -= 2.0

        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {"dist": dist, "ds": ds, "hit_wall": hit_wall, "v": self.v, "w": self.w, "speed": speed}

        if self.steps % 20 == 0:  # print every 20 steps so it doesn't spam
            print(
            f"step={self.steps} | "
            f"speed={speed:.2f} | "
            f"ds_fwd={ds_fwd:.3f} | "
            f"dist={dist:.3f} | "
            f"cp={len(self.visited_cps)} | "
            f"hit_wall={hit_wall}"
        )

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.client is not None:
            p.disconnect()
            self.client = None
