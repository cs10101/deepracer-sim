# config.py
import math

# ---- Track geometry (keep in sync with env args) ----
TRACK_W   = 2.0
CIRCUIT_L = 10.0
CIRCUIT_W = 6.0

# Perimeter / lap length for a rectangle track
LAP_LENGTH = 2.0 * (CIRCUIT_L + CIRCUIT_W)

# ---- Dynamics / limits (keep in sync with env args) ----
V_MAX = 6.0
W_MAX = 2.8

# ---- Simulation timing ----
# If your env uses a different timestep, set it here to match.
DT = 1.0 / 30.0   # common in PyBullet-ish setups

# ---- Episode length heuristic ----
# "Slow" cruise speed as a fraction of V_MAX (safe early-policy speed)
SLOW_SPEED_FRAC = 0.25

# Steps needed to complete 1 lap at slow speed:
# distance_per_step ~= v * dt
SLOW_SPEED = SLOW_SPEED_FRAC * V_MAX
DIST_PER_STEP = max(1e-6, SLOW_SPEED * DT)

# Add slack for turning, wobble, exploration, etc.
SLACK = 3.0

DT = 1.0 / 60.0

MAX_STEPS = int(math.ceil((LAP_LENGTH / DIST_PER_STEP) * SLACK))
