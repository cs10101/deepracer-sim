# This script loads a pre-trained PPO model and uses it to control the car in the RectCircuitEnv environment.

# importing libraries
import time # for sleep
from stable_baselines3 import PPO # importing the PPO algo from stable_baseline3
from circuit_env import RectCircuitEnv # importing the RectCircuitEnv environment

# importing the MAX_STEPS constant from the config file
from config import TRACK_W, CIRCUIT_L, CIRCUIT_W, V_MAX, W_MAX, MAX_STEPS, DT

# creating a vitrual enviornment for the agent to drive around
env = RectCircuitEnv(
    # these are the enviornment parameters
    render=True, # render the environment
    track_w = TRACK_W, # this is how wide the track is going to be. we set it to be 2 tiles wide
    circuit_l = CIRCUIT_L, # this is the total length of the circuit
    circuit_w = CIRCUIT_W, # this is the total width of the circuit
    v_max = V_MAX, # this is the max speed the agent can go
    w_max = W_MAX, # this is the maximum yaw rate the agent can have
    # max_steps = 1200
    max_steps = MAX_STEPS, # this is the maximum number of steps the agent can take in one session
    dt = DT # simulation timestep
)

# this is how we load the pre-trained model which will control the car in the rendered enviornment
model = PPO.load("scripts/ppo_rect_circuit_forward_only")

# reset the environment to get the initial observation
obs, _ = env.reset()

# this loop will keep the car driving untill we explicitly tell it to stop
while True:
    # the action variable stores the next action the model will make based on the current conditions
    action, _ = model.predict(obs, deterministic=True)
    # we then take the calculated action and then step the enviornment forward one step to get a new observation
    obs, reward, terminated, truncated, info = env.step(action)
    # we set the eviornment to run at 60 frames per second
    time.sleep(1/60)

    # if the car has hit a wall the session will reset
    # or if the agent has taken the maximum number of steps in one session
    if terminated or truncated:
        # prints a reason as to why the eviornment reset
        print(f"RESET -> terminated: {terminated}, truncated: {truncated}, info: {info}")
        # this resets the enviornment and gets a new initial observation
        obs, _ = env.reset()
