# this script trains a PPO agent to drive in the RectCircuitEnv environment

# imported libraries and modules
from stable_baselines3 import PPO # importing the PPO algorithm from stable_baseline3
from stable_baselines3.common.vec_env import DummyVecEnv # for vectorized environments
from circuit_env import RectCircuitEnv # importing the RectCircuitEnv environment
import os

# importing the MAX_STEPS constant from the config file
from config import TRACK_W, CIRCUIT_L, CIRCUIT_W, V_MAX, W_MAX, MAX_STEPS, DT

# this function is used to create the enviornment for the agent to train in
def make_env():
    # this returns a new instance of the RectCircutEnv environment with the specified parameters
    return RectCircuitEnv(
        render = False, #here we do not render the enviornemnt for training but we could set this to true if we wanted to see the agent training

        track_w = TRACK_W, # this is how wide the track is going to be. we set it to be 2 tiles wide

        circuit_l = CIRCUIT_L, # this is the total length of the circuit

        circuit_w = CIRCUIT_W, # this is the total width of the circuit

        v_max = V_MAX, # this is the max speed the agent can go

         # Yaw rate is the rate at which an object rotates around its vertical axis. 
        # In the context of a car, it refers to how quickly the car can change its heading or
        # direction while driving. A higher yaw rate means the car can make sharper turns,
        # while a lower yaw rate indicates more gradual turns.
        w_max = W_MAX, # this is the maximum yaw rate the agent can have
        
        max_steps = MAX_STEPS, # this is the maximum number of steps the agent can take in one session

        dt = DT # simulation timestep
    )

# creating a vectorized environment
env = DummyVecEnv([make_env])

# this is how we create the PPO model with the specified parameters
model = PPO(
    "CnnPolicy", # using a convolutional neural network policy
    env, # the environment we created above
    verbose=1, # setting verbosity to 1 to see training progress
    n_steps=2048, # number of steps to run for each environment per update
    batch_size=64, # size of minibatches for each gradient update
    learning_rate=3e-4, # learning rate for the optimizer
    gamma=0.99 # discount factor for future rewards
)

MODEL_PATH = "scripts/ppo_rect_circuit_forward_only.zip"

print("Looking for model at:", os.path.abspath(MODEL_PATH))
print("Exists?", os.path.exists(MODEL_PATH))

if os.path.exists(MODEL_PATH + ".zip"):
    print("Loading existing model...")
    model = PPO.load(MODEL_PATH, env=env)
else:
    print("Creating new model...")
    model = PPO("CnnPolicy", env, verbose=1)

# here we will try to train the model for 150,000 timesteps
try:
    # printing the starting timesteps of the model to ensure that training is resumed correctly
    print("Starting timesteps:", model.num_timesteps)
    # training the model
    model.learn(total_timesteps=150_000)

# if we want to interupt the training early we can use a keyboard interrupt
except KeyboardInterrupt:
    # prints a message to tell us that the training was interrupted and the model is being saved
    print("Training was interrupted, Saving model now...")

# saving the trained model to a file
model.save("scripts/ppo_rect_circuit_forward_only")
print("Saved model to scripts/ppo_rect_circuit_forward_only.zip")
