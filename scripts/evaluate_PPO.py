from pathlib import Path
import csv
import numpy as np

from stable_baselines3 import PPO
from circuit_env import RectCircuitEnv
from config import TRACK_W, CIRCUIT_L, CIRCUIT_W, V_MAX, W_MAX, MAX_STEPS, DT

# Project directories
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent

# Project paths
MODEL_PATH = (
    SCRIPTS_DIR 
    / "scripts"
    / "ppo_rect_circuit_forward_only.zip"
)

# evaluation results
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents = True, exist_ok=True)

# evaluation settings
NUM_EPISODES = 10

# load the existing trained agent
print(f"Project Root: {PROJECT_ROOT}")
print(f"Model Path: {MODEL_PATH}")
print(f"Model exists? {MODEL_PATH.is_file()}")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained PPO model not found at {MODEL_PATH}"
    )

# load the trained PPO model
model = PPO.load(str(MODEL_PATH), device="cpu")

# Recreate the original training environment
env = RectCircuitEnv(
    render = False,
    track_w = TRACK_W,
    circuit_l = CIRCUIT_L,
    circuit_w = CIRCUIT_W,
    v_max = V_MAX,
    w_max = W_MAX,
    max_steps = MAX_STEPS,
    dt = DT
)

results = []
episode_with_completed_lap = 0
all_lap_times = []
episode_with_failure = 0

try:
    for episode in range(1, NUM_EPISODES + 1):
        obs, info = env.reset(seed = 42 + episode)

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        last_info = {}

        while not (terminated or truncated):
            action, _ = model.predict(
                obs,
                deterministic = True
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            if info["completed_laps"] > 0 and steps % 100 == 0:
                print(
                    f"Laps completed: {info['completed_laps']} | "
                    f"fastest lap: {info['fastest_lap']}"
                )

            total_reward += float(reward)
            steps += 1
            last_info = info

        # time in the simulated environment
        duration = steps * DT

        # checkpoints visited during this episode
        checkpoints = len(env.visited_cps)

        # colission status from the final step
        hit_wall = bool(
            last_info.get("hit wall", False)
        )

        # episode outcome
        if hit_wall:
            outcome = "Wall collision"

        elif terminated:
            outcome = "off track / other termination"

        elif truncated:
            outcome = "step limit reached"

        else:
            outcome = "unknown"


        result = {
            "episode": episode,
            "total_reward": round(total_reward, 3),
            "steps": steps,
            "duration_seconds": round(duration, 3),
            "checkpoints": checkpoints,
            "hit_wall": hit_wall,
            "outcome": outcome
        }

        results.append(result)

        if last_info.get("completed_laps", 0) > 0:
            episode_with_completed_lap += 1

        episode_lap_times = last_info.get("lap_times", [])
        all_lap_times.extend(episode_lap_times)

        print(
            f"Episode {episode:02d} | "
            f"Reward: {total_reward:.2f} | "
            f"Steps: {steps} | "
            f"Duration: {duration:.2f}s | "
            f"Checkpoints: {checkpoints} | "
            f"Outcome: {outcome}"
        )

finally:
    env.close()

# save results
CSV_PATH = RESULTS_DIR / "ppo_baseline.csv"

if results:
    with open(CSV_PATH, 'w', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Evaluation resulsts saved to: {CSV_PATH}")

else:
    print("No evaluation resuslts to save.")

with open(CSV_PATH, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=results[0].keys()
    )
    writer.writeheader()
    writer.writerows(results)

# metric 1: lap completion rate
lap_completion_rate = (
    episode_with_completed_lap / NUM_EPISODES
) * 100

# metric 2: Fastest lap time
if all_lap_times:
    fastest_lap = min(all_lap_times)
else:
    fastest_lap = None

# metric 3: Average lap time
if all_lap_times:
    average_lap_time = np.mean(all_lap_times)
else:
    average_lap_time = None

# metric 4: Lap consistency
if len(all_lap_times) >= 2:
    lap_time_std = np.std(all_lap_times)
else:
    lap_time_std = None

# metric 5: Collision / off-track failure rate
hit_wall = bool(last_info.get("hit_wall", False))
off_track = bool(last_info.get("off_track", False))

episode_failed = hit_wall or off_track

if episode_failed:
    episode_with_failure += 1

collisions_rate = (
    episode_with_failure / NUM_EPISODES
) * 100


# summary statistics
rewards = [r["total_reward"] for r in results]
durations = [r["duration_seconds"] for r in results]
collisions = sum(r["hit_wall"] for r in results)

print("\n--- PPO BASELINE EVALUATION ---")
print(f"Episode: {NUM_EPISODES}")
print(f"Average reward: {np.mean(rewards):.2f}")
print(f"Reward std deviation: {np.std(rewards):.2f}")
print(f"Average duration: {np.mean(durations):.2f}s")
print(f"Wall collisions: {collisions}/{NUM_EPISODES}")
print(f"Lap completion rate: {lap_completion_rate:.2f}%")

# fastest lap
if fastest_lap is not None:
    print(f"Fastest lap: {fastest_lap:.2f}s")
else:
    print("Fastest lap: No completed laps")

# average lap time
if average_lap_time is not None:
    print(f"Average lap time:  {average_lap_time}s")
else:
    print("Average lap time: No completed laps")

# display lap consistency
if lap_time_std is not None:
    print(f"Lap time standard deviation: {lap_time_std:.3f}s")
else:
    print("lap time standard deviation: No completed laps")

# collision rate
print(f"Collision / off-track rate: {collisions_rate:.2f}%")

print(f"Results saved to: {CSV_PATH}")