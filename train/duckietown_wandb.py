from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3 import A2C, PPO

import wandb
from wandb.integration.sb3 import WandbCallback

#11_19
import gym
from gym_duckietown.envs import DuckietownEnv   # kell a gymnek talan

# GLOBAL VARIABLES
SETTINGS = {
    "env_name" : 'Duckietown-straight_road-v0',
    "n_envs" : 2,   #4
    "total_timesteps" : 5,  #25000
    "project" : "duckietown",
    "model_type" : A2C,
    "policy_type" : "CnnPolicy"
}
# Env names
# Duckietown-straight_road-v0
# Duckietown-4way-v0
# Duckietown-udem1-v0
# Duckietown-small_loop-v0
# Duckietown-small_loop_cw-v0
# Duckietown-zigzag_dists-v0
# Duckietown-loop_obstacles-v0 (static obstacles in the road)
# Duckietown-loop_pedestrians-v0 (moving obstacles in the road)
# MultiMap-v0 (wrapper cycling through all maps)

# CREATE WANDB
config = {
    "policy_type": SETTINGS["policy_type"],
    "total_timesteps": SETTINGS["total_timesteps"],
    "env_name": SETTINGS["env_name"],
}
run = wandb.init(
    entity="silchy",
    project=SETTINGS["project"],
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)

# CREATE GYM
env = make_vec_env(SETTINGS["env_name"], n_envs=SETTINGS["n_envs"])
env = VecFrameStack(env, n_stack=SETTINGS["n_envs"])

# env = VecMonitor(env, filename=f"monitor/{run.id}")
# env = VecVideoRecorder(env, filename=f"video/{run.id}")

# LEARNING
model = SETTINGS["model_type"](SETTINGS["policy_type"], env, verbose=1)
# model = SETTINGS["model_type"](SETTINGS["policy_type"], SETTINGS["env_name"], verbose=1)
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)

run.finish()