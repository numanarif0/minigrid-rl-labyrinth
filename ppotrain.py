import gymnasium as gym
from stable_baselines3 import PPO
import time
from minigrid.wrappers import FlatObsWrapper
import minigrid
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from env_wrapper import LavaPenaltyWarapper , MakeEnv
from stable_baselines3.common.vec_env import DummyVecEnv




env = DummyVecEnv([MakeEnv("MiniGrid-LavaCrossingS5N1-v0") for _ in range(4)])

#lr_schedule = lambda progress: 1e-4 * progress


model = PPO(policy="MlpPolicy",env=env,learning_rate=3e-4,n_steps=4096,
            batch_size=128,n_epochs=15,gamma=0.995,ent_coef=0.05,verbose=1)

eval_env = DummyVecEnv([MakeEnv("MiniGrid-LavaCrossingS5N1-v0") ])


evalCallBack = EvalCallback(eval_env=eval_env,best_model_save_path="./best_model/",
                            eval_freq=10_000,n_eval_episodes=10,verbose=1)

model.learn(total_timesteps=2_000_000,callback=evalCallBack)
