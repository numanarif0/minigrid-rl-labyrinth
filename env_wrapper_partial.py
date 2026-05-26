import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.monitor import Monitor

from env_wrapper import RewardWarapper


def MakePartialEnv(env_id, render_mode=None, agent_view_size=7):
    def _init():
        # MiniGrid returns partial observations by default; keep agent view size explicit.
        env = gym.make(env_id, render_mode=render_mode, agent_view_size=agent_view_size)
        env = RewardWarapper(env)
        env = ImgObsWrapper(env)
        env = Monitor(env)
        return env

    return _init
