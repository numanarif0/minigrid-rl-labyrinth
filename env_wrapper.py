import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper 
from stable_baselines3.common.monitor import Monitor

class LavaPenaltyWarapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env) 
        

    def step(self, action):
        obs , reward , terminated , truncated , info = self.env.step(action)

        if terminated and reward == 0:
            reward = -1.0

        elif not terminated and not truncated:
            reward = -0.001
        return obs , reward , terminated , truncated , info 
    

def MakeEnv(env_id):
    def _init():
        env = gym.make(env_id)
        env = LavaPenaltyWarapper(env)
        env = FlatObsWrapper(env)
        env = Monitor(env)
        return env 
    return _init