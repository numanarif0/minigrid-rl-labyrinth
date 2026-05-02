import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper 
from stable_baselines3.common.monitor import Monitor

class LavaPenaltyWarapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env) 
        self.target_location = None

    def reset(self ,**kwargs ):
        obs , info = self.env.reset(**kwargs)
        self.target_location = self.findTarget()
        self.previous_distance = self.calculateDistance()
        return obs , info



    def step(self, action):
        obs , reward , terminated , truncated , info = self.env.step(action)
        if terminated and reward > 0:
            reward = 1.0

        elif terminated and reward == 0:
            reward = -1.0

        else:
            new_distance = self.calculateDistance()
            distance_dif = self.previous_distance - new_distance
            self.previous_distance = new_distance
            reward = distance_dif * 0.1 - 0.001

        
        return obs , reward , terminated , truncated , info 

    def findTarget(self):
        for i in range(self.env.unwrapped.grid.height):
            for j in range(self.env.unwrapped.grid.width):
                target = self.env.unwrapped.grid.get(i,j)
                if target is not None and target.type == "goal":
                    return (i,j)

    def calculateDistance(self):
        if self.target_location is None:
            return 0 
        ax,ay = self.env.unwrapped.agent_pos
        gx,gy = self.findTarget()
        return abs(ax-gx) + abs(ay-gy)         

def MakeEnv(env_id):
    def _init():
        env = gym.make(env_id)
        env = LavaPenaltyWarapper(env)
        env = FlatObsWrapper(env)
        env = Monitor(env)
        return env 
    return _init