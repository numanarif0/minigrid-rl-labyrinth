import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper 
from stable_baselines3.common.monitor import Monitor
from minigrid.wrappers import ImgObsWrapper
from collections import deque

class RewardWarapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env) 
        self.target_location = None

    def reset(self ,**kwargs ):
        obs , info = self.env.reset(**kwargs)
        self.target_location = self.findTarget()
        self.previous_distance = self.calculateDistance()
        return obs , info



    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated and reward > 0:
            reward = 1.0
            info['is_success'] = True
        elif terminated:
            reward = -1.0
            info['is_success'] = False
        else:
            new_distance = self.calculateDistance()
            distance_dif = self.previous_distance - new_distance
            self.previous_distance = new_distance
            reward = distance_dif * 0.1 - 0.001
            if truncated:
                info['is_success'] = False  
        
        return obs, reward, terminated, truncated, info 

    
        
    def calculateDistance(self):
        if self.target_location is None:
            return 0 
        ax,ay = self.env.unwrapped.agent_pos
        gx,gy = self.findTarget()
        return bfs_distance(self.env.unwrapped.grid, (ax, ay), (gx, gy))

    def findTarget(self):
        for i in range(self.env.unwrapped.grid.width):
            for j in range(self.env.unwrapped.grid.height):
                target = self.env.unwrapped.grid.get(i,j)
                if target is not None and target.type == "goal":
                    return (i,j)
        return None  # Hedef bulunamazsa None döndür

    def isFinish(self):
        ax, ay = self.env.unwrapped.agent_pos
        target = self.findTarget()
        
        if target is None:
            return False
            
        gx, gy = target
        return (ax == gx and ay == gy)

def bfs_distance(grid, start, goal):
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        (x, y), d = queue.popleft()
        if (x, y) == goal:
            return d
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x+dx, y+dy
            cell = grid.get(nx, ny)
            if (nx, ny) not in visited and \
            (cell is None or cell.type != 'lava'):
                visited.add((nx, ny))
                queue.append(((nx, ny), d+1))
    return float('inf')

def MakeEnv(env_id,render_mode=None):
    def _init():
        env = gym.make(env_id,render_mode=render_mode)
        env = RewardWarapper(env)
        #env = FlatObsWrapper(env)
        env = ImgObsWrapper(env)
        env = Monitor(env)
        return env 
    return _init  

