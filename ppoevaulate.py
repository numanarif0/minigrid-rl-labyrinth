import gymnasium as gym
import minigrid
from stable_baselines3 import PPO
from minigrid.wrappers import FlatObsWrapper 
from env_wrapper import MakeEnv
from env_wrapper import LavaPenaltyWarapper

env = gym.make("MiniGrid-LavaCrossingS9N1-v0")
env = LavaPenaltyWarapper(env)
env = FlatObsWrapper(env)

model = PPO.load("./best_model/best_model",env=env)

rewardsList = []
stepsList = []

for episodes in range(100):
    obs , info = env.reset()
    total_rewards=0
    done=False
    step= 0
    
    while not done:
        step+=1
        action , _ = model.predict(obs,deterministic=True)
        obs, reward , terminated , truncated , info = env.step(action=action)
        done = terminated or truncated
        total_rewards=total_rewards+reward

    rewardsList.append(total_rewards)
    stepsList.append(step)

    print(f"Episode: {episodes+1} Rewards: {total_rewards} Steps: {step}")

print(f"Finish Train")

print(f"PPO mean reward: {sum(rewardsList)/len(rewardsList)}")
print(f"PPO mean step: {sum(stepsList)/len(stepsList)}")

win=0
for i in range(len(rewardsList)):
    if rewardsList[i]>0:
        win+=1

print(f"Succes Rate: {win*100/len(rewardsList)}")


print(f"------Random action agent------")


rand_rewardList = []
rand_stepList = []

for episodes in range(100):

    obs , info = env.reset()
    total_rewards = 0
    steps= 0
    done = False

    while not done: 
        steps+=1
        action = env.action_space.sample()
        obs , reward , terminated , truncated , info = env.step(action=action)
        total_rewards=total_rewards+reward
        done = terminated or truncated
    
    rand_rewardList.append(total_rewards)
    rand_stepList.append(steps)

env.close()
print(f"Random Mean Reward: {sum(rand_rewardList)/len(rand_rewardList)}")
win=0
for i in range(len(rand_rewardList)):
    if rand_rewardList[i]>0:
        win+=1
print(f"Random Success Rate: {win*100/len(rand_rewardList)}")