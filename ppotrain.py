import gymnasium as gym
from stable_baselines3 import PPO
import time
from minigrid.wrappers import FlatObsWrapper
import minigrid
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from env_wrapper import LavaPenaltyWarapper , MakeEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecTransposeImage

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Kullanılan device: {device}")
if th.cuda.is_available():
    print(f"CUDA GPU: {th.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {th.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
class KucukCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        # Gelen görüntü zaten (3, 7, 7) formatında!
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with th.no_grad():
            # observation_space.shape → (3, 7, 7) zaten doğru format
            exmpl = th.zeros(1, *observation_space.shape).float()
            n_flatten = self.cnn(exmpl).shape[1]  # permute yok!
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
    
    def forward(self, obs):
        # permute yok! SB3 zaten halletti
        return self.linear(self.cnn(obs.float()))


policy_kwargs = dict(
    features_extractor_class=KucukCNN,
    features_extractor_kwargs=dict(features_dim=128)
)

env = DummyVecEnv([
    MakeEnv("MiniGrid-LavaCrossingS9N1-v0"),  
    MakeEnv("MiniGrid-LavaCrossingS9N2-v0"),  
    MakeEnv("MiniGrid-LavaCrossingS9N3-v0"),  
    MakeEnv("MiniGrid-LavaCrossingS11N5-v0"), 
])

#env = VecNormalize(venv=env,norm_obs=True,norm_reward=True)

#lr_schedule = lambda progress: 1e-4 * progress


model = PPO(policy="CnnPolicy",policy_kwargs=policy_kwargs,env=env,learning_rate=2.5e-4,n_steps=1024,
            batch_size=256,n_epochs=5,gamma=0.995,ent_coef=0.03,verbose=1,device=device)

eval_env = DummyVecEnv([MakeEnv("MiniGrid-LavaCrossingS9N2-v0")])
eval_env = VecTransposeImage(eval_env)

#eval_env = VecNormalize(venv=eval_env,norm_obs=True,norm_reward=True)


evalCallBack = EvalCallback(eval_env=eval_env,best_model_save_path="./best_model/2",
                            eval_freq=10_000,n_eval_episodes=10,verbose=1)

model.learn(total_timesteps=10_000_000,callback=evalCallBack)
