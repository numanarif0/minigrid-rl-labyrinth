"""
DQN V2 â€” Tek model, tÃ¼m ortamlar aynÄ± anda

Ã–nceki yaklaÅŸÄ±mÄ±n sorunu:
- Sequential curriculum: S11N5 eÄŸitimi S9N1/S9N2/S9N3'Ã¼ unutturuyor
- Tek best_model.zip: son ortam kazanÄ±yor

V3 Ã§Ã¶zÃ¼mÃ¼:
- 4 ortam aynÄ± anda DummyVecEnv iÃ§inde â†’ replay buffer hepsini gÃ¶rÃ¼r
- Catastrophic forgetting yok, tek genel model oluÅŸur
- Orijinal env_wrapper (sadece BFS shaping, lava_penalty yok)
- Tuned DQN hyperparametreler
"""

import minigrid  # noqa: F401
import torch as th
import torch.nn as nn
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from env_wrapper import MakeEnv


device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(f"Device: {device}")
if th.cuda.is_available():
    print(f"CUDA GPU: {th.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {th.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


class KucukCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        in_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            example = th.zeros(1, *observation_space.shape).float()
            n_flatten = self.cnn(example).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.linear(self.cnn(obs.float()))


policy_kwargs = dict(
    features_extractor_class=KucukCNN,
    features_extractor_kwargs=dict(features_dim=128),
    normalize_images=False,
)

# 4 ortam: her biri 2'ÅŸer kopya â†’ toplam 8 ortam aynÄ± anda
# (kolay ortamlar daha fazla temsil edilir, zor ortam da var)
ENV_IDS = [
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-LavaCrossingS9N1-v0",   # 2x â€” kolaydan baÅŸlamak iÃ§in
    "MiniGrid-LavaCrossingS9N2-v0",
    "MiniGrid-LavaCrossingS9N2-v0",   # 2x
    "MiniGrid-LavaCrossingS9N3-v0",
    "MiniGrid-LavaCrossingS9N3-v0",   # 2x
    "MiniGrid-LavaCrossingS11N5-v0",
    "MiniGrid-LavaCrossingS11N5-v0",  # 2x
]

# Eval iÃ§in tek kopya her ortamdan
EVAL_ENV_IDS = [
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-LavaCrossingS9N2-v0",
    "MiniGrid-LavaCrossingS9N3-v0",
    "MiniGrid-LavaCrossingS11N5-v0",
]

TOTAL_TIMESTEPS = 5_000_000
MODEL_ROOT = Path("./best_model/dqn_runs_v2")


def main():
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    # TÃ¼m ortamlar aynÄ± anda
    env = DummyVecEnv([MakeEnv(env_id=e) for e in ENV_IDS])
    env = VecTransposeImage(env)

    # Eval: en zor ortamda deÄŸerlendir (S11N5 iyi giderse diÄŸerleri de gider)
    eval_env = DummyVecEnv([MakeEnv(env_id="MiniGrid-LavaCrossingS11N5-v0")])
    eval_env = VecTransposeImage(eval_env)

    model = DQN(
        policy="CnnPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        learning_rate=1e-4,
        buffer_size=200_000,       # BÃ¼yÃ¼k buffer: tÃ¼m ortamlardan deneyim saklansÄ±n
        learning_starts=10_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
        device=device,
        tensorboard_log="./dqn_tensorboard_v2/",
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(MODEL_ROOT),
        eval_freq=20_000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        tb_log_name="DQN_MultiEnv",
    )

    env.close()
    eval_env.close()
    print(f"\nEgitim tamamlandi. Model: {MODEL_ROOT}/best_model.zip")


if __name__ == "__main__":
    main()
