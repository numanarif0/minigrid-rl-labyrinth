"""
DQN V3

Degisiklikler (v3'e gore):
- Ortam dagilimi degisti: S11N5 3x (daha fazla zor pratik)
- Toplam adim: 5M â†’ 7M
- eval_freq: 20K â†’ 10K
- n_eval_episodes: 20 â†’ 30
- Mimari ayni: CnnPolicy + ImgObsWrapper + KucukCNN
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

MODEL_ROOT = Path("./best_model/dqn_runs_v3")


TRAIN_ENV_IDS = [
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-LavaCrossingS9N2-v0",
    "MiniGrid-LavaCrossingS9N2-v0",
    "MiniGrid-LavaCrossingS9N3-v0",
    "MiniGrid-LavaCrossingS9N3-v0",
    "MiniGrid-LavaCrossingS11N5-v0",
    "MiniGrid-LavaCrossingS11N5-v0",
    "MiniGrid-LavaCrossingS11N5-v0",
]

TOTAL_TIMESTEPS = 7_000_000


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


def main():
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if th.cuda.is_available():
        print(f"GPU: {th.cuda.get_device_name(0)}")

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    policy_kwargs = dict(
        features_extractor_class=KucukCNN,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
    )

    # Tum ortamlar ayni anda â€” catastrophic forgetting yok
    env = DummyVecEnv([MakeEnv(env_id=e) for e in TRAIN_ENV_IDS])
    env = VecTransposeImage(env)

    # Eval: S11N5 uzerinden (en zor, bunu gecerse digerleri de gecer)
    eval_env = DummyVecEnv([MakeEnv(env_id="MiniGrid-LavaCrossingS11N5-v0")])
    eval_env = VecTransposeImage(eval_env)

    model = DQN(
        policy="CnnPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        learning_rate=1e-4,
        buffer_size=200_000,
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
        tensorboard_log="./dqn_tensorboard_v3/",
    )

    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(MODEL_ROOT),
        eval_freq=10_000,
        n_eval_episodes=30,
        deterministic=True,
        verbose=1,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        tb_log_name="DQN_MultiEnv_v3",
    )

    env.close()
    eval_env.close()
    print(f"\nEgitim tamamlandi. Model: {MODEL_ROOT}/best_model.zip")


if __name__ == "__main__":
    main()
