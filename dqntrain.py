import minigrid  # noqa: F401  # registers MiniGrid environments
import torch as th
import torch.nn as nn
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from env_wrapper_partial import MakePartialEnv


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


ENV_IDS = [
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-LavaCrossingS9N2-v0",
    "MiniGrid-LavaCrossingS9N3-v0",
    "MiniGrid-LavaCrossingS11N5-v0",
]

N_STACK = 4
TIMESTEPS = [1_000_000, 1_500_000, 2_000_000, 2_500_000]

MODEL_ROOT = Path(f"./best_model/dqn_runs_stack{N_STACK}")


def train_curriculum(env_ids, timesteps):
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    for env_id, total_steps in zip(env_ids, timesteps):
        env = DummyVecEnv([MakePartialEnv(env_id=env_id)])
        env = VecTransposeImage(env)
        env = VecFrameStack(env, n_stack=N_STACK)

        eval_env = DummyVecEnv([MakePartialEnv(env_id=env_id)])
        eval_env = VecTransposeImage(eval_env)
        eval_env = VecFrameStack(eval_env, n_stack=N_STACK)

        best_model_dir = MODEL_ROOT / env_id
        best_model_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = best_model_dir / "best_model.zip"

        if best_model_path.exists():
            print(f"Loading existing model for {env_id}")
            model = DQN.load(
                path=best_model_path,
                env=env,
                device=device,
                tensorboard_log="./dqn_tensorboard/",
            )
        else:
            print(f"Creating new model for {env_id}")
            model = DQN(
                policy="CnnPolicy",
                policy_kwargs=policy_kwargs,
                env=env,
                learning_rate=2.5e-4,
                buffer_size=200_000,
                learning_starts=5_000,
                batch_size=128,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=500,
                exploration_fraction=0.3,
                exploration_final_eps=0.1,
                verbose=1,
                device=device,
                tensorboard_log="./dqn_tensorboard/",
            )

        eval_callback = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=str(best_model_dir),
            eval_freq=20_000,
            n_eval_episodes=20,
            deterministic=True,
            verbose=1,
        )

        model.learn(
            total_timesteps=total_steps,
            callback=eval_callback,
            tb_log_name=f"DQN_{env_id}",
        )

        env.close()
        eval_env.close()


train_curriculum(env_ids=ENV_IDS, timesteps=TIMESTEPS)
