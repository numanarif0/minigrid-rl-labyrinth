import minigrid  
import numpy as np
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv

from env_wrapper import MakeEnv


def evaluate_function(env, model, env_id, n_episodes=100):
    rewards_list = []
    steps_list = []
    win = 0

    for episode in range(n_episodes):
        obs = env.reset()
        total_rewards = 0
        done = False
        step = 0
        is_success = False

        while not done:
            step += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            done = done[0]
            total_rewards += reward[0]
            if done:
                is_success = info[0].get("is_success", False)

        if is_success:
            win += 1
        rewards_list.append(total_rewards)
        steps_list.append(step)

        status = "FINISH" if is_success else "FAIL"
        print(f"[{env_id}] Ep {episode+1:3d} | {status:6s} | Reward: {total_rewards:6.3f} | Steps: {step}")

    print(f"\n{'='*60}")
    print(f"Env: {env_id}")
    print(f"  Mean Reward:  {np.mean(rewards_list):.3f} (+/- {np.std(rewards_list):.3f})")
    print(f"  Mean Steps:   {np.mean(steps_list):.1f}")
    print(f"  Max Reward:   {max(rewards_list):.3f}")
    print(f"  Success Rate: {win}/{n_episodes} ({win*100/n_episodes:.1f}%)")
    print(f"{'='*60}\n")

    return {
        "env_id": env_id,
        "mean_reward": np.mean(rewards_list),
        "std_reward": np.std(rewards_list),
        "success_rate": win / n_episodes,
        "mean_steps": np.mean(steps_list),
    }


TRAIN_ENV_IDS = [
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-LavaCrossingS9N2-v0",
    "MiniGrid-LavaCrossingS9N3-v0",
    "MiniGrid-LavaCrossingS11N5-v0",
]

UNSEEN_ENV_IDS = [
    "MiniGrid-SimpleCrossingS9N1-v0",
    "MiniGrid-SimpleCrossingS9N3-v0",
    "MiniGrid-SimpleCrossingS11N5-v0",
]

MODEL_ROOT = Path("./best_model/dqn_runs_v2")
model_path = MODEL_ROOT / "best_model.zip"

if not model_path.exists():
    raise FileNotFoundError(f"Model bulunamadi: {model_path}. Once dqntrain_v2.py calistirin.")

print("\n" + "=" * 60)
print("EGITIM ORTAMLARI (DQN V2)")
print("=" * 60)
train_results = []
for env_id in TRAIN_ENV_IDS:
    env = DummyVecEnv([MakeEnv(env_id)])
    env = VecTransposeImage(env)
    model = DQN.load(model_path, env=env)
    result = evaluate_function(env=env, model=model, env_id=env_id)
    train_results.append(result)
    env.close()

print("\n" + "=" * 60)
print("GENELLEME TESTI (Hic Egitilmemis Ortamlar)")
print("=" * 60)
unseen_results = []
for env_id in UNSEEN_ENV_IDS:
    env = DummyVecEnv([MakeEnv(env_id)])
    env = VecTransposeImage(env)
    model = DQN.load(model_path, env=env)
    result = evaluate_function(env=env, model=model, env_id=env_id)
    unseen_results.append(result)
    env.close()

print("\n" + "=" * 60)
print("OZET (DQN V2)")
print("=" * 60)
print(f"\n{'--- Egitim Ortamlari ---'}")
print(f"{'Env':<38} {'Success':>10} {'Reward':>20}")
for r in train_results:
    print(f"{r['env_id']:<38} {r['success_rate']*100:>9.1f}% {r['mean_reward']:>10.3f} +/- {r['std_reward']:.3f}")

print(f"\n{'--- Genelleme (Gorulmemis) ---'}")
print(f"{'Env':<38} {'Success':>10} {'Reward':>20}")
for r in unseen_results:
    print(f"{r['env_id']:<38} {r['success_rate']*100:>9.1f}% {r['mean_reward']:>10.3f} +/- {r['std_reward']:.3f}")
