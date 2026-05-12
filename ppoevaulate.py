import gymnasium as gym
import minigrid
import numpy as np
from stable_baselines3 import PPO
from env_wrapper import MakeEnv
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv


def _reset_compat(env):
    reset_output = env.reset()
    if isinstance(reset_output, tuple):
        return reset_output[0]
    return reset_output


def _step_compat(env, action):
    step_output = env.step(action)
    if len(step_output) == 5:
        obs, reward, terminated, truncated, info = step_output
        done = terminated | truncated
        return obs, reward, done, info
    return step_output


def evaulateFunction(env, model, env_id, n_episodes=100):
    rewardsList = []
    stepsList = []
    win = 0

    for episode in range(n_episodes):
        obs = _reset_compat(env)
        total_rewards = 0
        done = False
        step = 0
        is_success = False

        while not done:
            step += 1
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = _step_compat(env, action)
            done = done[0] if isinstance(done, (list, tuple, np.ndarray)) else done
            total_rewards += reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward

            if done:
                info_item = info[0] if isinstance(info, (list, tuple)) else info
                is_success = info_item.get('is_success', False)

        if is_success:
            win += 1

        rewardsList.append(total_rewards)
        stepsList.append(step)

        status = "FINISH" if is_success else "FAIL"
        print(f"[{env_id}] Ep {episode+1:3d} | {status:6s} | Reward: {total_rewards:6.3f} | Steps: {step}")

    print(f"\n{'='*60}")
    print(f"Env: {env_id}")
    print(f"  Mean Reward:  {np.mean(rewardsList):.3f} (+/- {np.std(rewardsList):.3f})")
    print(f"  Mean Steps:   {np.mean(stepsList):.1f}")
    print(f"  Max Reward:   {max(rewardsList):.3f}")
    print(f"  Success Rate: {win}/{n_episodes} ({win*100/n_episodes:.1f}%)")
    print(f"{'='*60}\n")

    return {
        "env_id": env_id,
        "mean_reward": np.mean(rewardsList),
        "std_reward": np.std(rewardsList),
        "success_rate": win / n_episodes,
        "mean_steps": np.mean(stepsList),
    }


env_ids = [
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-LavaCrossingS9N2-v0",
    "MiniGrid-LavaCrossingS9N3-v0",
    "MiniGrid-LavaCrossingS11N5-v0",
]

results = []
for env_id in env_ids:
    env = DummyVecEnv([MakeEnv(env_id)])
    env = VecTransposeImage(env)
    model = PPO.load("./best_model/2/best_model", env=env)
    result = evaulateFunction(env=env, model=model, env_id=env_id)
    results.append(result)
    env.close()

print("\n" + "=" * 60)
print("OZET")
print("=" * 60)
print(f"{'Env':<35} {'Success':>10} {'Reward':>20}")
for r in results:
    print(f"{r['env_id']:<35} {r['success_rate']*100:>9.1f}% {r['mean_reward']:>10.3f} +/- {r['std_reward']:.3f}")
