import argparse
import os
import random
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, PPO 
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from gymnasium.wrappers import RecordVideo
import panda_gym  # noqa: F401 - required so Panda envs are registered


def evaluate(model_path: str, n_episodes: int, deterministic: bool, render: bool, env_type: str) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Make sure you saved your trained model with model.save(...)."
        )

    render_mode = "human" if render else "rgb_array"
    env = gym.make("PandaPush-v3", render_mode=render_mode, type=env_type, reward_type="dense")
    #env = RecordVideo(env, video_folder="eval_videos", episode_trigger=lambda episode_id: True)
    model = SAC.load(model_path, env=env)

    print(f"Learning rate: {model.learning_rate}")
    print(f"Batch size: {model.batch_size}")
    print(f"Gamma (discount factor): {model.gamma}")
    print(f"Tau (soft update coefficient): {model.tau}")


    
        
    

    episode_returns = []
    successes = []

    for episode in range(1, n_episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_return = 0.0

        while not (terminated or truncated):
            action,_ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)

        episode_returns.append(episode_return)

        if isinstance(info, dict) and "is_success" in info:
            successes.append(float(info["is_success"]))

        print(f"Episode {episode:03d} | return = {episode_return:.3f}")

    env.close()

    returns = np.array(episode_returns, dtype=np.float32)
    print("\n=== Evaluation summary ===")
    print(f"Episodes: {n_episodes}")
    print(f"Mean return: {returns.mean():.3f}")
    print(f"Std return:  {returns.std():.3f}")
    print(f"Min return:  {returns.min():.3f}")
    print(f"Max return:  {returns.max():.3f}")

    if successes:
        success_rate = float(np.mean(successes))
        print(f"Success rate: {success_rate:.2%}")
    
    


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAC on PandaPush-v3")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a PPO model zip file (e.g., ppo_panda_push.zip)",
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=500, 
        help="Number of eval episodes"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy sampling instead of deterministic actions",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render with a window (render_mode='human')",
    )
    parser.add_argument(
        "--env-type",
        type=str, default="target",
        choices=["source", "target"],
        help="Type of environment to evaluate on (default: target)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    evaluate(
        model_path=args.model_path,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
        env_type=args.env_type,
    )

