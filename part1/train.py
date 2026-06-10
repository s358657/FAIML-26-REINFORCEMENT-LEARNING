

import argparse
import csv
import os
import random
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import Agent, Policy


def moving_average(values, window=10):
    if len(values) == 0:
        return []
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start : i + 1])))
    return out


def evaluate(agent, env, episodes=10):
    returns = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += float(reward)
        returns.append(ep_return)
    return float(np.mean(returns)), float(np.std(returns))


def save_plots(save_dir, algo, rewards, losses):
    avg10 = moving_average(rewards, window=10)
    episodes = np.arange(1, len(rewards) + 1)

    plt.figure()
    plt.plot(episodes, rewards, label="episode reward")
    plt.plot(episodes, avg10, label="avg reward last 10")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"Hopper-v4 training reward - {algo}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{algo}_reward.png"))
    plt.close()

    plt.figure()
    plt.plot(episodes, losses, label="loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title(f"Hopper-v4 training loss - {algo}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{algo}_loss.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["reinforce", "reinforce_baseline", "actor_critic"], default="reinforce")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--baseline", type=float, default=20.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--save-dir", type=str, default="results_hopper")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    os.makedirs(args.save_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.render:
        env = gym.make("Hopper-v4", render_mode="human")
    else:
        env = gym.make("Hopper-v4")

    state, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    print("State space:", env.observation_space)
    print("Action space:", env.action_space)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = Policy(state_dim, action_dim)
    baseline = args.baseline if args.algo == "reinforce_baseline" else None
    agent = Agent(
        policy=policy,
        algorithm=args.algo,
        baseline=baseline,
        device=args.device,
        gamma=args.gamma,
        lr=args.lr,
    )

    rewards = []
    lengths = []
    losses = []
    actor_losses = []
    critic_losses = []
    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            action, action_log_prob = agent.get_action(state, evaluation=False)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_outcome(state, next_state, action_log_prob, float(reward), done)

            state = next_state
            ep_reward += float(reward)
            ep_len += 1

        metrics = agent.update_policy()

        rewards.append(ep_reward)
        lengths.append(ep_len)
        losses.append(metrics["loss"])
        actor_losses.append(metrics["actor_loss"])
        critic_losses.append(metrics["critic_loss"])

        if ep == 1 or ep % 10 == 0:
            avg10 = np.mean(rewards[-10:])
            elapsed = time.time() - start_time
            print(
                f"[{args.algo}] episode={ep:4d} "
                f"reward={ep_reward:8.2f} avg10={avg10:8.2f} "
                f"len={ep_len:4d} loss={metrics['loss']:10.4f} "
                f"time={elapsed:7.1f}s"
            )

    mean_eval, std_eval = evaluate(agent, env, episodes=args.eval_episodes)
    elapsed = time.time() - start_time
    print(f"Evaluation over {args.eval_episodes} episodes: mean={mean_eval:.2f}, std={std_eval:.2f}")
    print(f"Total training time: {elapsed:.2f}s")

    
    csv_path = os.path.join(args.save_dir, f"{args.algo}_history_seed{args.seed}.csv")
    avg10_values = moving_average(rewards, window=10)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward", "avg10", "length", "loss", "actor_loss", "critic_loss"])
        for i in range(args.episodes):
            writer.writerow([
                i + 1,
                rewards[i],
                avg10_values[i],
                lengths[i],
                losses[i],
                actor_losses[i],
                critic_losses[i],
            ])

    save_plots(args.save_dir, args.algo, rewards, losses)

    model_path = os.path.join(args.save_dir, f"{args.algo}_policy_seed{args.seed}.pt")
    torch.save(agent.policy.state_dict(), model_path)

    print("Saved:")
    print(" -", csv_path)
    print(" -", os.path.join(args.save_dir, f"{args.algo}_reward.png"))
    print(" -", os.path.join(args.save_dir, f"{args.algo}_loss.png"))
    print(" -", model_path)

    env.close()


if __name__ == "__main__":
    main()
