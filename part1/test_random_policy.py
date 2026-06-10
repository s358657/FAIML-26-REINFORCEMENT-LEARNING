

import argparse

import gymnasium as gym
import numpy as np


def get_body_names(model):
    if hasattr(model, "body_names"):
        return list(model.body_names)

    try:
        import mujoco
        return [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            for i in range(model.nbody)
        ]
    except Exception:
        return [f"body_{i}" for i in range(model.nbody)]


def print_env_info(env):
    print("State space:", env.observation_space)
    print("Action space:", env.action_space)

    model = env.unwrapped.model
    body_names = get_body_names(model)

    print("Number of DoFs (model.nv):", model.nv)
    print("Number of actuators (model.nu):", model.nu)
    print("Body masses:")
    for name, mass in zip(body_names, model.body_mass):
        print(f"  {name:>12s}: {mass:.6f}")

    if hasattr(model, "body_dofnum"):
        print("DoFs for each body:")
        for name, dofnum in zip(body_names, model.body_dofnum):
            print(f"  {name:>12s}: {int(dofnum)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Hopper-v4")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.render:
        env = gym.make(args.env, render_mode="human")
    else:
        env = gym.make(args.env, render_mode="rgb_array")

    print_env_info(env)

    returns = []
    lengths = []
    for ep in range(args.episodes):
        done = False
        state, _ = env.reset()
        ep_return = 0.0
        ep_len = 0

        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += float(reward)
            ep_len += 1

            if args.render:
                env.render()

        returns.append(ep_return)
        lengths.append(ep_len)

    print(f"Random policy over {args.episodes} episodes:")
    print(f"  mean return = {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"  mean length = {np.mean(lengths):.2f}")

    env.close()


if __name__ == "__main__":
    main()
