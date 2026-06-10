import argparse
from collections import deque
import os
import random

import gymnasium as gym
import numpy as np
import torch
import panda_gym   # type: ignore[import-not-found]
from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from rand_wrapper import RandomizationWrapper

class ADRLoggerCallback(BaseCallback):
    # used to log the current ADR bounds into terminal and tensorboard
    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        if "adr_level" in info:
            self.logger.record("adr/level", info["adr_level"])
            self.logger.record("adr/mass_min", info["adr_mass_min"])
            self.logger.record("adr/mass_max", info["adr_mass_max"])
        return True

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC on PandaPush-v3")
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="none",
        choices=["none", "udr", "adr"],
        help="Sampling strategy for the object mass",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="source",
        choices=["source", "target"],
        help="PandaPush environment type",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Number of training timesteps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    if args.env_type == "target":
        train_config = {
            'mass':(5.0,5.0),
            'friction':(1.0,1.0)
        }
        actual_strategy = "none"
    else:
        if args.sampling_strategy == "udr":
            train_config = {
                'mass':(1.0, 8.0),
                'friction':(1.0,1.0)
            }
            actual_strategy = "udr"
        elif args.sampling_strategy == "adr":
            train_config = {
                'mass':(0.5, 8.0),
                'friction':(0.5, 1.5)
            }
            actual_strategy = "adr"
        else:
            train_config = {
                'mass':(1.0, 1.0),
                'friction':(1.0, 1.0)
            }
            actual_strategy = "none"

    env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type=args.env_type,
        reward_type="dense",
    )

    eval_env = gym.make(
        "PandaPush-v3",
        render_mode="rgb_array",
        type='target',
        reward_type="dense",
    )

    log_dir ="./tb_logs/"
    save_dir = "./models/"
    os.makedirs(save_dir, exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=os.path.join(save_dir, f"SAC_clean_{args.env_type}"),
        log_path=log_dir,
        eval_freq=10_000,        
        n_eval_episodes=5,      
        deterministic=True,      
        verbose=1
    )
    adr_logger = ADRLoggerCallback()
    callbacks = CallbackList([eval_callback, adr_logger])

    #TODO: add randomization wrapper here
    env = RandomizationWrapper(env, mode=actual_strategy, config=train_config)
    #TODO: create model and train it
    model_sac = SAC("MultiInputPolicy", env, verbose=1, device=device, tensorboard_log=log_dir, seed=random_seed).learn(args.timesteps, callback=callbacks, tb_log_name=f"SAC_{args.sampling_strategy}_{args.env_type}")
    obs = env.reset()
        
    save_name_sac = f"sac_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k"
    # TODO: 
    save_name_sac = os.path.join(save_dir, f"sac_push_{args.sampling_strategy}_{args.env_type}_{args.timesteps // 1000}k")
    model_sac.save(save_name_sac)
    

if __name__ == "__main__":
    main()