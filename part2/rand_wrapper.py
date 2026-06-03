
from collections import deque

import gymnasium as gym
import numpy as np

class RandomizationWrapper(gym.Wrapper):
    """
    Wrapper that applies randomization to the environment.
    """
    def __init__(
        self,
        env,
        mass_range=(1.0, 1.0),
        config = dict, #dictionary with the sampling range for each parameter 
        mode="none",
    ):
        super().__init__(env)

        self.mode = mode
        self.mass_range = mass_range
        # global limits
        self.mass_min_limit, self.mass_max_limit = mass_range
        # dictionary with the sampling range for each parameter in udr
        self.config = config if config is not None else {}
        self.last_sample_type = None

        if self.mode == "adr":
            self.current_adr_bounds = {}
            self.nominal_values ={"mass":1.0, "friction":1.0} 
            for key,(min_val,max_val) in self.config.items(): # we initialize the adr bounds to the midpoint of the original range
                nominal = self.nominal_values.get(key, (min_val + max_val)/2)
                self.current_adr_bounds[key] = [nominal, nominal]
                self.adr_step = {"mass":0.1, "friction":0.05}  # how much to increase/decrease the range

                self.high_threshold = 0.80
                self.low_threshold = 0.30
                self.window_size = 50 #how many episodes before updating
                self.history = deque(maxlen=self.window_size)# to keep track of the success rate
                self.adr_level = 0

                self.episode_count = 0
                self.print_interval= 100

    # -----------------------
    # Mass Sampling
    # -----------------------

    def _sample_params(self):
        if self.mode == "none":
            return None
        else:
            if self.mode == "udr":
                sample_params = {}
                for key, (min_val, max_val) in self.config.items():
                    sample_params[key] = np.random.uniform(min_val, max_val)
                return sample_params
            elif self.mode == "adr":
                sample_params = {}
                for key, (min_val, max_val) in self.current_adr_bounds.items():
                    sample_params[key] = np.random.uniform(min_val, max_val)
                return sample_params

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        if self.mode =="adr" and done:
            is_success = info.get("is_success", 0.0)
            self.history.append(is_success)

            if len(self.history)>= self.history.maxlen:
                #compute the success rate over the 50 episoeds
                success_rate = np.mean(self.history)
                
                #if the success rate is too high, we need to increase the difficulty in order to keep learning
                #by increasing the range mass and friction
                if success_rate > self.high_threshold : 
                    for key in self.current_adr_bounds:
                        step = self.adr_step.get(key, 0.1)
                        global_min, global_max = self.config[key]
                        min_val, max_val = self.current_adr_bounds[key]
                        new_min = max(global_min, min_val - step)
                        new_max = min(global_max, max_val + step)
                        self.current_adr_bounds[key] = (new_min, new_max)
                    self.adr_level += 1
                    print(f"ADR level increased to {self.adr_level}. New bounds: {self.current_adr_bounds}")
                    
                #if the success rate is too low, we need to decrease the difficulty in order to avoid forgetting what it 
                # has learned so far by decreasing the range mass and friction
                elif success_rate < self.low_threshold and self.adr_level>0:
                    for key in self.current_adr_bounds:
                        step = self.adr_step.get(key, 0.1)
                        global_min, global_max = self.config[key] 
                        min_val, max_val = self.current_adr_bounds[key]
                        
                        new_min = max(global_min, min_val - step)
                        new_max = min(global_max, max_val + step)
                        self.current_adr_bounds[key] = [new_min, new_max]
                    self.adr_level -= 1
                    print(f"ADR level decreased to {self.adr_level}. New bounds: {self.current_adr_bounds}")
                
                self.history.clear()

        
            info["adr_level"] = self.adr_level
            if "mass" in self.current_adr_bounds:
                info["adr_mass_min"] = self.current_adr_bounds["mass"][0]
                info["adr_mass_max"] = self.current_adr_bounds["mass"][1]

        return obs, reward, terminated, truncated, info

    # -----------------------
    # Reset
    # -----------------------

    def reset(self, **kwargs):

        self.episode_count += 1
        new_params = self._sample_params()

        if new_params is not None:

            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            new_mass =float(new_params.get("mass", 1.0))
            new_friction = float(new_params.get("friction", 1.0))

            sim.physics_client.changeDynamics(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
                mass=new_mass,
                lateralFriction=new_friction,
            )
            if self.episode_count % self.print_interval == 0:
                print(
                f"[{self.mode}] mass={new_mass:.2f} "
                f"friction={new_friction:.2f}"
                )

        return super().reset(**kwargs)
