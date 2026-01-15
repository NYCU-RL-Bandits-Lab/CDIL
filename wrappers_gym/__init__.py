# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A collection of gym wrappers."""
import gym

from wrappers_gym.absorbing_wrapper import AbsorbingWrapper
from wrappers_gym.normalize_action_wrapper import check_and_normalize_box_actions
from wrappers_gym.normalize_state_wrapper import NormalizeStateWrapper
import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper

import sys
sys.path.append("/home/seanma0627/src/off-dynamics-rl")
from gymnasium import spaces
from envs.mujoco.call_mujoco_env  import call_mujoco_env


class GymToGymnasium(gym.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = getattr(env, "reward_range", (-float("inf"), float("inf")))
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)

    def reset(self, seed=None, options=None):
        # Try new API first, fall back to old gym API
        try:
            obs, info = self.env.reset(seed=seed, return_info=True)  # gym>=0.26
            return obs, (info if isinstance(info, dict) else {})
        except TypeError:
            if seed is not None:
                try:
                    self.env.reset(seed=seed)
                except TypeError:
                    if hasattr(self.env, "seed"):
                        self.env.seed(seed)
            obs, _ = self.env.reset()
            return obs, {}

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            return out  # already gymnasium style
        obs, reward, done, info = out
        terminated = bool(done)
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    @property
    def spec(self):
        return getattr(self.env, "spec", None)


def create_il_env(
    env_name,
    shift,
    scale,
    normalized_box_actions: bool = True,
    xml_path=None,
    robot=None,
):
    """Create a gym environment for imitation learning.
    Args:
        env_name: an environment name.
        seed: a random seed.
        shift: a numpy vector to shift observations.
        scale: a numpy vector to scale observations.
    Returns:
        An initialized gym environment.
    """
    if 'hopper' in env_name.lower():
        env = call_mujoco_env({'env_name':"hopper-gravity", 'shift_level': 2.0})
    elif 'ant' in env_name.lower():
        env = call_mujoco_env({'env_name':"ant-friction", 'shift_level': 0.1})
    else:
        env = gym.make(env_name)
    if normalized_box_actions:
        env = check_and_normalize_box_actions(env)

    if shift is not None:
        # Ensure compatibility: convert legacy gym envs to gymnasium style
        # if not isinstance(env, gymnasium.Env):
        #     env = GymToGymnasium(env)
        env = NormalizeStateWrapper(env, shift=shift, scale=scale)

    return AbsorbingWrapper(env)
