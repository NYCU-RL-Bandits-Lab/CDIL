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
import gymnasium
import gym
from wrappers.absorbing_wrapper import AbsorbingWrapper
from wrappers.normalize_action_wrapper import check_and_normalize_box_actions
from wrappers.normalize_state_wrapper import NormalizeStateWrapper
import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import GymWrapper


class GymToGymnasium(gymnasium.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = getattr(
            env, "reward_range", (-float("inf"), float("inf")))
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)

    def reset(self, seed=None, options=None):
        # Try new API first, fall back to old gym API
        try:
            obs, info = self.env.reset(
                seed=seed, return_info=True)  # gym>=0.26
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


def create_il_env(env_name, shift, scale, normalized_box_actions: bool = True, xml_path=None, robot=None):
    """Create a gym environment for imitation learning.
    Args:
        env_name: an environment name.
        seed: a random seed.
        shift: a numpy vector to shift observations.
        scale: a numpy vector to scale observations.
    Returns:
        An initialized gym environment.
    """
    if xml_path:
        env = gymnasium.make(env_name, xml_file=xml_path)
    elif robot:
        controller_config = load_controller_config(default_controller='JOINT_VELOCITY')
        env_suite = suite.make(
            env_name,
            robots=robot,
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_object_obs=True,
            use_camera_obs=False,
            reward_shaping=True,
            horizon=500,
        )
        keys = ["object-state"]
        for idx in range(len(env_suite.robots)):
            keys.append(f"robot{idx}_proprio-state")
        env = GymWrapper(env_suite, keys=keys)
    else:
        env = gym.make(env_name)
    if normalized_box_actions:
        env = check_and_normalize_box_actions(env)

    if shift is not None:
        env = NormalizeStateWrapper(env, shift=shift, scale=scale)

    return AbsorbingWrapper(env)
