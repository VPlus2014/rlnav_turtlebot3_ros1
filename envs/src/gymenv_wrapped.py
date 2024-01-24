#!/usr/bin/env python

import os
import time
from typing import Any, Callable, Dict, List, Optional, SupportsFloat, Tuple
from numpy import dtype, float32, ndarray
import rospy
import numpy as np

from .utils import *
from .gymenv_core import (
    Env_stage_core,
    EnvInfo,
    TurtlebotAction,
    Env_stage_core_init_args,
)
import gymnasium as gym_
from gymnasium.wrappers.flatten_observation import FlattenObservation
from .gymenv_wrappers import RescaleObservation


class Env_cont(gym_.Env):
    def __init__(
        self,
        config: Optional[Env_stage_core_init_args] = None,
        seed: Optional[int] = None,
    ):
        """经典形式,输入输出均为向量化数据"""
        super().__init__()
        super().reset(seed=seed)  # 初始化随机数发生器

        self.__core = core = Env_stage_core(config=config, seed=seed)
        self.__ANG_VEL_MAX = core.ANGULAR_VEL_MAX
        self.__LIN_VEL_CONST = core.LINEAR_VEL_MAX
        self._action = TurtlebotAction(**core.action_space.sample())
        self.action_space = gym_.spaces.Box(
            -1, 1, shape=[1], dtype=core._config.dtype, seed=self.np_random
        )

        env = FlattenObservation(core)  # 观测值向量化
        env = RescaleObservation(
            env,
            np.zeros_like(env.observation_space.low),
            np.ones_like(env.observation_space.low),
        )  # 观测值归一化
        self.env = env
        self.observation_space = self.env.observation_space
        return

    def reset(
        self, *, seed: int = None, options: Dict[str, Any] = None
    ) -> Tuple[ndarray, Dict[str, Any]]:
        super().reset(seed=seed)  # 重置随机数发生器
        return self.env.reset(seed=seed)

    def step(
        self,
        action: float,
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        action_ = self._action
        action_.lin_vel.itemset(self.__LIN_VEL_CONST)
        action_.ang_vel.itemset(float(action) * self.__ANG_VEL_MAX)
        rst = self.env.step(action_.__dict__)
        return rst

    def close(self):
        self.env.close()

    @property
    def log_dir(self):
        return self.__core.log_dir


class Env_disc(gym_.Env):
    def __init__(
        self,
        action_size: int = 3,
        config: Optional[Env_stage_core_init_args] = None,
        seed: Optional[int] = None,
    ):
        assert action_size >= 2, ("expected action_size>=2, but get", action_size)
        super().__init__()
        super().reset(seed=seed)

        env = Env_cont(config=config, seed=seed)
        self.env = env
        self.action_size = action_size
        self.action_space = gym_.spaces.Discrete(action_size, seed=self.np_random)
        self.observation_space = env.observation_space

    def reset(
        self, *, seed: int = None, options: Dict[str, Any] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed)  # 初始化随机数发生器
        return self.env.reset(seed=seed)

    def step(
        self,
        action: int,
    ) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        core = self.env
        core_action_space: gym_.spaces.Box = core.action_space
        # 动作转换
        action_ = core_action_space.low + (action / (self.action_size - 1)) * (
            core_action_space.high - core_action_space.low
        )
        return core.step(action_)

    def close(self):
        self.env.close()

    @property
    def log_dir(self):
        return self.env.log_dir


if __name__ == "__main__":
    raise Exception(f"不允许直接运行 {__file__}")
