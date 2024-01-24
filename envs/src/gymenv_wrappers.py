import gymnasium as gym_
from gymnasium.wrappers.rescale_action import RescaleAction
from typing import Any, Callable, Dict, List, Optional, SupportsFloat, Tuple, Union

import numpy as np


class RescaleObservation(gym_.ObservationWrapper, gym_.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym_.Env,
        min_observation: Union[float, int, np.ndarray],
        max_observation: Union[float, int, np.ndarray],
    ):
        """Initializes the :class:`RescaleObservation` wrapper.

        Args:
            env (Env): The environment to apply the wrapper
            min_obs (float, int or np.ndarray): The min values for each observation. This may be a numpy array or a scalar.
            max_obs (float, int or np.ndarray): The max values for each observation. This may be a numpy array or a scalar.
        """
        assert isinstance(
            env.observation_space, gym_.spaces.Box
        ), f"expected Box observation space, got {type(env.observation_space)}"
        assert np.less_equal(min_observation, max_observation).all(), (
            min_observation,
            max_observation,
        )
        gym_.utils.RecordConstructorArgs.__init__(
            self, min_observation=min_observation, max_observation=max_observation
        )
        gym_.ObservationWrapper.__init__(self, env)

        self.min_observation = (
            np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype)
            + min_observation
        )
        self.max_observation = (
            np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype)
            + max_observation
        )

        self.observation_space = gym_.spaces.Box(
            low=min_observation,
            high=max_observation,
            shape=env.observation_space.shape,
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Rescales the observation affinely from  [:attr:`min_observation`, :attr:`max_observation`] to the observation space of the base environment, :attr:`env`.

        Args:
            observation: The observation to rescale

        Returns:
            The rescaled observation
        """
        origin_low = self.env.observation_space.low
        origin_span = self.env.observation_space.high - origin_low
        assert np.all(origin_span > 0), ("invalid observation_space span", origin_span)

        rescaled_low = self.min_observation
        rescaled_high = self.max_observation
        observation = rescaled_low + (rescaled_high - rescaled_low) * (
            (observation - origin_low) / origin_span
        )
        observation = np.clip(observation, rescaled_low, rescaled_high)
        return observation
