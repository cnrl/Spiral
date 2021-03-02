"""
Module for reward dynamics.

TODO.

Define your reward functions here.
"""

from abc import ABC, abstractmethod


class AbstractReward(ABC):
    """
    Abstract class to define reward function.

    Make sure to implement the abstract methods in your child class.
    """

    @abstractmethod
    def compute(self, **kwargs) -> None:
        """
        Compute the reward.

        Returns
        -------
        None
            It should return the computed reward value.

        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Update the internal variables.

        Returns
        -------
        None

        """
        pass
