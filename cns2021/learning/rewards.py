"""
==============================================================================.

CNS2021 PROJECT TEMPLATE

==============================================================================.
                                                                              |
learning/rewards.py                                                           |
                                                                              |
Copyright (C) 2020-2021 CNRL <cnrl.ut.ac.ir>                                  |
                                                                              |
This program is free software:  you can redistribute it and/or modify it under|
the terms of the  GNU General Public License as published by the Free Software|
Foundation, either version 3 of the License, or(at your option) any later ver-|
sion.                                                                         |
                                                                              |
This file has been provided for educational purpose.  The aim of this template|
is to help students with developing a Spiking Neural Network framework from s-|
cratch and learn the basics. Follow the documents and comments to complete the|
code.                                                                         |
                                                                              |
==============================================================================.
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


"""
TODO.

Define your reward functions here.
"""
