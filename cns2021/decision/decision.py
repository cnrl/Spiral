"""
==============================================================================.

CNS2021 PROJECT TEMPLATE

==============================================================================.
                                                                              |
decision/decision.py                                                          |
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


class AbstractDecision(ABC):
    """
    Abstract class to define decision making strategy.

    Make sure to implement the abstract methods in your child class.
    """

    @abstractmethod
    def compute(self, **kwargs) -> None:
        """
        Infer the decision to be made.

        Returns
        -------
        None
            It should return the decision result.

        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Update the variables after making the decision.

        Returns
        -------
        None

        """
        pass


"""
TODO.

1. Implement the dynamics of decision making. You are free to define your stru-
   cture here.
2. Make sure to implement winner-take-all mechanism.
"""
