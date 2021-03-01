"""
==============================================================================.

CNS2021 PROJECT TEMPLATE

==============================================================================.
                                                                              |
encoding/encoders.py                                                          |
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
from typing import Optional

import torch


class AbstractEncoder(ABC):
    """
    Abstract class to define encoding mechanism.

    The computation procedure should be implemented in the `__call__` method.

    Arguments
    ---------
    time : int
        Length of encoded tensor.
    dt : float, Optional
        Simulation timestep. The default is 1.0.
    device : str, Optional
        The device to do the comutations. The default is "cpu".

    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        self.time = time
        self.dt = dt
        self.device = device

    @abstractmethod
    def __call__(self, data: torch.Tensor) -> None:
        """
        Compute the encoded tensor of the given data.

        Parameters
        ----------
        data : torch.Tensor
            The data tensor to encode.

        Returns
        -------
        None
            It should return the encoded tensor.

        """
        pass
