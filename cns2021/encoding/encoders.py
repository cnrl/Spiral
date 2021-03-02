"""
encoding/encoders.py
====================

Module for encoding data into spike.
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


class Intensity2LatencyEncoder(AbstractEncoder):
    """
    Intensity to Latency coding.

    Implement the intensity to latency coding.
    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )
        """
        TODO.

        Add other attributes if needed and fill the body accordingly.
        """

    def __call__(self, data: torch.Tensor) -> None:
        """
        TODO.

        Implement the computation for coding the data. Return resulting tensor.
        """
        pass
