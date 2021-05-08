"""
Module for neuronal dynamics and populations.
"""

from functools import reduce
from abc import abstractmethod
from operator import mul
from typing import Union, Iterable
from torch.distributions import Normal

import torch


class AbstractEncoder(torch.nn.Module):
    """
    Base class for implementing neural populations.

    Make sure to implement the abstract methods in your child class. Note that this template\
    will give you homogeneous neural populations in terms of excitations and inhibitions. You\
    can modify this by removing `is_inhibitory` and adding another attribute which defines the\
    percentage of inhibitory/excitatory neurons or use a boolean tensor with the same shape as\
    the population, defining which neurons are inhibitory.

    Arguments
    ---------
    shape : Iterable of int
        Define the topology of neurons in the population.
    spike_trace : bool, Optional
        Specify whether to record spike traces. The default is True.
    additive_spike_trace : bool, Optional
        Specify whether to record spike traces additively. The default is True.
    tau_s : float or torch.Tensor, Optional
        Time constant of spike trace decay. The default is 15.0.
    trace_scale : float or torch.Tensor, Optional
        The scaling factor of spike traces. The default is 1.0.
    is_inhibitory : False, Optional
        Whether the neurons are inhibitory or excitatory. The default is False.
    learning : bool, Optional
        Define the training mode. The default is True.

    """

    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        min_input: Union[float, torch.Tensor] = 0.,
        max_input: Union[float, torch.Tensor] = 1.,
        length: int = None,
        is_excitatory: Union[bool, torch.Tensor] = True,
        **kwargs
    ) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.length = length
        self.register_buffer("min", torch.tensor(min_input))
        self.register_buffer("max", torch.tensor(max_input))
        self.register_buffer("stage", torch.zeros(*self.input_shape, dtype=torch.bool))
        self.register_buffer("s", torch.zeros(*self.output_shape, dtype=torch.bool))
        self.register_buffer("is_excitatory", torch.tensor(is_excitatory))

    @abstractmethod
    def forward(self) -> None:
        """
        Compute the encoded tensor of the given data.

        Parameters
        ----------
        data : torch.Tensor
            The data tensor to encode.

        Returns
        -------
        None
            Set self.s.

        """
        pass

    @abstractmethod
    def encode(self, data: torch.Tensor) -> None:
        """
        Returns
        -------
        None
        """

        self.stage = (data-self.min)/(self.max-self.min)


    @abstractmethod
    def reset(self) -> None:
        """
        Refractor and reset the neurons.

        Returns
        -------
        None

        """
        self.s.zero_()


class Time2FirstSpikeEncoder(AbstractEncoder):
    """
    Time-to-First-Spike coding.

    Implement Time-to-First-Spike coding.
    """

    def __init__(
        self,
        shape: Iterable[int],
        **kwargs
    ) -> None:
        super().__init__(
            input_shape=shape,
            output_shape=shape,
            **kwargs
        )
        self.t = 0

    def forward(self) -> None:
        self.s = (self.stage==(self.length-self.t))
        self.t += 1

    def encode(self, data: torch.Tensor) -> None:
        super().encode(data)
        self.stage *= self.length
        self.stage = self.stage.round()

    def reset(self) -> None:
        self.t = 0
        super().reset()

    def decode(self, data: torch.Tensor) -> torch.Tensor:
        d = data.reshape(self.length,-1)
        times,neurons = torch.where(d)
        spike_times = torch.cat([neurons.reshape(*neurons.shape,1),times.reshape(*times.shape,1)], dim=1)
        spike_times = spike_times[spike_times[:,0].argsort()]
        d = torch.zeros(d.shape[1]) - 1
        d[spike_times[:,0]] = spike_times[:,1].type(torch.float)
        d = self.length - d
        return d.reshape(self.input_shape)


class PositionEncoder(AbstractEncoder):
    """
    Poisson coding.

    Implement Poisson coding.
    """

    def __init__(
        self,
        shape: Iterable[int],
        k: int = None, # resolution
        mean: Union[float, torch.Tensor] = None,
        std: Union[float, torch.Tensor] = None,
        ignore_threshold: Union[float, torch.Tensor] = 1e-3,
        **kwargs
    ) -> None:
        super().__init__(
            input_shape=shape,
            output_shape=(*shape,k),
            **kwargs
        )
        self.k = k
        if mean is None:
            mean = torch.linspace(self.min, self.max, self.k)
        self.register_buffer("mean", torch.tensor(mean))
        if std is None:
            std = ((self.max-self.min)/(self.k-1))/3
        self.register_buffer("std", torch.tensor(std))
        self.register_buffer("ignore_threshold", torch.tensor(ignore_threshold))
        self.t = 0

    def forward(self) -> None:
        self.s = (self.stage==(self.length-self.t))
        self.t += 1

    def encode(self, data: torch.Tensor) -> None:
        normal = Normal(self.mean, self.std)
        self.stage = torch.exp(normal.log_prob(data.reshape(*data.shape,1)))
        self.stage[self.stage<self.ignore_threshold] = float('NaN')
        self.stage /= torch.exp(normal.log_prob(self.mean))
        self.stage *= self.length
        self.stage = self.stage.round()

    def reset(self) -> None:
        super().reset()

    def decode(self, data: torch.Tensor) -> torch.Tensor:
        d = data.reshape(self.length,-1)
        times,neurons = torch.where(d)
        spike_times = torch.cat([neurons.reshape(*neurons.shape,1),times.reshape(*times.shape,1)], dim=1)
        spike_times = spike_times[spike_times[:,0].argsort()]
        d = torch.zeros(d.shape[1]) - 1
        d[spike_times[:,0]] = spike_times[:,1].type(torch.float)
        d = d.reshape(-1, self.k)
        d[d==-1] = float('NaN')
        d = self.length - d
        normal = Normal(self.mean, self.std)
        d /= self.length
        d *= torch.exp(normal.log_prob(self.mean))
        d_left = normal.icdf(torch.tensor(d))
        d_right = normal.icdf(torch.tensor(1-d))
        d = torch.cat([d_left.reshape(*d_left.shape,1),d_right.reshape(*d_right.shape,1)], dim=-1)
        mean = d.nansum(axis=(-2,-1))/(~d.isnan()).sum(axis=(-2,-1))
        diff = torch.abs(d-mean.reshape(*mean.shape,1,1))
        diff = diff.reshape(-1, diff.shape[-1])
        d = d.reshape(-1, d.shape[-1])
        d = d[torch.arange(d.shape[0]), diff.argmin(axis=-1)]
        d = d.reshape(-1, self.k)
        d = d.nansum(axis=-1)/(~d.isnan()).sum(axis=-1)
        d = d.reshape(self.input_shape)
        return d

class PoissonEncoder(AbstractEncoder):
    """
    Poisson coding.

    Implement Poisson coding.
    """

    def __init__(
        self,
        shape: Iterable[int],
        max_rate: Union[int, torch.Tensor] = None,
        **kwargs
    ) -> None:
        super().__init__(
            input_shape=shape,
            output_shape=shape,
            **kwargs
        )
        if max_rate is None:
            max_rate = self.length
        self.register_buffer("max_rate", torch.tensor(max_rate))

    def forward(self) -> None:
        self.s = torch.bernoulli(self.stage).type(torch.bool)

    def encode(self, data: torch.Tensor) -> None:
        super().encode(data)
        self.stage *= self.max_rate/self.length

    def reset(self) -> None:
        super().reset()

    def decode(self, data: torch.Tensor) -> torch.Tensor:
        return data.sum(axis=0).reshape(self.input_shape)