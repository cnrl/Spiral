"""
Module for neuronal dynamics and populations.
"""

from abc import abstractmethod
from typing import Union, Iterable
from torch.distributions import Normal
import torch
from .neural_populations import AbstractNeuralPopulation


class AbstractEncoder(AbstractNeuralPopulation):
    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        min_input: Union[float, torch.Tensor] = 0.,
        max_input: Union[float, torch.Tensor] = 1.,
        **kwargs
    ) -> None:
        super().__init__(output_shape, **kwargs)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.register_buffer("min", torch.tensor(min_input))
        self.register_buffer("max", torch.tensor(max_input))
        self.register_buffer("s", torch.zeros(*self.output_shape, dtype=torch.bool))


    @abstractmethod
    def forward(self,
            direct_input: torch.Tensor = torch.tensor(False),
            clamps: torch.Tensor = torch.tensor(False),
            unclamps: torch.Tensor = torch.tensor(False)) -> None:
        self.compute_spike(direct_input)
        self.s *= ~unclamps
        self.s += clamps
        for axon_set in self.axon_sets.values():
            axon_set.forward(self.s)


    @abstractmethod
    def compute_spike(self, direct_input: torch.Tensor = torch.tensor(False)) -> None:
        pass
        

    @abstractmethod
    def scale_input(self, data:torch.Tensor) -> torch.Tensor:
        return (data-self.min)/(self.max-self.min)


    @abstractmethod
    def encode(self, data: torch.Tensor) -> None:
        pass


    @abstractmethod
    def reset(self) -> None:
        self.s.zero_()




class LazyEncoder(AbstractEncoder):
    def __init__(
        self,
        shape: Iterable[int],
        **kwargs
    ) -> None:
        super().__init__(
            input_shape=(),
            output_shape=shape,
            **kwargs)


    @abstractmethod
    def compute_spike(self, direct_input: torch.Tensor = torch.tensor(False)) -> None:
        self.s = direct_input.type(torch.bool).reshape(self.output_shape)




class AlwaysOnEncoder(LazyEncoder):
    def __init__(self, shape, **args):
        super().__init__(shape, **args)

    def compute_spike(self, direct_input: torch.Tensor = torch.tensor(False)) -> None:
        self.s = torch.ones(self.output_shape).type(torch.bool)




class AlwaysOffEncoder(LazyEncoder):
    def __init__(self, shape, **args):
        super().__init__(shape, **args)

    def compute_spike(self, direct_input: torch.Tensor = torch.tensor(False)) -> None:
        self.s = torch.zeros(self.output_shape).type(torch.bool)




class TemporaryEncoder(AbstractEncoder):
    def __init__(
        self,
        input_shape: Iterable[int],
        output_shape: Iterable[int],
        time: float,
        dt: float = None,
        **kwargs
    ) -> None:
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            dt = None,
            **kwargs
        )
        self.time = time
        self.step = 0
        self.set_dt(dt)


    def set_dt(self, dt:float):
        super().set_dt(dt)
        if self.dt is not None:
            self.length = self.time//self.dt


    def forward(self,
            direct_input: torch.Tensor = torch.tensor(False),
            clamps: torch.Tensor = torch.tensor(False),
            unclamps: torch.Tensor = torch.tensor(False)):
        assert self.step<self.length, "There is no more encoded information in encoder."
        super().forward(
            direct_input = torch.tensor(False),
            clamps = torch.tensor(False),
            unclamps = torch.tensor(False))
        self.step += 1


    def encode(self, data: torch.Tensor) -> None:
        self.step = 0


    def reset(self) -> None:
        self.step = 0
        super().reset()




class Time2FirstSpikeEncoder(TemporaryEncoder):
    """
    Time-to-First-Spike coding.
    """

    def __init__(
        self,
        shape: Iterable[int],
        time: float,
        **kwargs
    ) -> None:
        super().__init__(
            input_shape=shape,
            output_shape=shape,
            time = time,
            **kwargs
        )


    def compute_spike(self, direct_input: torch.Tensor = torch.tensor(False)) -> None:
        self.s = (self.stage==(self.length-self.step))


    def encode(self, data: torch.Tensor) -> None:
        self.stage = self.scale_input(data)
        self.stage *= self.length
        self.stage = self.stage.round()
        super().encode(data)


    def reset(self) -> None:
        self.stage.zero_()
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




class PositionEncoder(TemporaryEncoder):
    """
    Poisson coding.
    """

    def __init__(
        self,
        shape: Iterable[int],
        time: float,
        k: int = None, # resolution
        mean: Union[float, torch.Tensor] = None,
        std: Union[float, torch.Tensor] = None,
        ignore_threshold: Union[float, torch.Tensor] = 1e-3,
        **kwargs
    ) -> None:
        super().__init__(
            input_shape=shape,
            output_shape=(*shape,k),
            time = time,
            **kwargs
        )
        self.k = k
        if mean is None:
            mean = torch.linspace(self.min, self.max, self.k)
        self.register_buffer("mean", torch.tensor(mean))
        if std is None:
            std = ((self.max-self.min)/(self.k-1))/2
        self.register_buffer("std", torch.tensor(std))
        self.register_buffer("ignore_threshold", torch.tensor(ignore_threshold))


    def compute_spike(self, direct_input: torch.Tensor = torch.tensor(False)) -> None:
        self.s = (self.stage==(self.length-self.step))


    def encode(self, data: torch.Tensor) -> None:
        normal = Normal(self.mean, self.std)
        self.stage = torch.exp(normal.log_prob(data.reshape(*data.shape,1)))
        self.stage[self.stage<self.ignore_threshold] = float('NaN')
        self.stage /= torch.exp(normal.log_prob(self.mean))
        self.stage *= self.length
        self.stage = self.stage.round()


    def reset(self) -> None:
        self.stage.zero_()
        super().reset()


    def decode(self, data: torch.Tensor) -> torch.Tensor: # is buggy
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
    """
    def __init__(
        self,
        shape: Iterable[int],
        rate: Union[int, torch.Tensor] = None,
        **kwargs
    ) -> None:
        super().__init__(
            input_shape=shape,
            output_shape=shape,
            **kwargs
        )
        self.register_buffer("rate", torch.tensor(rate))


    def compute_spike(self, direct_input: torch.Tensor = torch.tensor(False)) -> None:
        self.s = torch.bernoulli(self.stage).type(torch.bool)


    def encode(self, data: torch.Tensor) -> None:
        self.stage = self.scale_input(data)
        self.stage *= self.rate


    def reset(self) -> None:
        self.stage.zero_()
        super().reset()


    def decode(self, data: torch.Tensor) -> torch.Tensor:
        return data.float().mean(axis=0).reshape(self.input_shape) * (self.max-self.min) + self.min