"""
Module for neuronal dynamics and populations.
"""

from abc import abstractmethod
from typing import Union, Iterable, Callable
from torch.distributions import Normal
import torch
from .neural_populations import AbstractNeuralPopulation


class AbstractEncoder(AbstractNeuralPopulation):
    def __init__(
        self,
        name: str,
        input_shape: Iterable[int],
        output_shape: Iterable[int] = None,
        normalize: bool = True,
        filt: Callable = lambda x: x,
        **kwargs
    ) -> None:
        super().__init__(
            name=name,
            shape=output_shape if output_shape is not None else filt(torch.zeros(input_shape)).shape,
            **kwargs
        )
        self.filter = filt
        self.input_shape = input_shape
        self.output_shape = output_shape if output_shape is not None else self.filter(torch.zeros(input_shape)).shape
        self.register_buffer("s", torch.zeros(*self.output_shape, dtype=torch.bool))
        self.normalize = normalize


    @abstractmethod
    def forward(self,
            direct_input: torch.Tensor = torch.tensor(False),
            clamps: torch.Tensor = torch.tensor(False),
            unclamps: torch.Tensor = torch.tensor(False)) -> None:
        self.compute_spike(direct_input)
        self.s *= ~unclamps
        self.s += clamps
        for axon in self.axons.values():
            axon.forward(self.s)


    @abstractmethod
    def compute_spike(self, direct_input: torch.Tensor = torch.tensor(False)) -> None:
        pass
        

    def encode(self, data: torch.Tensor) -> None:
        data = self.filter(data)
        if self.normalize:
            data -= data.min()
            data /= data.max()
        self.encode_processed_data(data)


    @abstractmethod
    def encode_processed_data(self, data: torch.Tensor) -> None:
        pass




class LazyEncoder(AbstractEncoder):
    def __init__(
        self,
        name: str,
        shape: Iterable[int],
        **kwargs
    ) -> None:
        super().__init__(
            name=name,
            input_shape=(),
            output_shape=shape,
            **kwargs)


    @abstractmethod
    def compute_spike(self, direct_input: torch.Tensor = torch.tensor(False)) -> None:
        self.s = direct_input.type(torch.bool).reshape(self.output_shape)




class AlwaysOnEncoder(LazyEncoder):
    def __init__(
        self,
        name: str,
        shape: Iterable[int],
        **args):
        super().__init__(
            name=name,
            shape=shape,
            **args
        )

    def compute_spike(self, direct_input: torch.Tensor = torch.tensor(False)) -> None:
        self.s = torch.ones(self.output_shape).type(torch.bool)




class AlwaysOffEncoder(LazyEncoder):
    def __init__(
        self,
        name: str,
        shape: Iterable[int],
        **args):
        super().__init__(
            name=name,
            shape=shape,
            **args
        )

    def compute_spike(self, direct_input: torch.Tensor = torch.tensor(False)) -> None:
        self.s = torch.zeros(self.output_shape).type(torch.bool)




class TemporaryEncoder(AbstractEncoder):
    def __init__(
        self,
        name: str,
        time: float,
        input_shape: Iterable[int],
        output_shape: Iterable[int] = None,
        dt: float = None,
        **kwargs
    ) -> None:
        super().__init__(
            name=name,
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
        super().encode(data)


    def reset(self) -> None:
        self.step = 0
        super().reset()




class Time2FirstSpikeEncoder(TemporaryEncoder):
    """
    Time-to-First-Spike coding.
    """

    def __init__(
        self,
        name: str,
        shape: Iterable[int],
        time: float,
        **kwargs
    ) -> None:
        super().__init__(
            name=name,
            input_shape=shape,
            time = time,
            **kwargs
        )


    def compute_spike(self, direct_input: torch.Tensor = torch.tensor(False)) -> None:
        self.s = (self.stage==(self.length-self.step))


    def encode_processed_data(self, data: torch.Tensor) -> None:
        self.stage = data * self.length
        self.stage = self.stage.round()


    def reset(self) -> None:
        if hasattr(self, 'stage'): del self.stage
        super().reset()


    def decode(self, data: torch.Tensor) -> torch.Tensor:
        d = data.reshape(int(self.length),-1)
        times,neurons = torch.where(d)
        spike_times = torch.cat([neurons.reshape(*neurons.shape,1),times.reshape(*times.shape,1)], dim=1)
        spike_times = spike_times[spike_times[:,0].argsort()]
        d = torch.zeros(d.shape[1]) - 1
        d[spike_times[:,0]] = spike_times[:,1].type(torch.float)
        d = self.length - d
        d /= self.length
        return d.reshape(self.input_shape)




class PositionEncoder(TemporaryEncoder):
    """
    Poisson coding.
    """

    def __init__(
        self,
        name: str,
        shape: Iterable[int],
        time: float,
        k: int = None, # resolution
        mean: Union[float, torch.Tensor] = None,
        std: Union[float, torch.Tensor] = None,
        ignore_threshold: Union[float, torch.Tensor] = 1e-3,
        filt: Callable = lambda x: x,
        **kwargs
    ) -> None:
        super().__init__(
            name=name,
            input_shape=shape,
            output_shape=(*filt(torch.zeros(shape)).shape, k),
            time = time,
            filt=filt,
            **kwargs
        )
        self.k = k
        if mean is None:
            mean = torch.linspace(0, 1, self.k)
        self.register_buffer("mean", torch.tensor(mean))
        if std is None:
            std = (1/(self.k-1))/2
        self.register_buffer("std", torch.tensor(std))
        self.register_buffer("ignore_threshold", torch.tensor(ignore_threshold))


    def compute_spike(self, direct_input: torch.Tensor = torch.tensor(False)) -> None:
        self.s = (self.stage==(self.length-self.step))


    def encode_processed_data(self, data: torch.Tensor) -> None:
        normal = Normal(self.mean, self.std)
        self.stage = torch.exp(normal.log_prob(data.reshape(*data.shape,1)))
        self.stage[self.stage<self.ignore_threshold] = float('NaN')
        self.stage /= torch.exp(normal.log_prob(self.mean))
        self.stage *= self.length
        self.stage = self.stage.round()


    def reset(self) -> None:
        if hasattr(self, 'stage'): del self.stage
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
        name: str,
        shape: Iterable[int],
        rate: Union[int, torch.Tensor] = 1.,
        **kwargs
    ) -> None:
        super().__init__(
            name=name,
            input_shape=shape,
            output_shape=shape,
            **kwargs
        )
        self.register_buffer("rate", torch.tensor(rate))
        self.register_buffer("rate", torch.tensor(rate))


    def compute_spike(self, direct_input: torch.Tensor = torch.tensor(False)) -> None:
        self.s = torch.bernoulli(self.stage).type(torch.bool)


    def encode_processed_data(self, data: torch.Tensor) -> None:
        self.stage = data * self.rate


    def reset(self) -> None:
        if hasattr(self, 'stage'): del self.stage
        super().reset()


    def decode(self, data: torch.Tensor) -> torch.Tensor:
        return data.float().mean(axis=0).reshape(self.input_shape)




class RankOrderEncoder(Time2FirstSpikeEncoder):
    """
    Time-to-First-Spike coding.
    """

    def __init__(
        self,
        name: str,
        shape: Iterable[int],
        time: float,
        **kwargs
    ) -> None:
        super().__init__(
            name=name,
            shape=shape,
            time = time,
            **kwargs
        )


    def encode_processed_data(self, data: torch.Tensor) -> None:
        data = torch.unique(data, sorted=True, return_inverse=True)[1] + 1
        super().encode_processed_data(data)