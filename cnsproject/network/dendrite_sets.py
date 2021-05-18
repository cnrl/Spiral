"""
Module for connections between neural populations.
"""

from abc import ABC, abstractmethod
from typing import Union, Iterable
import torch
from .weight_initializations import constant_initialization

class AbstractDendriteSet(ABC, torch.nn.Module):
    def __init__(
        self,
        terminal: Iterable[int],
        population: Iterable[int],
        wmin: Union[float, torch.Tensor] = 0.,
        wmax: Union[float, torch.Tensor] = 1.,
        dt: float = None,
        **kwargs
    ) -> None:
        super().__init__()

        self.terminal_shape = terminal
        self.population_shape = population
        self.shape = (*self.terminal_shape,*self.population_shape)
        self.register_buffer("wmin", torch.tensor(wmin))
        self.register_buffer("wmax", torch.tensor(wmax))
        self.register_buffer("I", torch.zeros(*self.shape)) #mA
        self.set_dt(dt)


    def to_singlton_population_shape(self, tensor: torch.Tensor):
        if tensor.numel()==1 or tensor.shape==self.shape:
            return tensor
        else:
            return tensor.reshape((*self.terminal_shape,*[1]*len(self.population_shape)))


    def set_dt(self, dt:float):
        self.dt = torch.tensor(dt) if dt is not None else dt


    @abstractmethod
    def forward(self, neurotransmitters: torch.Tensor) -> None: #e: spike resonse  in shape (*self.terminal_shape,*self.population_shape)
        pass


    def backward(self, spikes: torch.Tensor) -> None: # population spike in shape(self.population_shape)
        self.s = spikes


    def reset(self) -> None:
        self.I.zero_()


    @abstractmethod
    def currents(self) -> torch.Tensor: # in shape *self.population_shape
        pass


    def spikes(self) -> torch.Tensor: # in shape (*self.population_shape)
        return self.s




class SimpleDendriteSet(AbstractDendriteSet):
    def __init__(
        self,
        w: torch.Tensor = None, # in shape (*self.terminal_shape, *self.population_shape) or *self.population_shape or 1
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if w is None:
            w = constant_initialization(
                self.terminal_shape,
                self.population_shape,
                self.wmin +
                (self.wmax - self.wmin) /
                (2 * torch.prod(torch.tensor(self.terminal_shape)))
            )
        self.register_buffer("w", w)
        self.w[self.w<self.wmin] = self.wmin
        self.w[self.w>self.wmax] = self.wmax


    def forward(self, neurotransmitters: torch.Tensor) -> None: #doesn't replace nan values
        neurotransmitters_singleton = self.to_singlton_population_shape(neurotransmitters)
        I = neurotransmitters_singleton * self.w
        I[neurotransmitters.isnan()] = 0
        self.I = self.I*neurotransmitters_singleton.isnan() + I


    def currents(self) -> torch.Tensor:
        return self.I.sum(axis=list(range(len(self.terminal_shape))))