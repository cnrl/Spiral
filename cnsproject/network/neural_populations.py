"""
Module for neuronal dynamics and populations.
"""

from functools import reduce
from abc import abstractmethod
from operator import mul
from typing import Union, Iterable

import torch

from .axon_sets import AbstractAxonSet
from .dendrite_sets import AbstractDendriteSet


class NeuralPopulation(torch.nn.Module):
    def __init__(
        self,
        shape: Iterable[int],
        # learning: bool = True,
        dt: float = None,
        **kwargs
    ) -> None:
        super().__init__()

        self.shape = shape
        self.n = reduce(mul, self.shape)
        self.register_buffer("s", torch.zeros(*self.shape, dtype=torch.bool))
        self.axon_sets = {}
        self.dendrite_sets = {}
        self.set_dt(dt)


    def set_dt(self, dt:float):
        self.dt = torch.tensor(dt) if dt is not None else dt
        for dendrite_set in self.dendrite_sets.values():
            dendrite_set.set_dt(dt)
        for axon_set in self.axon_sets.values():
            axon_set.set_dt(dt)

    def add_axon_set(self, axon_set: AbstractAxonSet, name: str):
        self.axon_sets[name] = axon_set
        axon_set.set_dt(self.dt)


    def add_dendrite_sets(self, dendrite_set: AbstractDendriteSet, name: str):
        self.dendrite_sets[name] = dendrite_set
        dendrite_set.set_dt(self.dt)

    
    def collect_I(self, direct_input: torch.Tensor = torch.tensor(0.)):
        I = self.s * 0.
        I += direct_input
        for dendrite_set in self.dendrite_sets.values():
            I += dendrite_set.get_output()
        return I


    @abstractmethod
    def forward(self,
            direct_input: torch.Tensor = torch.tensor(0.),
            clamps: torch.Tensor = torch.tensor(False),
            unclamps: torch.Tensor = torch.tensor(False)) -> None:
        self.compute_potential(self.collect_I(direct_input))
        self.compute_spike()
        self.s *= ~unclamps
        self.s += clamps
        for axon_set in self.axon_sets.values():
            axon_set.forward(self.s)
        for dendrite_set in self.dendrite_sets.values():
            dendrite_set.backward(self.s)


    @abstractmethod
    def compute_potential(self, I: torch.Tensor) -> None:
        pass


    @abstractmethod
    def compute_spike(self) -> None:
        pass


    @abstractmethod
    def reset(self) -> None:
        self.s.zero_()
        for axon_set in self.axon_sets.values():
            axon_set.reset()


    # def train(self, mode: bool = True) -> "NeuralPopulation":
    #     """
    #     Set the population's training mode.

    #     Parameters
    #     ----------
    #     mode : bool, optional
    #         Mode of training. `True` turns on the training while `False` turns\
    #         it off. The default is True.

    #     Returns
    #     -------
    #     NeuralPopulation

    #     """
    #     self.learning = mode
    #     return super().train(mode)




class LIFPopulation(NeuralPopulation):
    def __init__(
        self,
        shape: Iterable[int],
        tau: Union[float, torch.Tensor] = 20.,  #ms
        R: Union[float, torch.Tensor] = 1., #Ohm
        resting_potential: Union[float, torch.Tensor] = -70.6, #mV
        spike_threshold: Union[float, torch.Tensor] = -40., #mV
        **kwargs
    ) -> None:
        super().__init__(
            shape=shape,
            **kwargs
        )
        self.register_buffer("tau", torch.tensor(tau))
        self.register_buffer("R", torch.tensor(R))
        self.register_buffer("u_rest", torch.tensor(resting_potential))
        self.register_buffer("spike_threshold", torch.tensor(spike_threshold))
        self.register_buffer("u", torch.zeros(self.shape))
        self.u += self.u_rest
        self.register_buffer("s", torch.zeros(self.shape, dtype=torch.bool))


    def compute_potential(self, I: torch.Tensor) -> None:
        self.u -= self.dt/self.tau * ((self.u-self.u_rest) - self.R*I)


    def compute_spike(self) -> None:
        self.s = (self.u > self.spike_threshold)
        self.u *= ~self.s
        self.u += self.s*self.u_rest


    def reset(self) -> None:
        self.u.zero_()
        self.u += self.u_rest
        super().reset()




class ELIFPopulation(LIFPopulation):
    def __init__(
        self,
        shape: Iterable[int],
        sharpness: Union[float, torch.Tensor] = 2.,
        firing_threshold: Union[float, torch.Tensor] = -50.4, #mV
        spike_threshold: Union[float, torch.Tensor] = -40., #mV
        **kwargs
    ) -> None:
        super().__init__(
            shape=shape,
            spike_threshold=spike_threshold,
            **kwargs
        )
        self.register_buffer("sharpness", torch.tensor(sharpness))
        self.register_buffer("firing_threshold", torch.tensor(firing_threshold))


    def compute_potential(self, I: torch.Tensor) -> None:
        firing_term = self.sharpness*torch.exp((self.u-self.firing_threshold)/self.sharpness)
        super().compute_potential(I)
        self.u += self.dt/self.tau * firing_term




class AELIFPopulation(ELIFPopulation):
    def __init__(
        self,
        shape: Iterable[int],
        a_w: Union[float, torch.Tensor] = 4.,
        b_w: Union[float, torch.Tensor] = .0805,
        tau_w: Union[float, torch.Tensor] = 144.,
        **kwargs
    ) -> None:
        super().__init__(
            shape=shape,
            **kwargs
        )
        self.register_buffer("a_w", torch.tensor(a_w))
        self.register_buffer("b_w", torch.tensor(b_w))
        self.register_buffer("tau_w", torch.tensor(tau_w))
        self.register_buffer("w", torch.zeros(self.shape))


    def forward(self, **args) -> None: # I: mA
        super().forward(**args)
        self.compute_w()


    def compute_potential(self, I: torch.Tensor) -> None:
        super().compute_potential(I)
        self.u -= self.R*self.w


    def compute_w(self) -> None:
        self.w += self.dt/self.tau_w * (self.a_w*(self.u-self.u_rest) - self.w + self.b_w*self.tau_w*self.s)


    def reset(self) -> None:
        self.w.zero_()
        super().reset()
