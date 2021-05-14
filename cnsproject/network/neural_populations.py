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


    @abstractmethod
    def forward(self,
            direct_input: torch.Tensor = torch.tensor(0.),
            clamps: torch.Tensor = torch.tensor(False),
            unclamps: torch.Tensor = torch.tensor(False)) -> None:
        I = self.s * 0.
        I += direct_input
        for dendrite_set in self.dendrite_sets.values():
            I += dendrite_set.get_output()
        self.compute_potential(I)
        self.compute_spike()
        self.s *= ~unclamps
        self.s += clamps
        for axon_set in self.axon_sets.values():
            axon_set.forward(self.s)


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

    @abstractmethod
    def get_output(self) -> None:
        return self.s.clone()


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


# class InputPopulation(NeuralPopulation):
#     """
#     Neural population for user-defined spike pattern.

#     This class is implemented for future usage. Extend it if needed.

#     Arguments
#     ---------
#     shape : Iterable of int
#         Define the topology of neurons in the population.
#     spike_trace : bool, Optional
#         Specify whether to record spike traces. The default is True.
#     additive_spike_trace : bool, Optional
#         Specify whether to record spike traces additively. The default is True.
#     tau_s : float or torch.Tensor, Optional
#         Time constant of spike trace decay. The default is 15.0.
#     trace_scale : float or torch.Tensor, Optional
#         The scaling factor of spike traces. The default is 1.0.
#     learning : bool, Optional
#         Define the training mode. The default is True.

#     """

#     def __init__(
#         self,
#         **kwargs
#     ) -> None:
#         super().__init__(
#             **kwargs
#         )

#     def forward(self, s: torch.Tensor) -> None:
#         """
#         Simulate the neural population for a single step.

#         Parameters
#         ----------
#         traces : torch.Tensor
#             Input spike trace.

#         Returns
#         -------
#         None

#         """
#         self.s = s

#         super().forward(s)

#     def reset(self) -> None:
#         """
#         Reset all internal state variables.

#         Returns
#         -------
#         None

#         """
#         super().reset()


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
