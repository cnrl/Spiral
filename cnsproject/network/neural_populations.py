"""
Module for neuronal dynamics and populations.
"""

from abc import abstractmethod
from typing import Union, Iterable
import torch
from .axon_sets import AbstractAxonSet
from .dendrite_sets import AbstractDendriteSet


class AbstractNeuralPopulation(torch.nn.Module):
    def __init__(
        self,
        name: str,
        shape: Iterable[int],
        dt: float = None,
        **kwargs
    ) -> None:
        super().__init__()

        self.name = name
        self.shape = shape
        self.register_buffer("s", torch.zeros(*self.shape, dtype=torch.bool))
        self.axons = {}
        self.dendrites = {}
        self.set_dt(dt)


    def set_dt(self, dt:float):
        self.dt = torch.tensor(dt) if dt is not None else dt
        for dendrite_set in self.dendrites.values():
            dendrite_set.set_dt(dt)
        for axon_set in self.axons.values():
            axon_set.set_dt(dt)

    def add_axon(self, axon_set: AbstractAxonSet):
        axon_set.set_name(self.name+"_axon_"+str(len(self.axons)), soft=True)
        axon_set.set_population_shape(self.shape)
        axon_set.set_dt(self.dt)
        self.axons[axon_set.name] = axon_set


    def add_dendrite(self, dendrite_set: AbstractDendriteSet):
        dendrite_set.set_name(self.name+"_dendrite_"+str(len(self.dendrites)), soft=True)
        dendrite_set.set_population_shape(self.shape)
        dendrite_set.set_dt(self.dt)
        self.dendrites[dendrite_set.name] = dendrite_set

    
    def collect_I(self, direct_input: torch.Tensor = torch.tensor(0.)):
        I = self.s * 0.
        I += direct_input
        for dendrite_set in self.dendrites.values():
            I += dendrite_set.currents()
        return I


    @abstractmethod
    def forward(self,
            direct_input: torch.Tensor = torch.tensor(0.),
            clamps: torch.Tensor = torch.tensor(False),
            unclamps: torch.Tensor = torch.tensor(False)) -> None:
        self.compute_potential(self.collect_I(direct_input))
        self.compute_spike(clamps=clamps, unclamps=unclamps)
        spikes = self.spikes()
        for axon_set in self.axons.values():
            axon_set.forward(spikes)


    @abstractmethod
    def backward(self) -> None:
        spikes = self.spikes()
        for dendrite_set in self.dendrites.values():
            dendrite_set.backward(spikes)


    @abstractmethod
    def compute_potential(self, I: torch.Tensor) -> None:
        pass


    @abstractmethod
    def compute_spike(self,
            clamps: torch.Tensor = torch.tensor(False),
            unclamps: torch.Tensor = torch.tensor(False)) -> None:
        self.s =  ((self.s * ~unclamps) + clamps)


    @abstractmethod
    def reset(self) -> None:
        self.s.zero_()
        for axon_set in self.axons.values():
            axon_set.reset()


    @abstractmethod
    def spikes(self) -> torch.Tensor:
        return self.s


    def __add__(self, other: Union[list, AbstractAxonSet, AbstractDendriteSet]):
        if type(other) is list:
            for o in other:
                self.__add__(o)
        elif issubclass(type(other), AbstractAxonSet):
            self.add_axon(other)
        elif issubclass(type(other), AbstractDendriteSet):
            self.add_dendrite(other)
        else:
            assert False, f"You just can add AbstractAxonSet or AbstractDendriteSet to population. Your object is {type(other)}"
        return self


    def __rshift__(self, other: AbstractAxonSet):
        self.add_axon(other)
        return other
    def __rlshift__(self, other: AbstractAxonSet):
        self.add_axon(other)
        return other


    def __lshift__(self, other: AbstractDendriteSet):
        self.add_dendrite(other)
        return other
    def __rrshift__(self, other: AbstractDendriteSet):
        self.add_dendrite(other)
        return other




class LIFPopulation(AbstractNeuralPopulation):
    def __init__(
        self,
        name: str,
        shape: Iterable[int],
        tau: Union[float, torch.Tensor] = 20.,  #ms
        R: Union[float, torch.Tensor] = 1., #Ohm
        resting_potential: Union[float, torch.Tensor] = -70.6, #mV
        spike_threshold: Union[float, torch.Tensor] = -40., #mV
        **kwargs
    ) -> None:
        super().__init__(
            name=name,
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
        self.u *= ~self.s
        self.u += self.s*self.u_rest
        self.u -= self.dt/self.tau * ((self.u-self.u_rest) - self.R*I)


    def compute_spike(self, **args) -> None:
        self.s = (self.u > self.spike_threshold)
        super().compute_spike(**args)


    def reset(self) -> None:
        self.u.zero_()
        self.u += self.u_rest
        super().reset()




class ELIFPopulation(LIFPopulation):
    def __init__(
        self,
        name: str,
        shape: Iterable[int],
        sharpness: Union[float, torch.Tensor] = 2.,
        firing_threshold: Union[float, torch.Tensor] = -50.4, #mV
        spike_threshold: Union[float, torch.Tensor] = -40., #mV
        **kwargs
    ) -> None:
        super().__init__(
            name=name,
            shape=shape,
            spike_threshold=spike_threshold,
            **kwargs
        )
        self.register_buffer("sharpness", torch.tensor(sharpness))
        self.register_buffer("firing_threshold", torch.tensor(firing_threshold))


    def compute_potential(self, I: torch.Tensor) -> None:
        super().compute_potential(I)
        firing_term = self.sharpness*torch.exp((self.u-self.firing_threshold)/self.sharpness)
        self.u += self.dt/self.tau * firing_term




class AELIFPopulation(ELIFPopulation):
    def __init__(
        self,
        name: str,
        shape: Iterable[int],
        a_w: Union[float, torch.Tensor] = 4.,
        b_w: Union[float, torch.Tensor] = .0805,
        tau_w: Union[float, torch.Tensor] = 144.,
        **kwargs
    ) -> None:
        super().__init__(
            name=name,
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
