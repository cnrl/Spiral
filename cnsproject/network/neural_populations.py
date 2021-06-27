"""
Module for neuronal dynamics and populations.
"""

from abc import abstractmethod
from typing import Union, Iterable, Callable
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
        self.free_axon_index = 0
        self.free_dendrites_index = 0
        self.set_dt(dt)


    def set_dt(self, dt:float):
        self.dt = torch.tensor(dt) if dt is not None else dt
        for axon_set in self.axons.values():
            axon_set.set_dt(dt)


    def add_axon(self, axon_set: AbstractAxonSet) -> None:
        axon_set.set_name(self.name+"_axon_"+str(self.free_axon_index), soft=True)
        axon_set.set_population_shape(self.shape)
        axon_set.set_dt(self.dt)
        self.axons[axon_set.name] = axon_set
        self.add_module(axon_set.name, axon_set)
        self.free_axon_index += 1


    def remove_axon(self, name: str) -> None:
        del self.axons[name]


    def add_dendrite(self, dendrite_set: AbstractDendriteSet) -> None:
        dendrite_set.set_name(self.name+"_dendrite_"+str(self.free_dendrites_index), soft=True)
        dendrite_set.set_population_shape(self.shape)
        dendrite_set.set_dt(self.dt)
        self.dendrites[dendrite_set.name] = dendrite_set
        self.add_module(dendrite_set.name, dendrite_set)
        self.free_dendrites_index += 1


    def remove_dendrite(self, name: str) -> None:
        del self.dendrites[name]

            
    def use(self, other: Union[Iterable, AbstractAxonSet, AbstractDendriteSet]) -> None:
        if hasattr(other, '__iter__'):
            for o in other:
                self.use(o)
        elif issubclass(type(other), AbstractAxonSet):
            self.add_axon(other)
        elif issubclass(type(other), AbstractDendriteSet):
            self.add_dendrite(other)
        else:
            assert False, f"You just can add AbstractAxonSet or AbstractDendriteSet to population. Your object is {type(other)}"


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


    def __str__(self):
        return f"{', '.join([a.__str__() for a in self.dendrites.values()])} [{self.name}] {', '.join([a.__str__() for a in self.axons.values()])}"




class AbstractPopulationProxy(AbstractNeuralPopulation):
    def __init__(
        self,
        population: AbstractNeuralPopulation,
        name: str = None,
        dt: float = None,
        **kwargs
    ) -> None:
        super().__init__(
            name=name if name is not None else f'Proxy[{population.name}]',
            shape=population.shape,
            **kwargs
        )
        self.population = population
        self.set_dt(dt if dt is not None else population.dt)


    def set_dt(self, dt:float):
        if 'population' in self.__dict__['_modules']:
            self.population.set_dt(dt)
        super().set_dt(dt)
            

    def backward(self) -> None:
        self.population.backward()
        super().backward()


    def forward(self,
            direct_input: torch.Tensor = torch.tensor(0.),
            clamps: torch.Tensor = torch.tensor(False),
            unclamps: torch.Tensor = torch.tensor(False)) -> None:
        self.population.forward(direct_input=self.collect_I(direct_input), clamps=clamps, unclamps=unclamps)
        self.compute_spike(self.population.spikes())
        spikes = self.spikes()
        for axon_set in self.axons.values():
            axon_set.forward(spikes)

    
    @abstractmethod
    def compute_spike(self, plain_spikes: torch.Tensor,
            clamps: torch.Tensor = torch.tensor(False),
            unclamps: torch.Tensor = torch.tensor(False)) -> None:
        super().compute_spike(clamps, unclamps)


    def reset(self) -> None:
        self.population.reset()
        super().reset()




class DisposablePopulationProxy(AbstractPopulationProxy):
    def __init__(
        self,
        population: AbstractNeuralPopulation,
        **kwargs
    ) -> None:
        super().__init__(
            population=population,
            **kwargs
        )
        self.register_buffer("consumed", torch.zeros(self.shape, dtype=torch.bool))


    def compute_spike(self,
            plain_spikes: torch.Tensor,
            clamps: torch.Tensor = torch.tensor(False),
            unclamps: torch.Tensor = torch.tensor(False)) -> None:
        self.s = plain_spikes * ~self.consumed
        super().compute_spike(plain_spikes, clamps, unclamps)
        self.consumed += self.s


    def reset(self) -> None:
        super().reset()
        self.consumed.zero_()




class KWinnerTakeAllPopulationProxy(AbstractPopulationProxy):
    def __init__(
        self,
        population: AbstractNeuralPopulation,
        lateral: Iterable[int],
        k: int = 1,
        critical_comparable: Callable = lambda x: x.u,
        **kwargs
    ) -> None:
        super().__init__(
            population=population,
            **kwargs
        )
        self.lateral = lateral
        self.register_buffer("ban_neurons", torch.zeros(self.shape, dtype=torch.bool))
        self.k = k
        self.k_need = k
        self.critical_comparable = critical_comparable
        self.features_dim = len(self.shape)-len(self.lateral)


    def compute_spike(self,
            plain_spikes: torch.Tensor,
            clamps: torch.Tensor = torch.tensor(False),
            unclamps: torch.Tensor = torch.tensor(False)) -> None:
        s = plain_spikes.clone()
        comparable = self.critical_comparable(self.population)
        self.s.zero_()
        while self.k_need>0:
            s *= ~self.ban_neurons
            if not torch.any(s):
                break
            valid_comparable = comparable.clone()
            valid_comparable[~s] = -float('inf')
            this_spike = torch.zeros_like(s).reshape(-1)
            this_spike[valid_comparable.reshape(-1).topk(1).indices] = 1
            this_spike = this_spike.reshape(s.shape)
            self.s += this_spike.bool()
            self.ban_neurons[[l[0] for l in torch.where(this_spike)[:self.features_dim]]] = True
            if 0 not in self.lateral:
                self.ban_neurons[[slice(None, None, None) for i in range(self.features_dim)]+\
                                [slice(max(l[0]-self.lateral[i]//2, 0), l[0]+self.lateral[i]//2+1, None) \
                                for i,l in enumerate(torch.where(this_spike)[self.features_dim:])]] = True
            self.k_need -= 1
        super().compute_spike(plain_spikes, clamps, unclamps)
        

    def reset(self) -> None:
        super().reset()
        self.k_need = self.k
        self.ban_neurons.zero_()




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
