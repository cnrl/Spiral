"""

"""

from abc import abstractmethod
from construction_requirements_integrator import CRI, construction_required
from typing import Union, Iterable
import torch
from ..axon import Axon
from ..dendrite import Dendrite


class Soma(torch.nn.Module, CRI):
    def __init__(
        self,
        name: str,
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None
    ) -> None:
        super().__init__()
        CRI.__init__(
            self,
            shape=shape,
            dt=dt,
            ignore_resetting_error=True,
        )
        self.name = name
        self.axons = {}
        self.dendrites = {}


    def register_organ(self, organ: Union[Axon, Dendrite]) -> None:
        organ.meet_requirement('dt', self.dt)
        organ.meet_requirement('population_shape', self.shape)
        self.add_module(obj.name, obj)


    def __construct__(self, shape: Iterable[int], dt: Union[float, torch.Tensor]) -> None:
        self.shape = shape
        self.register_buffer("spike", torch.zeros(*self.shape, dtype=torch.bool))
        self.dt = torch.tensor(dt)
        for organs in [self.axons, self.dendrites]:
            for name,organ in self.organs.items():
                self.register_organ(organ)
                
    
    def use(self, organ: Union[Axon, Dendrite]) -> None:
        if not issubclass(type(organ), Axon) and not issubclass(type(organ), Dendrite):
            raise Exception(f"You just can add Axon or Dendrite to Soma. Your object is {type(other)}")
        
        organs = [self.dendrites, self.axons][issubclass(type(organ), Axon)]
        organ.meet_requirement('name', f"{self.name}_{type(organ)}_{len(organs)}")

        if organ.name in self.dendrites.keys():
            raise Exception(f"The soma is already using a dendrite named {organ.name}.")
        if organ.name in self.axons.keys():
            raise Exception(f"The soma is already using an axon named {organ.name}.")

        organs[organ.name] = organ
        if self.is_constructed:
            self.register_organ(organ)
        elif organ.is_constructed:
            self.meet_requirement('shape', organ.population_shape)
            self.meet_requirement('dt', organ.dt)


    def integrate_inputs(self, direct_input: torch.Tensor = torch.tensor(0.)) -> torch.Tensor:
        i = torch.zeros(self.shape)
        i += direct_input
        for dendrite_set in self.dendrites.values():
            i += dendrite_set.currents()
        return i


    @abstractmethod
    def main_process(self, inputs) -> torch.Tensor:
        pass


    @abstractmethod
    def fire_axon_hillock(self,
            clamps: torch.Tensor = torch.tensor(False),
            unclamps: torch.Tensor = torch.tensor(False)) -> None:
        self.spike = ((self.spike * ~unclamps) + clamps)


    def propagate_spike(self) -> None:
        for axon in self.axons.values():
            axon.forward(self.spike)
        for dendrite in self.dendrites.values():
            dendrite.backward(self.spike)
        

    @construction_required
    def progress(self,
            direct_input: torch.Tensor = torch.tensor(0.),
            clamps: torch.Tensor = torch.tensor(False),
            unclamps: torch.Tensor = torch.tensor(False)) -> None:
        self.main_process(self.integrate_inputs(direct_input=direct_input))
        self.axon_hillock_fire(clamps=clamps, unclamps=unclamps)
        self.propagate_spike()


    def reset(self) -> None:
        self.spike.zero_()
        for organs in [self.axons, self.dendrites]:
            for name,organ in self.organs.items():
                organ.reset()


    def __str__(self):
        return f"{', '.join([a.__str__() for a in self.dendrites.values()])} -> [{self.name}] -> {', '.join([a.__str__() for a in self.axons.values()])}"