"""
Module for connections between neural populations.
"""


from __future__ import annotations
from construction_requirements_integrator import CRI, construction_required
from add_on_class import AOC
from typing import Union
from typeguard import typechecked
import torch
from spiral.axon import Axon
from spiral.dendrite import Dendrite
from spiral.connectivity_pattern.connectivity_pattern import ConnectivityPattern




@typechecked
class Synapse(torch.nn.Module, CRI):
    def __init__(
        self,
        name: str = None,
        construction_permission: bool = False,
    ) -> None:
        torch.nn.Module.__init__(self)
        CPP.protect(self, 'name')
        self._name = name
        CPP.protect(self, 'dt')
        self.neuromodulatory_axons = {}
        self.__unregistered_neuromodulatory_axons = []
        CRI.__init__(
            self,
            axon=axon,
            dendrite=dendrite,
            construction_permission=construction_permission,
        )


    def __register_neuromodulatory_axon(
        self,
        axon: Axon,
    ) -> None:
        if not axon.is_constructed:
            raise Exception(f"Can not register an unconstructed axon: {axon} as neuromodulatory axon")
        if axon.dt!=self.dt:
            raise Exception(f"Neuromodulatory axon {axon.name} with dt={axon.dt} doesn't match synapse {self.name} with dt={self.dt}.")
        if axon.terminal_shape!=self.axon.terminal_shape:
            raise Exception(f"Neuromodulatory axon {axon.name} with terminal_shape={axon.terminal_shape} doesn't match synapse {self.name} with axon terminal_shape={self.axon.terminal_shape}.")
        if axon.name in self.neuromodulatory_axons.keys():
            raise Exception(f"The synapse is already using a neuromodulatory axon named {axon.name}.")

        self.add_module(axon.name, axon)
        self.neuromodulatory_axons[axon.name] = axon


    def __construct__(
        self,
        axon: Axon,
        dendrite: Dendrite,
    ):
        if not axon.is_constructed or not dendrite.is_constructed:
            raise Exception("Synapse can not connect unconstructed organs to each other.")
        
        if axon.dt!=dendrite.dt:
            raise Exception(f"Axon {axon.name} with dt={axon.dt} doesn't match dendrite {dendrite.name} with dt={dendrite.dt}.")
        self._dt = axon.dt
        
        if axon.terminal_shape!=dendrite.spine:
            raise Exception(f"Axon {axon.name} with terminal_shape={axon.terminal_shape} doesn't match dendrite {dendrite.name} with spine={dendrite.spine}.")

        if self.name is None:
            self._name = f"{self.__class__.__name__}_from_{axon.name}_to_{dendrite.name}"

        self.axon = axon
        self.dendrite = dendrite

        for neuromodulatory_axons in self.__unregistered_neuromodulatory_axons:
            self.__register_neuromodulatory_axon(neuromodulatory_axons)
        del self.__unregistered_neuromodulatory_axons


    def follow(
        self,
        axon: Axon,
    ) -> Synapse:
        if self.is_constructed:
            self.__register_neuromodulatory_axon(axon)
        else:
            self.__unregistered_neuromodulatory_axons.append(axon)
        return self

        
    def connect(
        self,
        organ: Union[Axon, Dendrite],
    ) -> Synapse:
        if issubclass(type(organ), Axon):
            self.meet_requirement(axon=organ)
        else:
            self.meet_requirement(dendrite=organ)
        return self


    def _integrate_neuromodulators(
        self,
        direct_neuromodulators: torch.Tensor = torch.as_tensor(0.)
    ) -> torch.Tensor:
        neuromodulators = torch.zeros(self.axon.terminal_shape)
        neuromodulators += direct_input
        for axon in self.neuromodulatory_axons.values():
            neuromodulators += axon.release()
        return neuromodulators


    @construction_required
    def forward(
        self,
        mask: torch.Tensor = torch.as_tensor(True),
        direct_neuromodulators: torch.Tensor = torch.as_tensor(0.)
    ) -> None:
        neurotransmitters = self.axon.release()
        neuromodulators = self._integrate_neuromodulators(direct_neuromodulators=direct_neuromodulators)
        self.dendrite.forward(neurotransmitters=neurotransmitters, neuromodulators=neuromodulators)


    def reset(
        self
    ) -> None:
        pass




@typechecked
class DisconnectorSynapticCover(AOC):
    def __pre_init__(
        self,
    ) -> None:
        if not issubclass(self.__core, Synapse):
            raise Exception("DisconnectorSynapticCover can only be added to Synapse or Synapse subclasses.")


    def __post_init__(
        self,
        connectivity_pattern: ConnectivityPattern,
    ) -> None:
        self.connectivity_pattern = connectivity_pattern


    def __construct__(
        self,
        axon: Axon,
        dendrite: Dendrite,
    ) -> None:
        self.__core.__construct__(
            self,
            axon=axon,
            dendrite=dendrite,
        )
        self.connectivity_pattern.meet_requirement(source=self.axon.terminal_shape)
        self.connectivity_pattern.meet_requirement(target=self.dendrite.shape)
        self.connectivity_pattern.meet_requirement(dt=self.dt)


    @construction_required
    def forward(
        self,
        mask: torch.Tensor = torch.as_tensor(True),
        direct_neuromodulators: torch.Tensor = torch.as_tensor(0.)
    ) -> None:
        self.__core.forward(
            self,
            mask=mask*self.connectivity_pattern(),
            direct_neuromodulators=direct_neuromodulators,
        )