"""
Module for connections between neural populations.
"""


from __future__ import annotations
from construction_requirements_integrator import CRI, construction_required
from add_on_class import AOC
from typing import Union, Iterable
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
        source: Iterable[int] = None,
        target: Iterable[int] = None,
        batch: int = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        torch.nn.Module.__init__(self)
        CPP.protect(self, 'name')
        CPP.protect(self, 'source')
        CPP.protect(self, 'target')
        CPP.protect(self, 'dt')
        CPP.protect(self, 'axon')
        CPP.protect(self, 'dendrite')
        self.neuromodulatory_axons = {}
        self.__unregistered_neuromodulatory_axons = []
        self._axon = None
        self._dendrite = None
        CRI.__init__(
            self,
            name=name,
            source=source,
            target=target,
            batch=batch,
            dt=dt,
            construction_permission=construction_permission,
        )


    @property
    def occupied(
        self
    ) -> bool:
        return (self.axon is not None) and (self.dendrite is not None)


    def __register_neuromodulatory_axon(
        self,
        organ: Axon,
    ) -> None:
        suggested_name = f"{self.name}_neuromodulatory_{organ.__class__.__name__}_{len(self.neuromodulatory_axons)}"
        for key,arg in {
            'name': suggested_name,
            'dt'  : self.dt,
            'shape': (1,),
            'batch': self.batch,
            }.items():
            if not organ.is_constructed:
                organ.meet_requirement(**{key: arg})

        if organ.dt!=self.dt:
            raise Exception(f"Neuromodulatory axon {organ.name} with dt={organ.dt} doesn't match synapse {self.name} with dt={self.dt}.")
        if organ.name in self.neuromodulatory_axons.keys():
            raise Exception(f"The synapse is already using a neuromodulatory axon named {organ.name}.")

        self.add_module(organ.name, organ)
        self.neuromodulatory_axons[organ.name] = organ


    def __construct__(
        self,
        name: str,
        source: Iterable[int],
        target: Iterable[int],
        batch: int,
        dt: Union[float, torch.Tensor],
    ):
        self._name = name
        self._source = (batch, *self.source)
        self._target = (batch, *self.target)
        self.register_buffer("_dt", torch.as_tensor(dt))

        self._axon.meet_requirement(name=name+"_axon")
        self._axon.meet_requirement(shape=source)
        self._axon.meet_requirement(batch=batch)
        self._axon.meet_requirement(dt=dt)

        self._dendrite.meet_requirement(name=name+"_dendrite")
        self._dendrite.meet_requirement(shape=target)
        self._dendrite.meet_requirement(batch=batch)
        self._dendrite.meet_requirement(spine=source)
        self._dendrite.meet_requirement(dt=dt)

        if self.axon.dt!=self.dt:
            raise Exception(f"Axon {self.axon.name} with dt={self.axon.dt} doesn't match synapse {self.name} with dt={self.dt}.")
        if self._dendrite.dt!=self.dt:
            raise Exception(f"Dendrite {self.dendrite.name} with dt={self.dendrite.dt} doesn't match synapse {self.name} with dt={self.dt}.")

        if (*self.axon.shape[1:], self.axon.terminal)!=self.source:
            raise Exception(f"Axon {self.axon.name} with shape={self.axon.shape} and terminal={self.axon.terminal} doesn't match synapse {self.name} with source={self.source}.")
        if self.dendrite.spine!=self.source:
            raise Exception(f"Dendrite {self.dendrite.name} with spine={self.dendrite.spine} doesn't match synapse {self.name} with source={self.source}.")

        if self.dendrite.shape!=self.target:
            raise Exception(f"Dendrite {self.dendrite.name} with shape={self.dendrite.shape} doesn't match synapse {self.name} with target={self.target}.")

        if self.axon.batch!=self.batch:
            raise Exception(f"Axon {self.axon.name} with batch={self.axon.batch} doesn't match synapse {self.name} with batch={self.batch}.")
        if self.dendrite.batch!=self.batch:
            raise Exception(f"Dendrite {self.dendrite.name} with batch={self.dendrite.batch} doesn't match synapse {self.name} with batch={self.batch}.")

        for neuromodulatory_axon in self.__unregistered_neuromodulatory_axons:
            self.__register_neuromodulatory_axon(neuromodulatory_axon)
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
        organ_is_axon = issubclass(type(organ), Axon)
        
        if organ_is_axon:
            self._axon = organ
        else:
            self._dendrite = organ

        batch = organ.shape[1] if organ.is_constructed else organ.requirement_value('batch')
        dt = organ.dt if organ.is_constructed else organ.requirement_value('dt')
        source = organ.spine[1:]                        if (not organ_is_axon and     organ.is_constructed) \
            else (*organ.shape[1:], *organ.terminal)    if (    organ_is_axon and     organ.is_constructed) \
            else organ.requirement_value('spine')       if (not organ_is_axon and not organ.is_constructed) \
            else (*organ.requirement_value('shape'), *organ.requirement_value('terminal'))
        target = organ.shape[1:]                        if (not organ_is_axon and     organ.is_constructed) \
            else organ.requirement_value('shape')       if (not organ_is_axon and not organ.is_constructed) \
            else None

        name = None
        if self.occupied:
            organ_names = {}
            for organ_type,registered_organ in {'axon': self._axon, 'dendrite': self._dendrite}.items():
                if registered_organ.is_constructed:
                    organ_names[organ_type] = registered_organ.name
                elif registered_organ.requirement_value('name') is not None:
                        organ_names[organ_type] = registered_organ.requirement_value('name')
            if all([name is not None for name in organ_names.values()]):
                name = f"{self.__class__.__name__}_from_{organ_names['axon']}_to_{organ_names['dendrite']}"


        for key,arg in {
            'name': name,
            'dt'  : dt,
            'source': source,
            'target': target,
            'batch': batch,
            }.items():
            if (not self.is_constructed) and (arg not None):
                self.meet_requirement(**{key: arg})

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
        self.connectivity_pattern.meet_requirement(source=self.source[1:])
        self.connectivity_pattern.meet_requirement(target=self.target[1:])
        self.connectivity_pattern.meet_requirement(dt=self.dt)
        self.connectivity_pattern.meet_requirement(batch=self.source[1])


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