"""
Module for connections between neural populations.
"""


from __future__ import annotations
import torch
from typing import Union, Iterable
from typeguard import typechecked
from constant_properties_protector import CPP
from construction_requirements_integrator import CRI, construction_required
from add_on_class import AOC
from spiral.axon.axon import Axon
from spiral.dendrite.dendrite import Dendrite
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
        CPP.protect(self, 'batch')
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
            axon=None,
            dendrite=None,
            construction_permission=construction_permission,
            ignore_overwrite_error=True,
        )


    def __register_neuromodulatory_axon(
        self,
        organ: Axon,
    ) -> None:
        suggested_name = f"{self.name}_neuromodulatory_{organ.__class__.__name__}_{len(self.neuromodulatory_axons)}"
        for key,arg in {
            'name': suggested_name,
            'dt'  : self.dt,
            'batch': self.batch,
            'shape': (),
            'terminal': self.source,
            }.items():
            if not organ.is_constructed:
                organ.meet_requirement(**{key: arg})

        if organ.dt!=self.dt:
            raise Exception(f"Neuromodulatory axon {organ.name} with dt={organ.dt} doesn't match synapse {self.name} with dt={self.dt}.")
        if organ.name in self.neuromodulatory_axons.keys():
            raise Exception(f"The synapse is already using a neuromodulatory axon named {organ.name}.")
        if (*organ.shape, organ.terminal)!=self.source:
            raise Exception(f"Neuromodulatory axon {organ.name} with shape={organ.shape} and terminal={organ.terminal} doesn't match synapse {self.name} with source={self.source}.")
        if organ.batch!=self.batch:
            raise Exception(f"Neuromodulatory axon {organ.name} with batch={organ.batch} doesn't match synapse {self.name} with batch={self.batch}.")
        
        self.add_module(organ.name, organ)
        self.neuromodulatory_axons[organ.name] = organ


    def __share_info_with_organ(
        self,
        organ: Union[Axon, Dendrite],
    ) -> None:
        organ_is_axon = issubclass(type(organ), Axon)

        if organ_is_axon:
            name = self.name if self.is_constructed else self.requirement_value('name')
            if name is not None: name += '_axon'
            for key,arg in {
                'name': name,
                'dt'  : self.dt if self.is_constructed else self.requirement_value('dt'),
                'shape': self.source if self.is_constructed else self.requirement_value('source'),
                'terminal': (),
                'batch': self.batch if self.is_constructed else self.requirement_value('batch'),
                }.items():
                if (not organ.is_constructed) and (arg is not None):
                    organ.meet_requirement(**{key: arg})

        else:
            name = self.name if self.is_constructed else self.requirement_value('name')
            if name is not None: name += '_dendrite'
            for key,arg in {
                'name': name,
                'dt'  : self.dt if self.is_constructed else self.requirement_value('dt'),
                'shape': self.target if self.is_constructed else self.requirement_value('target'),
                'spine': self.source if self.is_constructed else self.requirement_value('source'),
                'batch': self.batch if self.is_constructed else self.requirement_value('batch'),
                }.items():
                if (not organ.is_constructed) and (arg is not None):
                    organ.meet_requirement(**{key: arg})


    def __fetch_info_from_organ(
        self,
        organ: Union[Axon, Dendrite],
    ) -> None:
        if self.is_constructed:
            return
        organ_is_axon = issubclass(type(organ), Axon)

        batch = organ.batch if organ.is_constructed else organ.requirement_value('batch')
        dt = organ.dt if organ.is_constructed else organ.requirement_value('dt')
        source = organ.spine                      if (not organ_is_axon and     organ.is_constructed) \
            else (*organ.shape, *organ.terminal)  if (    organ_is_axon and     organ.is_constructed) \
            else organ.requirement_value('spine') if (not organ_is_axon and not organ.is_constructed) \
            else (*organ.requirement_value('shape'), *organ.requirement_value('terminal'))
        target = organ.shape                      if (not organ_is_axon and     organ.is_constructed) \
            else organ.requirement_value('shape') if (not organ_is_axon and not organ.is_constructed) \
            else None

        name = None
        axon = self.requirement_value('axon')
        dendrite = self.requirement_value('dendrite')
        if (axon is not None) and (dendrite is not None):
            axon = axon.name if axon.is_constructed else axon.requirement_value('name')
            dendrite = dendrite.name if dendrite.is_constructed else dendrite.requirement_value('name')
            if axon is not None and dendrite is not None:
                name = f"{self.__class__.__name__}_from_{axon}_to_{dendrite}"

        for key,arg in {
            'name': name,
            'dt'  : dt,
            'source': source,
            'target': target,
            'batch': batch,
            }.items():
            if (not self.is_constructed) and (arg is not None):
                self.meet_requirement(**{key: arg})


    def __construct__(
        self,
        name: str,
        source: Iterable[int],
        target: Iterable[int],
        batch: int,
        dt: Union[float, torch.Tensor],
        axon: Axon,
        dendrite: Dendrite,
    ) -> None:
        self._name = name
        self._source = (*source,)
        self._target = (*target,)
        self._batch = batch
        self.register_buffer("_dt", torch.as_tensor(dt))
        self._axon = axon
        self._dendrite = dendrite

        self.__share_info_with_organ(self._axon)
        self.__share_info_with_organ(self._dendrite)

        if self.axon.dt!=self.dt:
            raise Exception(f"Axon {self.axon.name} with dt={self.axon.dt} doesn't match synapse {self.name} with dt={self.dt}.")
        if self._dendrite.dt!=self.dt:
            raise Exception(f"Dendrite {self.dendrite.name} with dt={self.dendrite.dt} doesn't match synapse {self.name} with dt={self.dt}.")

        if (*self.axon.shape, *self.axon.terminal)!=self.source:
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
            self.meet_requirement(axon=organ)
        else:
            self.meet_requirement(dendrite=organ)

        self.__fetch_info_from_organ(organ)
        self.__share_info_with_organ(organ)

        return self


    def _integrate_neuromodulators(
        self,
        direct_neuromodulators: torch.Tensor = torch.as_tensor(0.)
    ) -> torch.Tensor:
        neuromodulators = torch.zeros(*self.axon.shape, *self.axon.terminal)
        neuromodulators += direct_neuromodulators
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
        name: str,
        source: Iterable[int],
        target: Iterable[int],
        batch: int,
        dt: Union[float, torch.Tensor],
        axon: Axon,
        dendrite: Dendrite,
    ) -> None:
        self.__core.__construct__(
            self,
            name=name,
            source=source,
            target=target,
            batch=batch,
            dt=dt,
            axon=axon,
            dendrite=dendrite,
        )
        self.connectivity_pattern.meet_requirement(source=self.source)
        self.connectivity_pattern.meet_requirement(target=self.target)
        self.connectivity_pattern.meet_requirement(dt=self.dt)
        self.connectivity_pattern.meet_requirement(batch=self.batch)


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