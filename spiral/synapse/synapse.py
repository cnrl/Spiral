"""
Synapse, also called neuronal junction, the site of transmission of electric nerve\
impulses between two nerve cells (neurons).\
This modul will hold connections between axons and dendrites.\
It also transmits neuromodulators to the connected dendrite.
"""


from __future__ import annotations
import torch
from typing import Union, Iterable
from typeguard import typechecked
from abc import ABC, abstractmethod
from constant_properties_protector import CPP
from construction_requirements_integrator import CRI, construction_required
from add_on_class import AOC, covering_around
from spiral.axon.axon import Axon
from spiral.dendrite.dendrite import Dendrite
from spiral.connectivity_pattern.connectivity_pattern import ConnectivityPattern




@typechecked
class Synapse(torch.nn.Module, CRI, ABC):
    """
    Base class for synapses.

    The purpose of the synapse is to receive neurotransmitters from a connected axon,
    neuromodulators from connected neuromodulatory axons,\
    transfer them to dendrite by considering connectivities\

    Properties
    ----------
    name : str, Protected
        The name to be uniquely accessible in Spiral network.\
        Read more about protected properties in constant-properties-protector package documentation.
    source : Iterable of int, Protected
        The topology of connected axon terminals.\
        Read more about protected properties in constant-properties-protector package documentation.
    target : Iterable of int, Protected
        The topology of connected dendrite spines.\
        Read more about protected properties in constant-properties-protector package documentation.
    batch : int, Protected
        Determines the batch size.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    axon: Axon, Protected
        Connected axon.\
        Read more about protected properties in constant-properties-protector package documentation.
    dendrite: Dendrite, Protected
        Connected dendrite.\
        Read more about protected properties in constant-properties-protector package documentation.
    neuromodulatory_axons : Dict[str, Axon]
        Dictionary containing connected neuromodulatory axons.\
        The keys in this dictionary are the names of the corresponding axons.
    is_constructed : bool
        Indicates the completion status of the construction.\
        Read more about construction completion in construction-requirements-integrator package documentation.

    Arguments
    ---------
    name : str, Construction Requirement
        The name to be uniquely accessible in Spiral network.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the connected axon and soma, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    source : Iterable of int, Construction Requirement
        The topology of connected axon terminals. Should be same as (*shape, *terminal) of the connected axon.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the connected axon, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    target : Iterable of int, Construction Requirement
        The topology of connected dendrite spines. Should be same as spines shape of the connected dendrite.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the connected dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    batch : int, Construction Requirement
        Determines the batch size. Should be same as network batch size.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the connected axon or dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.
        It will be automatically set based on the connected axon or dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    construction_permission : bool, Optional, default: True
        You can prevent the completeion of the construction by setting this parameter to `False`.\
        After that, you can change it calling `set_construction_permission(True)`.\
        It is useful when you are inheriting a module from it.\
        Read more about `construction_permission` in construction-requirements-integrator package documentation.
    """
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
        """
        This function examines ecxeptions and performs side effects for\
        the final registration of an neuromodulatory axon.
        
        Arguments
        ---------
        organ : Axon
            Attached organ that needs to be coordinated.
        
        Returns
        -------
        None
        
        """
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
        """
        This function tries to send information, such as dt and batch,\
        if any, that fit together in an interconnected network, to an organ\
        connected to the synapse.
        
        Arguments
        ---------
        organ : Axon or Dendrite
            Attached organ that needs to be coordinated.
        
        Returns
        -------
        None
        
        """
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
        """
        This function tries to receive information, such as dt and batch,\
        if any, that fit together in an interconnected network, from an organ\
        connected to the synapse, and set them for the synapse.
        
        Arguments
        ---------
        organ : Axon or Dendrite
            Attached organ that needs to be coordinated.
        
        Returns
        -------
        None
        
        """
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
        """
        This function is related to the completion of the construction process.\
        Read more about `__construct__` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        name : str
            The name to be uniquely accessible in Spiral network.
        source : Iterable of int
            The topology of connected axon terminals.
        target : Iterable of int
            The topology of connected dendrite spines.
        batch : int
            Determines the batch size.
        dt : float or torch.Tensor
            Time step in milliseconds.
        axon : Axon
            The connected axon.
        dendrite : Dendrite
            The connected dendrite.
        
        Returns
        -------
        None
        
        """
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
        """
        Attaches a neuromodulatory axon to the synapse.
        
        Arguments
        ---------
        axom : Axon
            The attaching neuromodulatory axon.
        
        Returns
        -------
        self: Synapse
            With the aim of making chains possible.
        
        """
        if self.is_constructed:
            self.__register_neuromodulatory_axon(axon)
        else:
            self.__unregistered_neuromodulatory_axons.append(axon)
        return self

        
    def connect(
        self,
        organ: Union[Axon, Dendrite],
    ) -> Synapse:
        """
        Connects an organ to the synapse.
        
        Arguments
        ---------
        organ : Axon or Dendrite
            Connecting organ.
        
        Returns
        -------
        self: Soma
            With the aim of making chains possible.
        
        """
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
        direct_neuromodulator: torch.Tensor = torch.as_tensor(0.)
    ) -> torch.Tensor:
        """
        Calculates the sum of neuromodulators from neuromodulatory axons or direct inputs.
        
        Arguments
        ---------
        direct_input : torch.Tensor
            Direct neuromodulatory input.
        
        Returns
        -------
        total_neuromodulator_current : torch.Tensor
            The sum of neuromodulators from neuromodulatory axons or direct inputs.
        
        """
        neuromodulator = torch.zeros(self.batch, *self.axon.shape, *self.axon.terminal, device=self.dt.device)
        neuromodulator += direct_neuromodulator
        for axon in self.neuromodulatory_axons.values():
            neuromodulator += axon.release()
        return neuromodulator


    @abstractmethod
    def forward(
        self,
        mask: torch.Tensor = torch.as_tensor(True),
        direct_neuromodulator: torch.Tensor = torch.as_tensor(0.)
    ) -> None:
        """
        Simulate the synapse activity for a single step.
        
        Returns
        -------
        None
        
        """
        pass


    def reset(
        self
    ) -> None:
        """
        Refractor and reset the somas and connected organs.
        
        Returns
        -------
        None
        
        """
        pass




@typechecked
class FullyConnectedSynapse(Synapse):
    """
    A fully connected synapse will transmit the output of each axon terminal to each\
    spine of the dendrite.
    """

    @construction_required
    def forward(
        self,
        mask: torch.Tensor = torch.as_tensor(True),
        direct_neuromodulator: torch.Tensor = torch.as_tensor(0.)
    ) -> None:
        """
        Simulate the synapse activity for a single step.
        
        Returns
        -------
        None
        
        """
        neurotransmitter = self.axon.release()\
                            .reshape(self.batch, *self.source, 1, *[1]*len(self.target))
        neuromodulator = self._integrate_neuromodulators(direct_neuromodulator=direct_neuromodulator)\
                            .reshape(self.batch, *self.source, 1, *[1]*len(self.target))
        mask = mask * torch.ones(self.batch).diag().reshape(self.batch, *[1]*len(self.source), self.batch, *[1]*len(self.target))
        self.dendrite.forward(neurotransmitter=neurotransmitter*mask, neuromodulator=neuromodulator*mask)




@typechecked
@covering_around([FullyConnectedSynapse])
class DisconnectorSynapticCover(AOC):
    """
    Add-on class to add disconnectivity to synaptic connections.

    This module adds disconnectivity, means that some dendrite spines will always receive\
    empty inputs from some axon terminals.\
    Read more about add-on classes in add-on-class package documentation.\
    Because this is an add-on class, we will only introduce the properties and functions it adds.

    Add-On Properties
    -----------------
    connectivity_pattern : ConnectivityPattern
        Specifies which connections exist and which do not.\
        This connectivity can change over time.\
        Read more about ConnectivityPattern in Spiral.ConnectivityPattern module documentation.

    Add-On Arguments
    ----------------
    connectivity_pattern : ConnectivityPattern, Necessary
        Specifies which connections exist and which do not.
        Read more about ConnectivityPattern in Spiral.ConnectivityPattern module documentation.
    """
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
        """
        This function is related to the completion of the construction process.\
        Read more about `__construct__` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        name : str
            The name to be uniquely accessible in Spiral network.
        source : Iterable of int
            The topology of connected axon terminals.
        target : Iterable of int
            The topology of connected dendrite spines.
        batch : int
            Determines the batch size.
        dt : float or torch.Tensor
            Time step in milliseconds.
        axon : Axon
            The connected axon.
        dendrite : Dendrite
            The connected dendrite.
        
        Returns
        -------
        None
        
        """
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
        direct_neuromodulator: torch.Tensor = torch.as_tensor(0.)
    ) -> None:
        """
        Simulate the synapse activity for a single step.
        
        Returns
        -------
        None
        
        """
        self.__core.forward(
            self,
            mask=mask*self.connectivity_pattern(),
            direct_neuromodulator=direct_neuromodulator,
        )