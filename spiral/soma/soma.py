"""
The soma, or cell body, is where the signals from the dendrites are joined and passed on.\
This body will host the axons and dendrites.\
It also simulates the dynamics of neurons.\
The same module will be responsible for calculating the spike or current intensity sent to the end of the axon.
"""


from abc import ABC, abstractmethod
from construction_requirements_integrator import CRI, construction_required
from constant_properties_protector import CPP
from typing import Union, Iterable
from typeguard import typechecked
import torch
from spiral.axon import Axon
from spiral.dendrite import Dendrite




@typechecked
class Soma(torch.nn.Module, CRI, ABC):
    """
    Basic class for all types of soma.

    There are several types of soma in this package: spiking soma, current driven soma and etc.\
    Each of these types also has several subtype of soma: interneuron soma, sensory soma, neuromodulatory soma and etc.\
    Each of these types and subtypes has a different purpose and behaves differently.\
    But in the end, they are all types of neurons and have common features in their bodies,\
    including how they interact with axons and dendrites.\
    This abstract class implements these common behaviors.

    Properties
    ----------
    name : str, Protected
        The name to be uniquely accessible in Spiral network.\
        Read more about protected properties in constant-properties-protector package documentation.
    shape: Iterable of Int, Protected
        The topology of somas in the population.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: float or torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    axons: Dict[str, Axon]
        Dictionary containing connected axons.\
        The keys in this dictionary are the names of the corresponding axons.
    dendrites: Dict[str, Dendrite]
        Dictionary containing connected dendrites.\
        The keys in this dictionary are the names of the corresponding dendrites.
    is_constructed: bool
        Indicates the completion status of the construction.\
        Read more about construction completion in construction-requirements-integrator package documentation.

    Arguments
    ---------
    name : str, Necessary
        Each module in a Spiral network needs a name to be uniquely accessible.
    shape : Iterable of Int, Construction Requirement
        Defines the topology of somas in the population.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.
        Read more about construction requirement in construction-requirements-integrator package documentation.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.
        Read more about construction requirement in construction-requirements-integrator package documentation.
    construction_permission : bool, Optional, default: True
        You can prevent the completeion of the construction by setting this parameter to `False`.\
        After that, you can change it calling `set_construction_permission(True)`.\
        It is useful when you are inheriting a module from it.\
        Read more about `construction_permission` in construction-requirements-integrator package documentation.
    """
    def __init__(
        self,
        name: str,
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        torch.nn.Module.__init__(self)
        CPP.protect(self, 'name')
        CPP.protect(self, 'shape')
        CPP.protect(self, 'dt')
        self._name = name
        self.axons = {}
        self.dendrites = {}
        self.__unregistered_organs = []
        CRI.__init__(
            self,
            shape=shape,
            dt=dt,
            construction_permission=construction_permission,
            ignore_overwrite_error=True,
        )


    def __register_organ(
        self,
        organ: Union[Axon, Dendrite]
    ) -> None:
        """
        It coordinates the organ attached to this soma.\
        `shape` and `dt` are parameters that need to be coordinated between interconnected organs.\
        Read more about `meet_requirement` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        organ : Axon or Dendrite
            Attached organ that needs to be coordinated.
        
        Returns
        -------
        None
        
        """
        organs = [self.dendrites, self.axons][issubclass(type(organ), Axon)]

        suggested_name = f"{self.name}_{organ.__class__.__name__}_{len(organs)}"
        organ.meet_requirement(name=suggested_name)
        organ.meet_requirement(dt=self.dt)
        organ.meet_requirement(shape=self.shape)

        if organ.name in self.dendrites.keys():
            raise Exception(f"The soma is already using a dendrite named {organ.name}.")
        if organ.name in self.axons.keys():
            raise Exception(f"The soma is already using an axon named {organ.name}.")

        self.add_module(organ.name, organ)
        organs[organ.name] = organ


    def __construct__(
        self,
        shape: Iterable[int],
        dt: Union[float, torch.Tensor],
    ) -> None:
        """
        This function is related to the completion of the construction process.\
        Read more about `__construct__` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        shape : Iterable of Int
            Defines the topology of somas in the population.
        dt : float or torch.Tensor
            Time step in milliseconds.
        
        Returns
        -------
        None
        
        """
        self._shape = shape
        self.register_buffer("_dt", torch.tensor(dt))
        for organs in [self.axons, self.dendrites]:
            for name,organ in organs.items():
                self.__register_organ(organ)
        del self.__unregistered_organs
                
    
    def use(
        self,
        organ: Union[Axon, Dendrite]
    ) -> None:
        """
        Attaches an organ to the soma.
        
        Arguments
        ---------
        organ : Axon or Dendrite
            Attaching organ.
        
        Returns
        -------
        None
        
        """
        if self.is_constructed:
            self.__register_organ(organ)
        else:
            self.__unregistered_organs.append(organ)
            if organ.is_constructed:
                self.meet_requirement(shape=organ.shape)
                self.meet_requirement(dt=organ.dt)


    def _integrate_inputs(
        self,
        direct_input: torch.Tensor = torch.tensor(0.)
    ) -> torch.Tensor:
        """
        Calculates the sum of currents from dendrites or direct inputs.
        
        Arguments
        ---------
        direct_input : torch.Tensor
            Direct current input in milliamperes.
        
        Returns
        -------
        total_input_current : torch.Tensor
            The sum of currents from dendrites or direct inputs in milliamperes.
        
        """
        i = torch.zeros(self.shape)
        i += direct_input
        for dendrite_set in self.dendrites.values():
            i += dendrite_set.currents()
        return i


    @abstractmethod
    @construction_required
    def progress(
        self
    ) -> None:
        """
        Simulate the soma activity for a single step.
        
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
        for organs in [self.axons, self.dendrites]:
            for name,organ in organs.items():
                organ.reset()