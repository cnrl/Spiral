"""
Spiking soma is a type of soma that propagates spikes on its axons as output signals.
"""


from abc import ABC, abstractmethod
import torch
from constant_properties_protector import CPP
from spiral.analysis import Analyzer, analysis_point, analytics
from typing import Union, Iterable
from typeguard import typechecked
from ..soma import Soma
from construction_requirements_integrator import construction_required




@typechecked
class SpikingSoma(Soma, ABC):
    """
    Basic class for all types of spiking soma.

    There are several types of spiking soma: integrate and fire, hodgkin huxley and etc.\
    Each of these types has a different purpose and behaves differently.\
    But in the end, they are all types of spiking neurons and have common features in their bodies,\
    including how they propagate spikes on their axons and how they can be analyzed.\
    This abstract class implements these common behaviors.

    Properties
    ----------
    name : str, Protected
        The name to be uniquely accessible in Spiral network.\
        Read more about protected properties in constant-properties-protector package documentation.
    shape: Iterable of Int, Protected
        The topology of somas in the population.\
        Read more about protected properties in constant-properties-protector package documentation.
    spike: torch.Tensor[bool], Protected
        Indicates which neurons are firing.\
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
    analyzable: bool
        Indicates the analyzability status of the module.
    monitor: Monitor
        Exists just if the module be analyzable.\
        You can get a recorded sequence of important features through the `monitor`.\
        Read more about `Monitor` in monitor module documentation.

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
    analyzable: bool, Optional, default: False
        If it is `True`, it will record its behavior and provide you with functions for drawing plots.
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
        analyzable: bool = False,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            shape=shape,
            dt=dt,
            construction_permission=construction_permission,
        )
        CPP.protect(self, 'spike')
        Analyzer.__init__(self, analyzable)
        Analyzer.scout(self, state_variables=['spike'])


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
        super().__construct__(
            shape=shape,
            dt=dt,
        )
        self.register_buffer("_spike", torch.zeros(*self.shape, dtype=torch.bool))
                
    
    @abstractmethod
    def _process(
        self,
        inputs: torch.Tensor
    ) -> None:
        """
        Calculates the dynamics of neurons.

        Arguments
        ---------
        inputs : torch.Tensor
            Input current in milliamperes.

        Returns
        -------
        None
        
        """
        pass


    @abstractmethod
    def _fire_axon_hillock(
        self,
        clamps: torch.Tensor = torch.as_tensor(False),
        unclamps: torch.Tensor = torch.as_tensor(False),
    ) -> None:
        """
        Compute spikes.\
        This process simulates the process of firing that occurs in axon hillock part of the neuron body.        
        
        Arguments
        ---------
        clamps : torch.Tensor[bool], Optional, default: torch.tensor(False)
            Forcing neurons to fire.
        unclamps : torch.Tensor[bool], Optional, default: torch.tensor(False)
            Forcing neurons not to fire.

        Returns
        -------
        None
        
        """
        self._spike = ((self.spike * ~unclamps) + clamps)


    def __propagate_spike(
        self
    ) -> None:
        """
        Propagates spikes on its axons and dendrites (backward learning).
        
        Returns
        -------
        None
        
        """
        for axon in self.axons.values():
            axon.forward(action_potential=self.spike)
        for dendrite in self.dendrites.values():
            dendrite.backward(action_potential=self.spike)
        

    @construction_required
    @analysis_point
    def progress(
        self,
        direct_input: torch.Tensor = torch.as_tensor(0.),
        clamps: torch.Tensor = torch.as_tensor(False),
        unclamps: torch.Tensor = torch.as_tensor(False)
    ) -> None:
        """
        Simulate the soma activity for a single step.

        Arguments
        ---------
        inputs : torch.Tensor
            Input current in milliamperes.
        clamps : torch.Tensor[bool], Optional, default: torch.tensor(False)
            Forcing neurons to fire.
        unclamps : torch.Tensor[bool], Optional, default: torch.tensor(False)
            Forcing neurons not to fire.
        
        Returns
        -------
        None
        
        """
        self._process(inputs=self._integrate_inputs(direct_input=direct_input))
        self._fire_axon_hillock(clamps=clamps, unclamps=unclamps)
        self.__propagate_spike()
        super().progress()


    @construction_required
    def reset(
        self
    ) -> None:
        """
        Refractor and reset the somas and connected organs.
        
        Returns
        -------
        None
        
        """
        self._spike.zero_()
        super().reset()
        if self.analyzable:
            self.monitor.reset()


    @analytics
    def plot_spikes(
        self,
        axes,
        **kwargs
    ) -> None:
        """
        Draw a raster plot of spikes on `axes`.

        Arguments
        ---------
        axes : Matplotlib plotable module
            The axes to draw on.
        **kwargs : keyword arguments
            kwargs will be directly passed to matplotlib plot function.
        
        Returns
        -------
        None
        
        """
        spikes = self.monitor['spike']
        spikes = spikes.reshape(spikes.shape[0], -1)
        time_range = (0, spikes.shape[0])
        x,y = torch.where(spikes)
        x = x*self.dt
        axes.scatter(x, y, **kwargs)
        axes.set_ylabel('spike')
        axes.set_xlabel('time (ms)')
        axes.set_xlim(time_range)
        axes.get_yaxis().set_ticks([])


    @analytics
    def plot_population_activity(
        self,
        axes,
        **kwargs
    ) -> None:
        """
        Draw a plot of population activity on `axes`.

        Arguments
        ---------
        axes : Matplotlib plotable module
            The axes to draw on.
        **kwargs : keyword arguments
            kwargs will be directly passed to matplotlib plot function.
        
        Returns
        -------
        None
        
        """
        spikes = self.monitor['spike']
        spikes = spikes.reshape(spikes.shape[0], -1)
        time_range = (0, spikes.shape[0])
        x = torch.arange(*time_range)*self.dt
        y = spikes.sum(axis=1)
        axes.plot(x, y, **kwargs)
        axes.set_xlabel('time (ms)')
        axes.set_ylabel(f'activity (#spikes/{self.dt}ms')
        axes.set_xlim(time_range)