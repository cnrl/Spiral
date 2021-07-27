"""
Dendrites, also dendrons, are branched protoplasmic extensions of a nerve cell that propagate\
the electrochemical stimulation received from other neural cells to the cell body, or soma,\
of the neuron from which the dendrites project. Electrical stimulation is transmitted onto dendrites\
by upstream neurons (usually via their axons) via synapses. The same module will be responsible for\
applying synaptic plasticity and modeling different forms of integrating inputs.
"""


import torch
from typing import Union, Iterable, Callable
from typeguard import typechecked
from abc import ABC, abstractmethod
from constant_properties_protector import CPP
from construction_requirements_integrator import CRI, construction_required
from spiral.analysis import Analyzer, analysis_point, analytics
from spiral.synaptic_plasticity.synaptic_plasticity import SynapticPlasticity




@typechecked
class Dendrite(torch.nn.Module, CRI, ABC):
    """
    Abstract class for different types of dendrites.

    The purpose of the dendrite is to receive electrochemical stimulations from different connected synapses,\
    integrate them and transfer it to the connected soma.\
    There are several types of dendrites: LinearDendrite, ConvolutionalDendrite, etc.
    But in the end, they are all types of dendrites and have common features in their bodies,\
    including how they interact with synapses, somas and synaptic plasticities.\
    This abstract class implements these common behaviors.

    Properties
    ----------
    name : str, Protected
        The name to be uniquely accessible in the connected soma or synapse.\
        Read more about protected properties in constant-properties-protector package documentation.
    shape : Iterable of int, Protected
        The topology of dendrites in the population.\
        Read more about protected properties in constant-properties-protector package documentation.
    batch : int, Protected
        Determines the batch size.\
        Read more about protected properties in constant-properties-protector package documentation.
    spine : Iterable of int, Protected
        The topology of each dendrite's spines.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    max : torch.Tensor
        Shows the maximum possible synaptic weight.
    min : torch.Tensor
        Shows the minimum possible synaptic weight.
    plasticity_model : SynapticPlasticity
        Determines how the synaptic plasticity is calculated.\
        Read more about synaptic plasticities in Spiral.SynapticPlasticity module documentation.
    plasticity : bool
        Determines enabality of synaptic plasticity.
    neurotransmitter : torch.Tensor, Protected
        Last input neurotransmitter.\
        Read more about protected properties in constant-properties-protector package documentation.
    neuromodulator : torch.Tensor, Protected
        Last input neuromodulator.\
        Read more about protected properties in constant-properties-protector package documentation.
    action_potential : torch.Tensor, Protected
        Last action potential of the connected soma.\
        Read more about protected properties in constant-properties-protector package documentation.
    is_constructed : bool
        Indicates the completion status of the construction.\
        Read more about construction completion in construction-requirements-integrator package documentation.
    synaptic_weights: torch.Tensor, Property, Read only
        Returns synaptic weights.
    analyzable : bool
        Indicates the analyzability status of the module.
    monitor : Monitor
        Exists just if the module be analyzable.\
        You can get a recorded sequence of important features through the `monitor`.\
        Read more about `Monitor` in monitor module documentation.

    Arguments
    ---------
    name : str, Construction Requirement
        The name to be uniquely accessible in the connected soma or synapse.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the source soma, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    shape : Iterable of int, Construction Requirement
        The topology of dendrites in the population. Should be same as shape of the connected soma.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the source soma, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    batch : int, Construction Requirement
        Determines the batch size. Should be same as network batch size.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the source soma, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    spine : Iterable of int, Construction Requirement
        The topology of each dendrite's spines. Should be same as shape of the connected synapse.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the connected synapse, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.
        It will be automatically set based on the source soma, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    maximum_weight : float or torch.Tensor, Optional, default: 1.
        Determines the maximum possible synaptic weight.
    minimum_weight : float or torch.Tensor, Optional, default: 0.
        Determines the minimum possible synaptic weight.
    plasticity_model : SynapticPlasticity, Optional, default: SynapticPlasticity()
        Determines how the synaptic plasticity is calculated.\
        Read more about synaptic plasticities in Spiral.SynapticPlasticity module documentation.
    plasticity : bool, Optional, default: True
        Determines enabality of synaptic plasticity.
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
        name: str = None,
        shape: Iterable[int] = None,
        batch: int = None,
        spine: Iterable[int] = None,
        dt: float = None,
        maximum_weight: Union[float, torch.Tensor] = 1.,
        minimum_weight: Union[float, torch.Tensor] = 0.,
        plasticity_model: SynapticPlasticity = None,
        plasticity: bool = True,
        analyzable: bool = False,
        construction_permission: bool = True,
    ) -> None:
        torch.nn.Module.__init__(self)
        CPP.protect(self, 'name')
        CPP.protect(self, 'shape')
        CPP.protect(self, 'spine')
        CPP.protect(self, 'dt')
        CPP.protect(self, 'batch')
        CPP.protect(self, 'neurotransmitter')
        CPP.protect(self, 'neuromodulator')
        CPP.protect(self, 'action_potential')
        self.register_buffer('max', torch.as_tensor(maximum_weight))
        self.register_buffer('min', torch.as_tensor(minimum_weight))
        self.plasticity_model = SynapticPlasticity() if plasticity_model is None else plasticity_model
        self.plasticity = plasticity
        Analyzer.__init__(self, analyzable)
        Analyzer.scout(self, state_calls={'transmit_current': self.transmit_current})
        CRI.__init__(
            self,
            name=name,
            shape=shape,
            batch=batch,
            spine=spine,
            dt=dt,
            construction_permission=construction_permission,
            ignore_overwrite_error=True,
        )


    @property
    @abstractmethod
    def synaptic_weights(
        self
    ) -> torch.Tensor:
        """
        Returns synaptic weights.

        Returns
        -------
        weights: torch.Tensor
            Synaptic weights.
        
        """
        pass


    def __construct__(
        self,
        name: str,
        shape: Iterable[int],
        batch: int,
        spine: Iterable[int],
        dt: Union[float, torch.Tensor],
    ) -> None:
        """
        This function is related to the completion of the construction process.\
        Read more about `__construct__` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        name : str
            The name to be uniquely accessible in the connected soma or synapse.
        shape : Iterable of int
            The topology of axon in the population.
        batch : int
            Determines the batch size.
        spine : Iterable of int
            The topology of each dendrite's spines.
        dt : float or torch.Tensor
            Time step in milliseconds.
        
        Returns
        -------
        None
        
        """
        self._name = name
        self._shape = (*shape,)
        self._spine = (*spine,)
        self._batch = batch
        self.register_buffer("_dt", torch.as_tensor(dt))
        self.plasticity_model.meet_requirement(source=spine)
        self.plasticity_model.meet_requirement(target=shape)
        self.plasticity_model.meet_requirement(batch=batch)
        self.plasticity_model.meet_requirement(dt=self.dt)
        self.plasticity_model.meet_requirement(maximum_weight=self.min)
        self.plasticity_model.meet_requirement(minimum_weight=self.max)
        self.register_buffer('_neurotransmitter', torch.zeros(self.batch, *self.spine))
        self.register_buffer('_neuromodulator', torch.zeros(self.batch, *self.spine))
        self.register_buffer('_action_potential', torch.zeros(self.batch, *self.shape))


    def _keep_weight_limits(
        self,
        w: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns synaptic weights to their allowable range.

        Returns
        -------
        allowable_weights: torch.Tensor
            Weights in allowable range.
        
        """
        w[w>self.max] = self.max
        w[w<self.min] = self.min
        return w


    @analysis_point
    def forward(
        self,
        neurotransmitter: torch.Tensor,
        neuromodulator: torch.Tensor,
    ) -> None:
        """
        Simulate the dendrite activity for a single step.

        Arguments
        ---------
        neurotransmitter : torch.Tensor
            Input neurotransmitters.
        neuromodulator : torch.Tensor
            Input neuromodulators.

        Returns
        -------
        None
        
        """
        self._neurotransmitter = neurotransmitter
        self._neuromodulator = neuromodulator


    def _update_synaptic_weights(
        self,
        synaptic_weights_plasticity: torch.Tensor,
    ) -> None:
        """
        Updates synaptic weights based on synaptic plasticity output.

        Arguments
        ---------
        synaptic_weights_plasticity : torch.Tensor
            The synaptic plasticity output.

        Returns
        -------
        None
        
        """
        pass


    @construction_required
    def backward(
        self,
        action_potential: torch.Tensor
    ) -> None:
        """
        Simulate the dendrite learning for a single step.

        Arguments
        ---------
        action_potential : torch.Tensor
            The action potential of the connected soma.

        Returns
        -------
        None
        
        """
        synaptic_weights_plasticity = self.plasticity_model(
            neurotransmitter=self.neurotransmitter,
            neuromodulator=self.neuromodulator,
            action_potential=self.action_potential,
            synaptic_weights=self.synaptic_weights,
        )
        if self.plasticity:
            self._update_synaptic_weights(synaptic_weights_plasticity)
        self._action_potential = action_potential


    @abstractmethod
    def transmit_current(
        self,
    ) -> torch.Tensor:
        """
        Calculates the output current.
        
        Returns
        -------
        current: torch.Tensor
            Output current.
        
        """
        pass


    def reset(
        self
    ) -> None:
        """
        Refractor and reset the axon and related moduls.
        
        Returns
        -------
        None
        
        """
        self._neurotransmitter.zero_()
        self._neuromodulator.zero_()
        self._action_potential.zero_()
        self.plasticity_model.reset()
        if self.analyzable:
            self.monitor.reset()


    @analytics
    def plot_transmiting_current(
        self,
        axes,
        **kwargs
    ) -> None:
        """
        Draw a plot of transmiting current on `axes`.

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
        y = self.monitor['transmit_current'].reshape(self.monitor['transmit_current'].shape[0],-1)
        time_range = (0, y.shape[0])
        x = torch.arange(*time_range)*self.dt
        population_alpha = 1/y.shape[1]
        aggregated = y.mean(axis=1)
        axes.plot(x, aggregated, color='blue', **kwargs)
        axes.plot(x, y, alpha=population_alpha, color='blue')
        axes.set_ylabel('transmiting current')
        axes.set_xlabel('time (ms)')
        axes.set_xlim(time_range)




@typechecked
class LinearDendrite(Dendrite):
    """
    Class for linear dendrites.

    A linear dendrite computes transmiting current to a soma as a linear combination of input values in spins.

    Properties
    ----------
    name : str, Protected
        The name to be uniquely accessible in the connected soma or synapse.\
        Read more about protected properties in constant-properties-protector package documentation.
    shape : Iterable of int, Protected
        The topology of dendrite in the population.\
        Read more about protected properties in constant-properties-protector package documentation.
    batch : int, Protected
        Determines the batch size.\
        Read more about protected properties in constant-properties-protector package documentation.
    spine : Iterable of int, Protected
        The topology of each dendrite's spines.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    max : torch.Tensor
        Shows the maximum possible synaptic weight.
    min : torch.Tensor
        Shows the minimum possible synaptic weight.
    plasticity_model : SynapticPlasticity
        Determines how the synaptic plasticity is calculated.\
        Read more about synaptic plasticities in Spiral.SynapticPlasticity module documentation.
    plasticity : bool
        Determines enabality of synaptic plasticity.
    neurotransmitter : torch.Tensor, Protected
        Last input neurotransmitter.\
        Read more about protected properties in constant-properties-protector package documentation.
    neuromodulator : torch.Tensor, Protected
        Last input neuromodulator.\
        Read more about protected properties in constant-properties-protector package documentation.
    action_potential : torch.Tensor, Protected
        Last action potential of the connected soma.\
        Read more about protected properties in constant-properties-protector package documentation.
    is_constructed : bool
        Indicates the completion status of the construction.\
        Read more about construction completion in construction-requirements-integrator package documentation.
    synaptic_weights: torch.Tensor, Property, Read only
        Returns synaptic weights.
    analyzable : bool
        Indicates the analyzability status of the module.
    monitor : Monitor
        Exists just if the module be analyzable.\
        You can get a recorded sequence of important features through the `monitor`.\
        Read more about `Monitor` in monitor module documentation.

    Arguments
    ---------
    name : str, Construction Requirement
        The name to be uniquely accessible in the connected soma or synapse.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the source soma, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    shape : Iterable of int, Construction Requirement
        The topology of dendrite in the population. Should be same as shape of the connected soma.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the source soma, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    batch : int, Construction Requirement
        Determines the batch size. Should be same as network batch size.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the source soma, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    spine : Iterable of int, Construction Requirement
        The topology of each dendrite's spines. Should be same as shape of the connected synapse.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the connected synapse, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    initial_weights : float or torch.Tensor or Callable, Optional, default: torch.rand
        Determines the initial values for synaptic weights.\
        If it be Callable, the required shape will be passed to it.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.
        It will be automatically set based on the source soma, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    maximum_weight : float or torch.Tensor, Optional, default: 1.
        Determines the maximum possible synaptic weight.
    minimum_weight : float or torch.Tensor, Optional, default: 0.
        Determines the minimum possible synaptic weight.
    plasticity_model : SynapticPlasticity, Optional, default: SynapticPlasticity()
        Determines how the synaptic plasticity is calculated.\
        Read more about synaptic plasticities in Spiral.SynapticPlasticity module documentation.
    plasticity : bool, Optional, default: True
        Determines enabality of synaptic plasticity.
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
        name: str = None,
        shape: Iterable[int] = None,
        batch: int = None,
        spine: Iterable[int] = None,
        initial_weights: Union[float, torch.Tensor, Callable] = torch.rand,
        dt: float = None,
        maximum_weight: Union[float, torch.Tensor] = 1.,
        minimum_weight: Union[float, torch.Tensor] = 0.,
        plasticity_model: SynapticPlasticity = None,
        plasticity: bool = True,
        analyzable: bool = False,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            shape=shape,
            batch=batch,
            spine=spine,
            dt=dt,
            maximum_weight=maximum_weight,
            minimum_weight=minimum_weight,
            plasticity_model=plasticity_model,
            plasticity=plasticity,
            analyzable=analyzable,
            construction_permission=False,
        )
        self.add_to_construction_requirements(initial_weights=initial_weights)
        CPP.protect(self, 'w')
        self.set_construction_permission(construction_permission)
        Analyzer.scout(self, state_variables=['w'])


    @property
    def synaptic_weights(
        self
    ) -> torch.Tensor:
        """
        Returns synaptic weights.

        Returns
        -------
        weights: torch.Tensor
            Synaptic weights.
        
        """
        return self.w


    def __construct__(
        self,
        name: str,
        shape: Iterable[int],
        batch: int,
        spine: Iterable[int],
        initial_weights: Union[float, torch.Tensor, Callable],
        dt: Union[float, torch.Tensor],
    ) -> None:
        """
        This function is related to the completion of the construction process.\
        Read more about `__construct__` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        name : str
            The name to be uniquely accessible in the connected soma or synapse.
        shape : Iterable of int
            The topology of axon in the population.
        batch : int
            Determines the batch size.
        spine : Iterable of int
            The topology of each dendrite's spines.
        dt : float or torch.Tensor
            Time step in milliseconds.
        
        Returns
        -------
        None
        
        """
        super().__construct__(
            name=name,
            shape=shape,
            batch=batch,
            spine=spine,
            dt=dt,
        )
        if callable(initial_weights):
            initial_weights = initial_weights((*self.spine, *self.shape))
        initial_weights = torch.as_tensor(initial_weights)
        if initial_weights.numel()==1:
            initial_weights = initial_weights.reshape(*self.spine, *self.shape)
        if initial_weights.shape!=(*self.spine, *self.shape):
            raise Exception(f"`initial_weights` must be in shape {*self.spine, *self.shape}")
        self.register_buffer("_w", initial_weights)
        self._w = self._keep_weight_limits(self.w)


    @construction_required
    def transmit_current(
        self,
    ) -> torch.Tensor:
        """
        Calculates the output current.
        
        Returns
        -------
        current: torch.Tensor
            Output current.
        
        """
        output = self.neurotransmitter.reshape(self.batch, *self.spine, *[1]*len(self.shape))
        output = output * self.synaptic_weights
        output = output.sum(axis=list(range(len(self.spine)+1)))
        return output


    def _update_synaptic_weights(
        self,
        synaptic_weights_plasticity: torch.Tensor,
    ) -> None:
        """
        Updates synaptic weights based on synaptic plasticity output.

        Arguments
        ---------
        synaptic_weights_plasticity : torch.Tensor
            The synaptic plasticity output.

        Returns
        -------
        None
        
        """
        self._w += synaptic_weights_plasticity
        self._w = self._keep_weight_limits(self.w)


    @analytics
    def plot_synaptic_weights(
        self,
        axes,
        **kwargs
    ) -> None:
        """
        Draw a plot of synaptic weights on `axes`.

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
        y = self.monitor['w'].reshape(self.monitor['w'].shape[0],-1)
        time_range = (0, y.shape[0])
        x = torch.arange(*time_range)*self.dt
        population_alpha = 1/y.shape[1]
        axes.plot(x, y, alpha=population_alpha)
        axes.set_ylabel('synaptic weights')
        axes.set_xlabel('time (ms)')
        axes.set_xlim(time_range)