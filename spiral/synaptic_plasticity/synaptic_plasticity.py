"""
In neuroscience, synaptic plasticity is the ability of synapses to strengthen\
or weaken over time, in response to increases or decreases in their activity.\
This module will determine and tell the dendrite proper synaptic weight changes.
"""


import torch
from typing import Union, Iterable
from typeguard import typechecked
from abc import ABC, abstractmethod
from constant_properties_protector import CPP
from construction_requirements_integrator import CRI, construction_required
from spiral.response_function.response_function import ResponseFunction
from spiral.learning_rate.synaptic_plasticity_rate import SynapticPlasticityRate




@typechecked
class SynapticPlasticity(torch.nn.Module, CRI, ABC):
    """
    Basic class for synaptic plasticities.\
    Since this is a base class, it receives different inputs for construction that it may not use all of them.\
    But its children will need them.

    Properties
    ----------
    source : Iterable of int, Protected
        The topology of each dendrite's spines.\
        Read more about protected properties in constant-properties-protector package documentation.
    target : Iterable of int, Protected
        The topology of dendrites in the population.\
        Read more about protected properties in constant-properties-protector package documentation.
    batch : int, Protected
        Determines the batch size.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    max : torch.Tensor
        Shows the maximum possible synaptic weight.
    min : torch.Tensor
        Shows the minimum possible synaptic weight.

    Arguments
    ---------
    source : Iterable of int, Construction Requirement
        The topology of each dendrite's spines. Should be same as spines shape of the owner dendrite.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    target : Iterable of int, Construction Requirement
        The topology of dendrites in the population. Should be same as shape of the owner dendrite.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    batch : int, Construction Requirement
        Determines the batch size. Should be same as network batch size.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    maximum_weight : float or torch.Tensor, Construction Requirement
        Determines the maximum possible synaptic weight.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    minimum_weight : float or torch.Tensor, Construction Requirement
        Determines the minimum possible synaptic weight.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    construction_permission : bool, Optional, default: True
        You can prevent the completeion of the construction by setting this parameter to `False`.\
        After that, you can change it calling `set_construction_permission(True)`.\
        It is useful when you are inheriting a module from it.\
        Read more about `construction_permission` in construction-requirements-integrator package documentation.
    """
    def __init__(
        self,
        source: Iterable[int] = None,
        target: Iterable[int] = None,
        batch: int = None,
        maximum_weight: Union[float, torch.Tensor] = None,
        minimum_weight: Union[float, torch.Tensor] = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        torch.nn.Module.__init__(self)
        CPP.protect(self, 'source')
        CPP.protect(self, 'target')
        CPP.protect(self, 'batch')
        CPP.protect(self, 'dt')
        CRI.__init__(
            self,
            source=source,
            target=target,
            batch=batch,
            maximum_weight=maximum_weight,
            minimum_weight=minimum_weight,
            dt=dt,
            ignore_overwrite_error=True,
            construction_permission=construction_permission,
        )


    def __construct__(
        self,
        source: Iterable[int],
        target: Iterable[int],
        batch: int,
        maximum_weight: Union[float, torch.Tensor],
        minimum_weight: Union[float, torch.Tensor],
        dt: Union[float, torch.Tensor],
    ) -> None:
        """
        This function is related to the completion of the construction process.\
        Read more about `__construct__` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        source : Iterable of int
            The topology of each dendrite's spines.
        target : Iterable of int
            The topology of dendrites in the population.
        batch : int
            Determines the batch size.
        dt : float or torch.Tensor
            Time step in milliseconds.
        maximum_weight : float or torch.Tensor
            Determines the maximum possible synaptic weight.
        minimum_weight : float or torch.Tensor
            Determines the minimum possible synaptic weight.
        
        Returns
        -------
        None
        
        """
        self._source = source
        self._target = target
        self._batch = batch
        self.register_buffer('max', torch.as_tensor(maximum_weight))
        self.register_buffer('min', torch.as_tensor(minimum_weight))
        self.register_buffer("_dt", torch.as_tensor(dt))


    @abstractmethod
    def __call__(
        self,
        neurotransmitter: torch.Tensor,
        neuromodulator: torch.Tensor,
        action_potential: torch.Tensor,
        synaptic_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate the module activity for a single step and returns the proper synaptic weight changes.
        
        Arguments
        ---------
        neurotransmitter : torch.Tensor
            Last axon terminal released neurotransmitters.
        neuromodulator : torch.Tensor
            Last synapse neuromodulators.
        action_potential : torch.Tensor
            Last dendrite action potential.
        synaptic_weights : torch.Tensor
            Last synaptic weights.

        Returns
        -------
        dw: torch.Tensor
            The proper synaptic weight changes.
        
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
        pass




@typechecked
class CompositeSynapticPlasticity(SynapticPlasticity):
    """
    It will make a composition of the input synaptic plasticities.\
    Order of the input synaptic plasticities is not important.\
    It always returns the sum of compositing outputs.

    Properties
    ----------
    synaptic_plasticities: Iterable[SynapticPlasticity]
        The compositing synaptic plasticities.
    source : Iterable of int, Protected
        The topology of each dendrite's spines.\
        Read more about protected properties in constant-properties-protector package documentation.
    target : Iterable of int, Protected
        The topology of dendrites in the population.\
        Read more about protected properties in constant-properties-protector package documentation.
    batch : int, Protected
        Determines the batch size.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    max : torch.Tensor
        Shows the maximum possible synaptic weight.
    min : torch.Tensor
        Shows the minimum possible synaptic weight.

    Arguments
    ---------
    synaptic_plasticities: Iterable[SynapticPlasticity], Necessary
        The compositing synaptic plasticities.
    source : Iterable of int, Construction Requirement
        The topology of each dendrite's spines. Should be same as spines shape of the owner dendrite.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    target : Iterable of int, Construction Requirement
        The topology of dendrites in the population. Should be same as shape of the owner dendrite.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    batch : int, Construction Requirement
        Determines the batch size. Should be same as network batch size.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    maximum_weight : float or torch.Tensor, Construction Requirement
        Determines the maximum possible synaptic weight.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    minimum_weight : float or torch.Tensor, Construction Requirement
        Determines the minimum possible synaptic weight.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    construction_permission : bool, Optional, default: True
        You can prevent the completeion of the construction by setting this parameter to `False`.\
        After that, you can change it calling `set_construction_permission(True)`.\
        It is useful when you are inheriting a module from it.\
        Read more about `construction_permission` in construction-requirements-integrator package documentation.
    """
    def __init__(
        self,
        synaptic_plasticities: Iterable[SynapticPlasticity],
        source: Iterable[int] = None,
        target: Iterable[int] = None,
        maximum_weight: Union[float, torch.Tensor] = None,
        minimum_weight: Union[float, torch.Tensor] = None,
        batch: int = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            source=source,
            target=target,
            batch=batch,
            maximum_weight=maximum_weight,
            minimum_weight=minimum_weight,
            dt=dt,
            construction_permission=False,
        )
        self.synaptic_plasticities = synaptic_plasticities
        for i,synaptic_plasticity in enumerate(synaptic_plasticities):
            self.add_module(str(i), synaptic_plasticity)
        self.set_construction_permission(construction_permission)


    def __construct__(
        self,
        source: Iterable[int],
        target: Iterable[int],
        batch: int,
        maximum_weight: Union[float, torch.Tensor],
        minimum_weight: Union[float, torch.Tensor],
        dt: Union[float, torch.Tensor],
    ) -> None:
        """
        This function is related to the completion of the construction process.\
        It will help the given response functions to be constructed too.\
        Read more about `__construct__` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        source : Iterable of int
            The topology of each dendrite's spines.
        target : Iterable of int
            The topology of dendrites in the population.
        batch : int
            Determines the batch size.
        dt : float or torch.Tensor
            Time step in milliseconds.
        maximum_weight : float or torch.Tensor
            Determines the maximum possible synaptic weight.
        minimum_weight : float or torch.Tensor
            Determines the minimum possible synaptic weight.
        
        Returns
        -------
        None
        
        """
        super().__construct__(
            source=source,
            target=target,
            batch=batch,
            maximum_weight=maximum_weight,
            minimum_weight=minimum_weight,
            dt=dt,
        )
        for synaptic_plasticity in self.synaptic_plasticities:
            synaptic_plasticity.meet_requirement(source=source)
            synaptic_plasticity.meet_requirement(target=target)
            synaptic_plasticity.meet_requirement(batch=batch)
            synaptic_plasticity.meet_requirement(maximum_weight=maximum_weight)
            synaptic_plasticity.meet_requirement(minimum_weight=minimum_weight)
            synaptic_plasticity.meet_requirement(dt=dt)


    @construction_required
    def __call__(
        self,
        neurotransmitter: torch.Tensor,
        neuromodulator: torch.Tensor,
        action_potential: torch.Tensor,
        synaptic_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate the module activity for a single step and returns the proper synaptic weight changes.
        
        Arguments
        ---------
        neurotransmitter : torch.Tensor
            Last axon terminal released neurotransmitters.
        neuromodulator : torch.Tensor
            Last synapse neuromodulators.
        action_potential : torch.Tensor
            Last dendrite action potential.
        synaptic_weights : torch.Tensor
            Last synaptic weights.

        Returns
        -------
        dw: torch.Tensor
            The proper synaptic weight changes.
        
        """
        dw = 0
        for synaptic_plasticity in self.synaptic_plasticities:
            dw = dw + synaptic_plasticity(
                neurotransmitter=neurotransmitter,
                neuromodulator=neuromodulator,
                action_potential=action_potential,
                synaptic_weights=synaptic_weights,
            )
        return dw


    def reset(
        self
    ) -> None:
        """
        Refractor and reset the axon and related moduls.
        
        Returns
        -------
        None
        
        """
        for synaptic_plasticity in self.synaptic_plasticities:
            synaptic_plasticity.reset()




@typechecked
class ConvergentSynapticPlasticity(SynapticPlasticity):
    """
    This module provides a synaptic plasticity that tries to converge synaptic weights to a certain point.\
    By setting the convergence point to zero, you reach a decaying synaptic plasticity.

    Properties
    ----------
    tau: torch.Tensor
        Convergence time constant in milliseconds.
    convergence: torch.Tensor
        Convergence point.
    source : Iterable of int, Protected
        The topology of each dendrite's spines.\
        Read more about protected properties in constant-properties-protector package documentation.
    target : Iterable of int, Protected
        The topology of dendrites in the population.\
        Read more about protected properties in constant-properties-protector package documentation.
    batch : int, Protected
        Determines the batch size.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    max : torch.Tensor
        Shows the maximum possible synaptic weight.
    min : torch.Tensor
        Shows the minimum possible synaptic weight.

    Arguments
    ---------
    tau: float or torch.Tensor, Optional, default: 250.
        Convergence time constant in milliseconds.
    convergence: float or torch.Tensor, Optional, default: 0.
        Convergence point.
    source : Iterable of int, Construction Requirement
        The topology of each dendrite's spines. Should be same as spines shape of the owner dendrite.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    target : Iterable of int, Construction Requirement
        The topology of dendrites in the population. Should be same as shape of the owner dendrite.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    batch : int, Construction Requirement
        Determines the batch size. Should be same as network batch size.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    maximum_weight : float or torch.Tensor, Construction Requirement
        Determines the maximum possible synaptic weight.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    minimum_weight : float or torch.Tensor, Construction Requirement
        Determines the minimum possible synaptic weight.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    construction_permission : bool, Optional, default: True
        You can prevent the completeion of the construction by setting this parameter to `False`.\
        After that, you can change it calling `set_construction_permission(True)`.\
        It is useful when you are inheriting a module from it.\
        Read more about `construction_permission` in construction-requirements-integrator package documentation.
    """
    def __init__(
        self,
        tau: Union[float, torch.Tensor] = 250.,
        convergence: Union[float, torch.Tensor] = 0.,
        source: Iterable[int] = None,
        target: Iterable[int] = None,
        batch: int = None,
        maximum_weight: Union[float, torch.Tensor] = None,
        minimum_weight: Union[float, torch.Tensor] = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            source=source,
            target=target,
            batch=batch,
            maximum_weight=maximum_weight,
            minimum_weight=minimum_weight,
            dt=dt,
            construction_permission=construction_permission,
        )
        self.register_buffer("tau", torch.as_tensor(tau))
        self.register_buffer("convergence", torch.as_tensor(convergence))


    @construction_required
    def __call__(
        self,
        neurotransmitter: torch.Tensor,
        neuromodulator: torch.Tensor,
        action_potential: torch.Tensor,
        synaptic_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate the module activity for a single step and returns the proper synaptic weight changes.
        
        Arguments
        ---------
        neurotransmitter : torch.Tensor
            Last axon terminal released neurotransmitters.
        neuromodulator : torch.Tensor
            Last synapse neuromodulators.
        action_potential : torch.Tensor
            Last dendrite action potential.
        synaptic_weights : torch.Tensor
            Last synaptic weights.

        Returns
        -------
        dw: torch.Tensor
            The proper synaptic weight changes.
        
        """
        return - (synaptic_weights - self.convergence) * self.dt / self.tau




@typechecked
class STDP(SynapticPlasticity):
    """
    Models Spike-Time Dependent Plasticity learning rule.

    Properties
    ----------
    presynaptic_tagging : ResponseFunction
        Determines presynaptic tagging model.\
        Read more about tagging and ResponseFunction in Spiral.ResponseFunction module documentation.
    postsynaptic_tagging : ResponseFunction
        Determines postsynaptic tagging model.\
        Read more about tagging and ResponseFunction in Spiral.ResponseFunction module documentation.
    ltp_rate : SynapticPlasticityRate
        Determines LTP term learning rate.\
        Read more about learning rate and SynapticPlasticityRate in Spiral.SynapticPlasticityRate module documentation.
    ltd_rate : SynapticPlasticityRate
        Determines LTD term learning rate.\
        Read more about learning rate and SynapticPlasticityRate in Spiral.SynapticPlasticityRate module documentation.
    source : Iterable of int, Protected
        The topology of each dendrite's spines.\
        Read more about protected properties in constant-properties-protector package documentation.
    target : Iterable of int, Protected
        The topology of dendrites in the population.\
        Read more about protected properties in constant-properties-protector package documentation.
    batch : int, Protected
        Determines the batch size.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    max : torch.Tensor
        Shows the maximum possible synaptic weight.
    min : torch.Tensor
        Shows the minimum possible synaptic weight.

    Arguments
    ---------
    presynaptic_tagging : ResponseFunction, Necessary
        Determines presynaptic tagging model.\
        Read more about tagging and ResponseFunction in Spiral.ResponseFunction module documentation.
    postsynaptic_tagging : ResponseFunction, Necessary
        Determines postsynaptic tagging model.\
        Read more about tagging and ResponseFunction in Spiral.ResponseFunction module documentation.
    ltp_rate : SynapticPlasticityRate, Optional, default: SynapticPlasticityRate(rate=0.01)
        Determines LTP term learning rate.\
        Read more about learning rate and SynapticPlasticityRate in Spiral.SynapticPlasticityRate module documentation.
    ltd_rate : SynapticPlasticityRate, Optional, default: SynapticPlasticityRate(rate=0.01)
        Determines LTD term learning rate.\
        Read more about learning rate and SynapticPlasticityRate in Spiral.SynapticPlasticityRate module documentation.
    source : Iterable of int, Construction Requirement
        The topology of each dendrite's spines. Should be same as spines shape of the owner dendrite.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    target : Iterable of int, Construction Requirement
        The topology of dendrites in the population. Should be same as shape of the owner dendrite.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    batch : int, Construction Requirement
        Determines the batch size. Should be same as network batch size.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    maximum_weight : float or torch.Tensor, Construction Requirement
        Determines the maximum possible synaptic weight.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    minimum_weight : float or torch.Tensor, Construction Requirement
        Determines the minimum possible synaptic weight.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the owner dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    construction_permission : bool, Optional, default: True
        You can prevent the completeion of the construction by setting this parameter to `False`.\
        After that, you can change it calling `set_construction_permission(True)`.\
        It is useful when you are inheriting a module from it.\
        Read more about `construction_permission` in construction-requirements-integrator package documentation.
    """
    def __init__(
        self,
        presynaptic_tagging: ResponseFunction,
        postsynaptic_tagging: ResponseFunction,
        ltp_rate: SynapticPlasticityRate = None,
        ltd_rate: SynapticPlasticityRate = None,
        source: Iterable[int] = None,
        target: Iterable[int] = None,
        batch: int = None,
        maximum_weight: Union[float, torch.Tensor] = None,
        minimum_weight: Union[float, torch.Tensor] = None,
        dt: Union[float, torch.Tensor] = None,
        related_to_fair_synapse: bool = False,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            source=source,
            target=target,
            batch=batch,
            maximum_weight=maximum_weight,
            minimum_weight=minimum_weight,
            dt=dt,
            construction_permission=False,
        )
        self.__related_to_fair_synapse = related_to_fair_synapse
        self.presynaptic_tagging = presynaptic_tagging
        self.postsynaptic_tagging = postsynaptic_tagging
        self.ltp_rate = SynapticPlasticityRate(rate=0.01) if ltp_rate is None else ltp_rate
        self.ltd_rate = SynapticPlasticityRate(rate=0.01) if ltd_rate is None else ltd_rate
        self.set_construction_permission(construction_permission)


    def __construct__(
        self,
        source: Iterable[int],
        target: Iterable[int],
        batch: int,
        maximum_weight: Union[float, torch.Tensor],
        minimum_weight: Union[float, torch.Tensor],
        dt: Union[float, torch.Tensor],
    ) -> None:
        """
        This function is related to the completion of the construction process.\
        Read more about `__construct__` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        source : Iterable of int
            The topology of each dendrite's spines.
        target : Iterable of int
            The topology of dendrites in the population.
        batch : int
            Determines the batch size.
        dt : float or torch.Tensor
            Time step in milliseconds.
        maximum_weight : float or torch.Tensor
            Determines the maximum possible synaptic weight.
        minimum_weight : float or torch.Tensor
            Determines the minimum possible synaptic weight.
        
        Returns
        -------
        None
        
        """
        super().__construct__(
            source=source,
            target=target,
            batch=batch,
            maximum_weight=maximum_weight,
            minimum_weight=minimum_weight,
            dt=dt,
        )
        self.presynaptic_tagging.meet_requirement(
            shape=(self.batch, *self.source, self.batch, *self.target)
            if not self.__related_to_fair_synapse else
            (self.batch, *self.source, 1, *[1]*len(self.target))
        )
        self.presynaptic_tagging.meet_requirement(dt=self.dt)
        self.postsynaptic_tagging.meet_requirement(shape=(self.batch, *self.target))
        self.postsynaptic_tagging.meet_requirement(dt=self.dt)
        self.ltp_rate.meet_requirement(dt=self.dt)
        self.ltp_rate.meet_requirement(maximum_weight=self.max)
        self.ltp_rate.meet_requirement(minimum_weight=self.min)
        self.ltd_rate.meet_requirement(dt=self.dt)
        self.ltd_rate.meet_requirement(maximum_weight=self.max)
        self.ltd_rate.meet_requirement(minimum_weight=self.min)


    @construction_required
    def __call__(
        self,
        neurotransmitter: torch.Tensor,
        neuromodulator: torch.Tensor,
        action_potential: torch.Tensor,
        synaptic_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate the module activity for a single step and returns the proper synaptic weight changes.
        
        Arguments
        ---------
        neurotransmitter : torch.Tensor
            Last axon terminal released neurotransmitters.
        neuromodulator : torch.Tensor
            Last synapse neuromodulators.
        action_potential : torch.Tensor
            Last dendrite action potential.
        synaptic_weights : torch.Tensor
            Last synaptic weights.

        Returns
        -------
        dw: torch.Tensor
            The proper synaptic weight changes.
        
        """
        presynaptic_tag = self.presynaptic_tagging(action_potential=neurotransmitter)
        postsynaptic_tag = self.postsynaptic_tagging(action_potential=action_potential)
        ltp_rate = self.ltp_rate(synaptic_weights=synaptic_weights.reshape(1, *self.source, 1, *self.target))
        ltd_rate = self.ltd_rate(synaptic_weights=synaptic_weights.reshape(1, *self.source, 1, *self.target))
        ltp = ltp_rate * presynaptic_tag * action_potential
        ltd = ltd_rate * neurotransmitter * postsynaptic_tag
        dw = (ltp - ltd) * self.dt
        return dw.mean([0, len(self.source)+1])


    def reset(
        self
    ) -> None:
        """
        Refractor and reset the axon and related moduls.
        
        Returns
        -------
        None
        
        """
        self.presynaptic_tagging.reset()
        self.postsynaptic_tagging.reset()
        self.ltp_rate.reset()
        self.ltd_rate.reset()