"""
This module will provide learning rates for synaptic plasticities.
"""


import torch
from typing import Union, Iterable
from typeguard import typechecked
from abc import ABC, abstractmethod
from constant_properties_protector import CPP
from construction_requirements_integrator import CRI, construction_required




@typechecked
class SynapticPlasticityRate(torch.nn.Module, CRI):
    """
    Basic class for synaptic plasticities rates. This module will provide constant learning rate.\
    Note that, contrary to what is implemented in this module, the learning rate may be time dependent.\
    For this reason, it also receives a fixed time module.\
    Synaptic weights-dependent learning rates are also common.\
    For this reason, the __call__ function receives values of synaptic weights\
    and possible weight range is required for construction.

    Properties
    ----------
    rate : torch.Tensor
        Determines learning rate constant value.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    max : torch.Tensor
        Shows the maximum possible synaptic weight.
    min : torch.Tensor
        Shows the minimum possible synaptic weight.
    is_constructed: bool
        Indicates the completion status of the construction.\
        Read more about construction completion in construction-requirements-integrator package documentation.

    Arguments
    ---------
    rate : float or torch.Tensor, Necessary
        Determines learning rate constant value.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the source dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    maximum_weight : float or torch.Tensor, Construction Requirement
        Determines the maximum possible synaptic weight.
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the source dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    minimum_weight : float or torch.Tensor, Construction Requirement
        Determines the minimum possible synaptic weight.
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the source dendrite, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    construction_permission : bool, Optional, default: True
        You can prevent the completeion of the construction by setting this parameter to `False`.\
        After that, you can change it calling `set_construction_permission(True)`.\
        It is useful when you are inheriting a module from it.\
        Read more about `construction_permission` in construction-requirements-integrator package documentation.
    """
    def __init__(
        self,
        rate: Union[float, torch.Tensor],
        dt: Union[float, torch.Tensor] = None,
        maximum_weight: Union[float, torch.Tensor] = None,
        minimum_weight: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.register_buffer("rate", torch.as_tensor(rate))
        CPP.protect(self, 'dt')
        CRI.__init__(
            self,
            dt=dt,
            maximum_weight=maximum_weight,
            minimum_weight=minimum_weight,
            ignore_overwrite_error=True,
            construction_permission=construction_permission,
        )


    def __construct__(
        self,
        dt: Union[float, torch.Tensor],
        maximum_weight: Union[float, torch.Tensor],
        minimum_weight: Union[float, torch.Tensor],
    ) -> None:
        """
        This function is related to the completion of the construction process.\
        Read more about `__construct__` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        dt : float or torch.Tensor
            Time step in milliseconds.
        
        Returns
        -------
        None
        
        """
        self.register_buffer("_dt", torch.as_tensor(dt))
        self.register_buffer("max", torch.as_tensor(maximum_weight))
        self.register_buffer("min", torch.as_tensor(minimum_weight))


    @construction_required
    def __call__(
        self,
        synaptic_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the learning rate.
        
        Arguments
        ---------
        synaptic_weights : torch.Tensor
            Synaptic weights.

        Returns
        -------
        learning_rate: torch.Tensor
            The learning rate.
        
        """
        return self.rate


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
class WeightDependentRate(SynapticPlasticityRate):
    """
    This modul will provide weight dependent learning rate.

    Read more about this module in SynapticPlasticityRate module documentaions.

    """

    @construction_required
    def __call__(
        self,
        synaptic_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the learning rate.
        
        Arguments
        ---------
        synaptic_weights : torch.Tensor
            Synaptic weights.

        Returns
        -------
        learning_rate: torch.Tensor
            The learning rate.
        
        """
        return self.rate * (self.max-synaptic_weights) * (synaptic_weights-self.min)




@typechecked
class DescendingSynapticPlasticityRate(SynapticPlasticityRate):
    """
    This module will provide a learning rate that decays during process.

    Properties
    ----------
    rate : torch.Tensor
        Determines prime learning rate constant value.
    tau: torch.Tensor
        Decay time constant in milliseconds.
    current_rate : torch.Tensor
        Shows current learning rate (after decaying).
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    max : torch.Tensor
        Shows the maximum possible synaptic weight.
    min : torch.Tensor
        Shows the minimum possible synaptic weight.
    is_constructed: bool
        Indicates the completion status of the construction.\
        Read more about construction completion in construction-requirements-integrator package documentation.

    Arguments
    ---------
    rate : float or torch.Tensor, Necessary
        Determines learning rate constant value.
    tau : float or torch.Tensor, Optional, default: 1000.
        Determines decay time constant in milliseconds.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the synapse, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    maximum_weight : float or torch.Tensor, Construction Requirement
        Determines the maximum possible synaptic weight.
        Read more about construction requirement in construction-requirements-integrator package documentation.
    minimum_weight : float or torch.Tensor, Construction Requirement
        Determines the minimum possible synaptic weight.
        Read more about construction requirement in construction-requirements-integrator package documentation.
    construction_permission : bool, Optional, default: True
        You can prevent the completeion of the construction by setting this parameter to `False`.\
        After that, you can change it calling `set_construction_permission(True)`.\
        It is useful when you are inheriting a module from it.\
        Read more about `construction_permission` in construction-requirements-integrator package documentation.
    """
    def __init__(
        self,
        rate: Union[float, torch.Tensor],
        tau: Union[float, torch.Tensor] = 1000.,
        dt: Union[float, torch.Tensor] = None,
        maximum_weight: Union[float, torch.Tensor] = None,
        minimum_weight: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            rate=rate,
            dt=dt,
            maximum_weight=maximum_weight,
            minimum_weight=minimum_weight,
            construction_permission=construction_permission,
        )
        self.register_buffer("tau", torch.as_tensor(tau))
        self.register_buffer("current_rate", self.rate.detach().clone())


    @construction_required
    def __call__(
        self,
        synaptic_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the learning rate.
        
        Arguments
        ---------
        synaptic_weights : torch.Tensor
            Synaptic weights.

        Returns
        -------
        learning_rate: torch.Tensor
            The learning rate.
        
        """
        learning_rate = self.current_rate.detach().clone()
        self.current_rate -= self.current_rate * self.dt / self.tau
        return learning_rate


    def reset(
        self
    ) -> None:
        """
        Refractor and reset the axon and related moduls.
        
        Returns
        -------
        None
        
        """
        self.current_rate = self.rate.detach().clone()