"""
This module specifies how axon terminals connect to dendritic spines in a synapse set.
"""


import torch
from typing import Union, Iterable
from typeguard import typechecked
from abc import ABC, abstractmethod
from constant_properties_protector import CPP
from construction_requirements_integrator import CRI, construction_required
from add_on_class import AOC, covering_around




@typechecked
class ConnectivityPattern(torch.nn.Module, CRI, ABC):
    """
    Basic class for all connectivity patters.

    Properties
    ----------
    source : Iterable of int, Protected
        The topology of axon terminals in the synapse.\
        Read more about protected properties in constant-properties-protector package documentation.
    target : Iterable of int, Protected
        The topology of dendrite spines in the synapse.\
        Read more about protected properties in constant-properties-protector package documentation.
    batch : int, Protected
        Determines the batch size.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    is_constructed: bool
        Indicates the completion status of the construction.\
        Read more about construction completion in construction-requirements-integrator package documentation.

    Arguments
    ---------
    source : Iterable of int, Construction Requirement
        Defines the topology of axon terminals in the synapse.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the synapse, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    target : Iterable of int, Construction Requirement
        Defines the topology of dendrite spines in the synapse.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the synapse, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    batch : int, Construction Requirement
        Determines the batch size. Should be same as network batch size.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the synapse, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the synapse, if you don't set it earlier.\
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
            dt=dt,
            ignore_overwrite_error=True,
            construction_permission=construction_permission,
        )


    def __construct__(
        self,
        source: Iterable[int],
        target: Iterable[int],
        batch: int,
        dt: Union[float, torch.Tensor],
    ) -> None:
        """
        This function is related to the completion of the construction process.\
        Read more about `__construct__` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        source : Iterable of int, Construction Requirement
            Defines the topology of axon terminals in the synapse.
        target : Iterable of int, Construction Requirement
            Defines the topology of dendrite spines in the synapse.\
        batch : int
            Determines the batch size.
        dt : float or torch.Tensor
            Time step in milliseconds.
        
        Returns
        -------
        None
        
        """
        self._source = source
        self._target = target
        self._batch = batch
        self.register_buffer("_dt", torch.as_tensor(dt))


    @abstractmethod
    @construction_required
    def __call__(
        self,
    ) -> torch.Tensor:
        """
        Calculates and returns the connection pattern.\
        Note that the connection pattern can change over time.

        Returns
        -------
        connectivity: torch.Tensor[bool]
            The output connectivity pattern.
        
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
@covering_around([ConnectivityPattern])
class InvertConnectivity(AOC):
    """
    Add-on class to invert given connectivity.

    Because this is an add-on class, we will only introduce the properties and functions it adds.\
    Read more about add-on classes in add-on-class package documentation.
    """

    @construction_required
    def __call__(
        self,
    ) -> torch.Tensor:
        """
        Calculates and returns the connection pattern.\
        According to the definition of this module,\
        this module in this function receives the output of the given pattern and inverts it.

        Returns
        -------
        connectivity: torch.Tensor[bool]
            The output connectivity pattern.
        
        """
        return ~(self.__core.__call__(self)).bool()




@typechecked
class AggConnectivity(ConnectivityPattern, ABC):
    """
    This modul will aggregate the output of given connectivity patterns.

    Properties
    ----------
    connectivity_patterns : Iterable[ConnectivityPattern]
        The given connectivity patterns.
    source: Iterable of int, Protected
        The topology of axon terminals in the synapse.\
        Read more about protected properties in constant-properties-protector package documentation.
    target: Iterable of int, Protected
        The topology of dendrite spines in the synapse.\
        Read more about protected properties in constant-properties-protector package documentation.
    batch : int, Protected
        Determines the batch size.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    is_constructed: bool
        Indicates the completion status of the construction.\
        Read more about construction completion in construction-requirements-integrator package documentation.

    Arguments
    ---------
    connectivity_patterns : Iterable[ConnectivityPattern], Necessary
        The given connectivity patterns to aggregate their outputs.
    source : Iterable of int, Construction Requirement
        Defines the topology of axon terminals in the synapse.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the synapse, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    target : Iterable of int, Construction Requirement
        Defines the topology of dendrite spines in the synapse.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the synapse, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    batch : int, Construction Requirement
        Determines the batch size. Should be same as network batch size.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the synapse, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the synapse, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    construction_permission : bool, Optional, default: True
        You can prevent the completeion of the construction by setting this parameter to `False`.\
        After that, you can change it calling `set_construction_permission(True)`.\
        It is useful when you are inheriting a module from it.\
        Read more about `construction_permission` in construction-requirements-integrator package documentation.
    """
    def __init__(
        self,
        connectivity_patterns: Iterable[ConnectivityPattern],
        source: Iterable[int] = None,
        target: Iterable[int] = None,
        batch: int = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            source=source,
            target=target,
            batch=batch,
            dt=dt,
            construction_permission=False,
        )
        self.connectivity_patterns = connectivity_patterns
        for i,connectivity_pattern in enumerate(connectivity_patterns):
            self.add_module(str(i), ConnectivityPattern)
        self.set_construction_permission(construction_permission)


    def __construct__(
        self,
        source: Iterable[int],
        target: Iterable[int],
        batch: int,
        dt: Union[float, torch.Tensor],
    ) -> None:
        """
        This function is related to the completion of the construction process.\
        It will help the given connectivity patterns to be constructed too.\
        Read more about `__construct__` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        source : Iterable of int, Construction Requirement
            Defines the topology of axon terminals in the synapse.
        target : Iterable of int, Construction Requirement
            Defines the topology of dendrite spines in the synapse.\
        batch : int
            Determines the batch size.
        dt : float or torch.Tensor
            Time step in milliseconds.
        
        Returns
        -------
        None
        
        """
        super().__construct__(
            source=source,
            target=target,
            batch=batch,
            dt=dt,
        )
        for connectivity_pattern in self.connectivity_patterns:
            connectivity_pattern.meet_requirement(source=source)
            connectivity_pattern.meet_requirement(target=target)
            connectivity_pattern.meet_requirement(batch=batch)
            connectivity_pattern.meet_requirement(dt=dt)


    @abstractmethod
    @construction_required
    def __call__(
        self,
    ) -> torch.Tensor:
        """
        Calculates and returns the connection pattern.\
        Note that the connection pattern can change over time.

        Returns
        -------
        connectivity: torch.Tensor[bool]
            The output connectivity pattern.
        
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
        for connectivity_pattern in self.connectivity_patterns:
            connectivity_pattern.reset()




@typechecked
class AndConnectivity(AggConnectivity):
    """
    This modul will compute the AND function on the output of given connectivity patterns.

    Read more about this module in AggConnectivity module documentaions.

    """

    @construction_required
    def __call__(
        self,
    ) -> torch.Tensor:
        """
        Calculates and returns the connection pattern.\
        According to the definition of this module,\
        this module in this function receives the output of the given patterns and\
        returns output of the AND function on them.

        Returns
        -------
        connectivity: torch.Tensor[bool]
            The output connectivity pattern.
        
        """
        pattern = torch.as_tensor(True)
        for connectivity_pattern in self.connectivity_patterns:
            pattern = pattern * connectivity_pattern()
        return pattern.bool()




@typechecked
class OrConnectivity(AggConnectivity):
    """
    This modul will compute the OR function on the output of given connectivity patterns.

    Read more about this module in AggConnectivity module documentaions.

    """

    @construction_required
    def __call__(
        self,
    ) -> torch.Tensor:
        """
        Calculates and returns the connection pattern.\
        According to the definition of this module,\
        this module in this function receives the output of the given patterns and\
        returns output of the OR function on them.

        Returns
        -------
        connectivity: torch.Tensor[bool]
            The output connectivity pattern.
        
        """
        pattern = torch.as_tensor(True)
        for connectivity_pattern in self.connectivity_patterns:
            pattern = pattern + connectivity_pattern()
        return pattern.bool()




@typechecked
class FixedConnectivity(ConnectivityPattern, ABC):
    """
    This modul will assume that connectivity will not change through time.\
    So it will record one connectivity in the construction process and always returns it.

    Properties
    ----------
    connectivity : torch.Tensor[bool]
        The output connectivity patterns.
    source: Iterable of int, Protected
        The topology of axon terminals in the synapse.\
        Read more about protected properties in constant-properties-protector package documentation.
    target: Iterable of int, Protected
        The topology of dendrite spines in the synapse.\
        Read more about protected properties in constant-properties-protector package documentation.
    batch : int, Protected
        Determines the batch size.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    is_constructed: bool
        Indicates the completion status of the construction.\
        Read more about construction completion in construction-requirements-integrator package documentation.

    Arguments
    ---------
    source : Iterable of int, Construction Requirement
        Defines the topology of axon terminals in the synapse.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the synapse, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    target : Iterable of int, Construction Requirement
        Defines the topology of dendrite spines in the synapse.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the synapse, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    batch : int, Construction Requirement
        Determines the batch size. Should be same as network batch size.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the synapse, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the synapse, if you don't set it earlier.\
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
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            source=source,
            target=target,
            batch=batch,
            dt=dt,
            construction_permission=construction_permission,
        )


    def __construct__(
        self,
        source: Iterable[int],
        target: Iterable[int],
        batch: int,
        dt: Union[float, torch.Tensor],
    ) -> None:
        """
        This function is related to the completion of the construction process.\
        Read more about `__construct__` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        source : Iterable of int, Construction Requirement
            Defines the topology of axon terminals in the synapse.
        target : Iterable of int, Construction Requirement
            Defines the topology of dendrite spines in the synapse.\
        batch : int
            Determines the batch size.
        dt : float or torch.Tensor
            Time step in milliseconds.
        
        Returns
        -------
        None
        
        """
        super().__construct__(
            source=source,
            target=target,
            batch=batch,
            dt=dt,
        )
        self.is_constructed = True
        self.register_buffer('connectivity', self._generate_connectivity())


    @abstractmethod
    @construction_required
    def _generate_connectivity(
        self
    ) -> torch.Tensor:
        """
        This function will be called once in construction process.\
        It should calculate and return a connectivity pattern.\
        The module will always return this connectivity as output.

        Returns
        -------
        connectivity: torch.Tensor[bool]
            The output connectivity pattern.
        
        """
        pass


    @construction_required
    def __call__(
        self,
    ) -> torch.Tensor:
        """
        Returns the connection pattern.

        Returns
        -------
        connectivity: torch.Tensor[bool]
            The output connectivity pattern.
        
        """
        return self.connectivity




@typechecked
class AutapseConnectivity(FixedConnectivity):
    """
    This module provides a diagonal matrix as a connection matrix.\
    It will be useful for modeling autapse synaptic connections.

    Read more about this module in FixedConnectivity module documentaions.

    """

    def __construct__(
        self,
        source: Iterable[int],
        target: Iterable[int],
        batch: int,
        dt: Union[float, torch.Tensor],
    ) -> None:
        """
        This function is related to the completion of the construction process.\
        Read more about `__construct__` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        source : Iterable of int, Construction Requirement
            Defines the topology of axon terminals in the synapse.
        target : Iterable of int, Construction Requirement
            Defines the topology of dendrite spines in the synapse.\
        batch : int
            Determines the batch size.
        dt : float or torch.Tensor
            Time step in milliseconds.
        
        Returns
        -------
        None
        
        """
        if source!=target:
            raise Exception(f"Can not build an autapse connectivity pattern between two different shapes: {source} != {target}")
        super().__construct__(
            source=source,
            target=target,
            batch=batch,
            dt=dt,
        )


    @construction_required
    def _generate_connectivity(
        self,
    ) -> torch.Tensor:
        """
        This function provides a diagonal matrix as the connection matrix.

        Returns
        -------
        connectivity: torch.Tensor[bool]
            The output connectivity pattern.
        
        """
        return torch.diag(torch.ones(torch.prod(torch.as_tensor(self.source)))).reshape(1, *self.source, 1, *self.target).bool()




@typechecked
class RandomConnectivity(FixedConnectivity):
    """
    This modul will assume that connectivity will not change through time.\
    So it will record one connectivity in the construction process and always returns it.

    Properties
    ----------
    rate : torch.Tensor
        Determines the rate of connectivity.
    connectivity : torch.Tensor[bool]
        The output connectivity patterns.
    source: Iterable of int, Protected
        The topology of axon terminals in the synapse.\
        Read more about protected properties in constant-properties-protector package documentation.
    target: Iterable of int, Protected
        The topology of dendrite spines in the synapse.\
        Read more about protected properties in constant-properties-protector package documentation.
    batch : int, Protected
        Determines the batch size.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    is_constructed: bool
        Indicates the completion status of the construction.\
        Read more about construction completion in construction-requirements-integrator package documentation.

    Arguments
    ---------
    rate : float or torch.Tensor (single value), Necessary
        Determines the rate of connectivity.
    source : Iterable of int, Construction Requirement
        Defines the topology of axon terminals in the synapse.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the synapse, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    target : Iterable of int, Construction Requirement
        Defines the topology of dendrite spines in the synapse.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the synapse, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    batch : int, Construction Requirement
        Determines the batch size. Should be same as network batch size.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the synapse, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the synapse, if you don't set it earlier.\
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
        source: Iterable[int] = None,
        target: Iterable[int] = None,
        batch: int = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            source=source,
            target=target,
            batch=batch,
            dt=dt,
            construction_permission=False,
        )
        self.register_buffer('rate', torch.as_tensor(rate))
        if self.rate.numel()!=1:
            raise Exception("Rate must be a single float value.")
        self.set_construction_permission(construction_permission)


    @construction_required
    def _generate_connectivity(
        self
    ) -> torch.Tensor:
        """
        This function provides a random connectivity matrix with the given connectivity rate.

        Returns
        -------
        connectivity: torch.Tensor[bool]
            The output connectivity pattern.
        
        """
        return torch.rand(1, *self.source, 1, *self.target).uniform_() > self.rate




@typechecked
class RandomFixedCouplingConnectivity(RandomConnectivity):
    """
    This module provides a random matrix as a connection matrix\
    folloing "random fixed coupling" strategy.

    Read more about this module in RandomConnectivity module documentaions.

    """

    @construction_required
    def _generate_connectivity(
        self,
    ) -> torch.Tensor:
        """
        This function provides a random connectivity matrix with the given connectivity rate\
        and follows "random fixed coupling" strategy.

        Returns
        -------
        connectivity: torch.Tensor[bool]
            The output connectivity pattern.
        
        """
        count = int(torch.prod(torch.as_tensor(*self.source, *self.target))*self.rate)
        output = torch.rand(1, *self.source, 1, *self.target)
        threshold = output.reshape(-1).sort()[0][-count]
        return (output >= threshold)




@typechecked
class RandomFixedPresynapticPartnersConnectivity(RandomConnectivity):
    """
    This module provides a random matrix as a connection matrix\
    folloing "random fixed presynaptic partners" strategy.

    Read more about this module in RandomConnectivity module documentaions.

    """

    @construction_required
    def _generate_connectivity(
        self,
    ) -> torch.Tensor:
        """
        This function provides a random connectivity matrix with the given connectivity rate\
        and follows "random fixed presynaptic partners" strategy.

        Returns
        -------
        connectivity: torch.Tensor[bool]
            The output connectivity pattern.
        
        """
        count = int(torch.prod(torch.as_tensor(self.source))*self.rate)
        output = torch.rand(1, *self.source, 1, *self.target)
        flatted = output.reshape(-1, *self.target)
        threshold = torch.topk(flatted, flatted.shape[0], dim=0, largest=False)[0][-count]
        return (output >= threshold)
