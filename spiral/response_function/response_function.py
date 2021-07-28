"""
This module will provide response functions for action potentials.
"""


import torch
from typing import Union, Iterable
from typeguard import typechecked
from constant_properties_protector import CPP
from construction_requirements_integrator import CRI, construction_required
from spiral.analysis import Analyzer, analysis_point, analytics




@typechecked
class ResponseFunction(torch.nn.Module, CRI):
    """
    Basic class for response functions.\
    This module will provide no-operation response function and always returns the input.\
    Since this is a base class, it receives different inputs for construction that it\
    may not use all of them. But its children will need them.

    Properties
    ----------
    shape : Iterable of int, Protected
        The topology of input action potential.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    is_constructed: bool
        Indicates the completion status of the construction.\
        Read more about construction completion in construction-requirements-integrator package documentation.

    Arguments
    ---------
    shape : Iterable of int, Construction Requirement
        The topology of input action potential.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the axon or synaptic plasticity, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the axon or synaptic plasticity, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    construction_permission : bool, Optional, default: True
        You can prevent the completeion of the construction by setting this parameter to `False`.\
        After that, you can change it calling `set_construction_permission(True)`.\
        It is useful when you are inheriting a module from it.\
        Read more about `construction_permission` in construction-requirements-integrator package documentation.
    """
    def __init__(
        self,
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        torch.nn.Module.__init__(self)
        CPP.protect(self, 'shape')
        CPP.protect(self, 'dt')
        CRI.__init__(
            self,
            shape=shape,
            dt=dt,
            ignore_overwrite_error=True,
            construction_permission=construction_permission,
        )


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
        shape : Iterable of int
            The topology of input action potential.
        dt : float or torch.Tensor
            Time step in milliseconds.
        
        Returns
        -------
        None
        
        """
        self._shape = shape
        self.register_buffer("_dt", torch.as_tensor(dt))


    @construction_required
    def __call__(
        self,
        action_potential: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate the module activity for a single step and returns the response.
        
        Arguments
        ---------
        action_potential : torch.Tensor
            Input action potential.

        Returns
        -------
        response: torch.Tensor
            The response.
        
        """
        return action_potential.float()


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
class CompositeResponseFunction(ResponseFunction): #In order
    """
    It will make a composition of response functions and pass the output of one as input of the other.\
    The response functions will be chained in order that they are inserted.

    Properties
    ----------
    response_functions: Iterable[ResponseFunction]
        The compositing response functions.
    shape : Iterable of int, Protected
        The topology of input action potential.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    is_constructed: bool
        Indicates the completion status of the construction.\
        Read more about construction completion in construction-requirements-integrator package documentation.

    Arguments
    ---------
    response_functions: Iterable[ResponseFunction], Necessary
        The compositing response functions.
    shape : Iterable of int, Construction Requirement
        The topology of input action potential.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the axon or synaptic plasticity, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the axon or synaptic plasticity, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    construction_permission : bool, Optional, default: True
        You can prevent the completeion of the construction by setting this parameter to `False`.\
        After that, you can change it calling `set_construction_permission(True)`.\
        It is useful when you are inheriting a module from it.\
        Read more about `construction_permission` in construction-requirements-integrator package documentation.
    """
    def __init__(
        self,
        response_functions: Iterable[ResponseFunction],
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            shape=shape,
            dt=dt,
            construction_permission=False,
        )
        self.response_functions = response_functions
        for i,response_function in enumerate(response_functions):
            self.add_module(str(i), response_function)
        self.set_construction_permission(construction_permission)


    def __construct__(
        self,
        shape: Iterable[int],
        dt: Union[float, torch.Tensor],
    ) -> None:
        """
        This function is related to the completion of the construction process.\
        It will help the given response functions to be constructed too.\
        Read more about `__construct__` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        shape : Iterable of int
            The topology of input action potential.
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
        for response_function in self.response_functions:
            response_function.meet_requirement(shape=shape)
            response_function.meet_requirement(dt=dt)


    @construction_required
    def __call__(
        self,
        action_potential: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate the module activity for a single step and returns the response.
        
        Arguments
        ---------
        action_potential : torch.Tensor
            Input action potential.

        Returns
        -------
        response: torch.Tensor
            The response.
        
        """
        for response_function in self.response_functions:
            action_potential = response_function(action_potential=action_potential)
        return action_potential


    def reset(
        self
    ) -> None:
        """
        Refractor and reset the axon and related moduls.
        
        Returns
        -------
        None
        
        """
        for response_function in self.response_functions:
            response_function.reset()




@typechecked
class ScalingResponseFunction(ResponseFunction):
    """
    This module simply multiplies the input by a scaling factor.

    Properties
    ----------
    scale : torch.Tensor
        Scaling factor that applies on inputs.
    shape : Iterable of int, Protected
        The topology of input action potential.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    is_constructed: bool
        Indicates the completion status of the construction.\
        Read more about construction completion in construction-requirements-integrator package documentation.

    Arguments
    ---------
    scale : float or torch.Tensor, Necessary
        Scaling factor that applies on inputs.
    shape : Iterable of int, Construction Requirement
        The topology of input action potential.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the axon or synaptic plasticity, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the axon or synaptic plasticity, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    construction_permission : bool, Optional, default: True
        You can prevent the completeion of the construction by setting this parameter to `False`.\
        After that, you can change it calling `set_construction_permission(True)`.\
        It is useful when you are inheriting a module from it.\
        Read more about `construction_permission` in construction-requirements-integrator package documentation.
    """
    def __init__(
        self,
        scale: Union[float, torch.Tensor],
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            shape=shape,
            dt=dt,
            construction_permission=construction_permission,
        )
        self.register_buffer("scale", torch.as_tensor(scale))


    @construction_required
    def __call__(
        self,
        action_potential: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate the module activity for a single step and returns the response.
        
        Arguments
        ---------
        action_potential : torch.Tensor
            Input action potential.

        Returns
        -------
        response: torch.Tensor
            The response.
        
        """
        return action_potential*self.scale




@typechecked
class LeakyResponseFunction(ResponseFunction):
    """
    This module will model a leaky response function that integrates inputs in all steps and\
    has leakage through the time.

    Properties
    ----------
    tau : torch.Tensor
        Leakage time constant.
    response : torch.Tensor, Protected
        Last updated response.\
        Read more about protected properties in constant-properties-protector package documentation.
    shape : Iterable of int, Protected
        The topology of input action potential.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    is_constructed: bool
        Indicates the completion status of the construction.\
        Read more about construction completion in construction-requirements-integrator package documentation.
    analyzable : bool
        Indicates the analyzability status of the module.
    monitor : Monitor
        Exists just if the module be analyzable.\
        You can get a recorded sequence of important features through the `monitor`.\
        Read more about `Monitor` in monitor module documentation.

    Arguments
    ---------
    tau : float or torch.Tensor, Optional, default: 15.
        Determines leakage time constant.
    shape : Iterable of int, Construction Requirement
        The topology of input action potential.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the axon or synaptic plasticity, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the axon or synaptic plasticity, if you don't set it earlier.\
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
        tau: Union[float, torch.Tensor] = 15.,
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None,
        analyzable: bool = False,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            shape=shape,
            dt=dt,
            construction_permission=construction_permission,
        )
        self.register_buffer("tau", torch.as_tensor(tau))
        CPP.protect(self, 'response')
        Analyzer.__init__(self, analyzable)
        Analyzer.scout(self, state_variables=['response'])


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
        shape : Iterable of int
            The topology of input action potential.
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
        self.register_buffer("_response", torch.zeros(*self.shape))


    @construction_required
    @analysis_point
    def __call__(
        self,
        action_potential: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate the module activity for a single step and returns the response.
        
        Arguments
        ---------
        action_potential : torch.Tensor
            Input action potential.

        Returns
        -------
        response: torch.Tensor
            The response.
        
        """
        self._response += action_potential.float() - self.response * self.dt / self.tau
        return self.response


    @construction_required
    def reset(
        self,
    ) -> None:
        """
        Refractor and reset the axon and related moduls.
        
        Returns
        -------
        None
        
        """
        self._response.zero_()
        if self.analyzable:
            self.monitor.reset()


    @analytics
    def plot_response(
        self,
        axes,
        **kwargs
    ) -> None:
        """
        Draw a plot of response value on `axes`.

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
        y = self.monitor['response'].reshape(self.monitor['response'].shape[0],-1)
        time_range = (0, y.shape[0])
        x = torch.arange(*time_range)*self.dt
        population_alpha = 1/y.shape[1]
        aggregated = y.mean(axis=1)
        axes.plot(x, aggregated, color='black', **kwargs)
        axes.plot(x, y, alpha=population_alpha, color='black')
        axes.set_ylabel('response')
        axes.set_xlabel('time (ms)')
        axes.set_xlim(time_range)




@typechecked
class FlatResponseFunction(ResponseFunction):
    """
    This module will get inputs and integrate them through time, as response value.

    Properties
    ----------
    response : torch.Tensor, Protected
        Last updated response.\
        Read more about protected properties in constant-properties-protector package documentation.
    shape : Iterable of int, Protected
        The topology of input action potential.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    is_constructed: bool
        Indicates the completion status of the construction.\
        Read more about construction completion in construction-requirements-integrator package documentation.

    Arguments
    ---------
    shape : Iterable of int, Construction Requirement
        The topology of input action potential.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the axon or synaptic plasticity, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the axon or synaptic plasticity, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    construction_permission : bool, Optional, default: True
        You can prevent the completeion of the construction by setting this parameter to `False`.\
        After that, you can change it calling `set_construction_permission(True)`.\
        It is useful when you are inheriting a module from it.\
        Read more about `construction_permission` in construction-requirements-integrator package documentation.
    """
    def __init__(
        self,
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            shape=shape,
            dt=dt,
            construction_permission=construction_permission,
        )
        CPP.protect(self, 'response')


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
        shape : Iterable of int
            The topology of input action potential.
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
        self.register_buffer("_response", torch.zeros(*self.shape))


    @construction_required
    def __call__(
        self,
        action_potential: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate the module activity for a single step and returns the response.
        
        Arguments
        ---------
        action_potential : torch.Tensor
            Input action potential.

        Returns
        -------
        response: torch.Tensor
            The response.
        
        """
        self._response += action_potential
        return self.response


    @construction_required
    def reset(
        self,
    ) -> None:
        """
        Refractor and reset the axon and related moduls.
        
        Returns
        -------
        None
        
        """
        self._response.zero_()




@typechecked
class LimitedFlatResponseFunction(ResponseFunction):
    """
    This module will integrate inputs in a limited time period as response value.

    Properties
    ----------
    duration : torch.Tensor, Protected
        The watching time period length in milliseconds.
        Read more about protected properties in constant-properties-protector package documentation.
    action_potential_history : torch.Tensor, Protected
        The history of input action potentials in watching time period.
        Read more about protected properties in constant-properties-protector package documentation.
    response : torch.Tensor, Protected
        Last updated response.\
        Read more about protected properties in constant-properties-protector package documentation.
    shape : Iterable of int, Protected
        The topology of input action potential.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    is_constructed: bool
        Indicates the completion status of the construction.\
        Read more about construction completion in construction-requirements-integrator package documentation.

    Arguments
    ---------
    duration : float or torch.Tensor, Optional, default=10.
        The watching time period length in milliseconds.
    shape : Iterable of int, Construction Requirement
        The topology of input action potential.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the axon or synaptic plasticity, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the axon or synaptic plasticity, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    construction_permission : bool, Optional, default: True
        You can prevent the completeion of the construction by setting this parameter to `False`.\
        After that, you can change it calling `set_construction_permission(True)`.\
        It is useful when you are inheriting a module from it.\
        Read more about `construction_permission` in construction-requirements-integrator package documentation.
    """
    def __init__(
        self,
        duration: Union[float, torch.Tensor] = 10.,
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            shape=shape,
            dt=dt,
            construction_permission=False,
        )
        self.add_to_construction_requirements(duration=duration)
        CPP.protect(self, 'action_potential_history')
        CPP.protect(self, 'duration')
        self.set_construction_permission(construction_permission)


    def __construct__(
        self,
        shape: Iterable[int],
        dt: Union[float, torch.Tensor],
        duration: Union[float, torch.Tensor],
    ) -> None:
        """
        This function is related to the completion of the construction process.\
        Read more about `__construct__` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        shape : Iterable of int
            The topology of input action potential.
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
        self.register_buffer("_duration", (torch.as_tensor(duration)//self.dt).type(torch.int64))
        if len(self.duration.shape)!=0 and self.duration.shape!=self.shape:
            raise Exception(f"Wrong shape for response limited flat function duration. Expected {self.shape} or a single value but got {self.duration.shape}")
        history_length = self.duration.max()+1
        self.register_buffer("_action_potential_history", torch.zeros((history_length,*self.shape)))


    @construction_required
    def __call__(
        self,
        action_potential: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate the module activity for a single step and returns the response.
        
        Arguments
        ---------
        action_potential : torch.Tensor
            Input action potential.

        Returns
        -------
        response: torch.Tensor
            The response.
        
        """
        self._action_potential_history = torch.cat([action_potential.unsqueeze(0), self.action_potential_history])
        if self.duration.numel()>1:
            self._action_potential_history = self.action_potential_history.scatter(
                dim=0, index=self.duration.unsqueeze(0)+1, src=torch.zeros_like(self.duration).unsqueeze(0).float()
            )
        self._action_potential_history = self.action_potential_history[:-1]
        return self.action_potential_history.sum(axis=0)


    @construction_required
    def reset(
        self,
    ) -> None:
        """
        Refractor and reset the axon and related moduls.
        
        Returns
        -------
        None
        
        """
        self._action_potential_history.zero_()