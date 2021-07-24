"""
An axon, or nerve fiber, is a long, slender projection of a nerve cell, or neuron,\
in vertebrates, that typically conducts electrical impulses known as action potentials\
away from the nerve cell body. The function of the axon is to transmit information to\
different neurons, muscles, and glands.\
The same module will be responsible for applying delays in action potential transmition and\
modeling the response function of action potential in synaptic transmition.
"""


import torch
from typing import Union, Iterable
from typeguard import typechecked
from constant_properties_protector import CPP
from construction_requirements_integrator import CRI, construction_required
from spiral.analysis import Analyzer, analysis_point, analytics
from spiral.response_function.response_function import ResponseFunction
from spiral.myelination.myelination import Myelination




@typechecked
class Axon(torch.nn.Module, CRI):
    """
    Class for usual axons.

    The purpose of the axon is to receive action potential from a connected neuron,\
    transfer it to terminals by considering time delays, amplify them according to\
    the condition of the terminals, apply response function, and transfer neurotransmitters\
    to the connected synapse.\
    This axonal model has the ability to connect to all types of neurons (spike or flow axis)\
    and their different modeling.\
    The destination of this module will always be synapses. The same module can also be used\
    as an neuromodulatory axon for transmitting neuromodulators. For this, it is only important\
    to connect the axon to the synapse as a neuromodulatory axon.

    Properties
    ----------
    name : str, Protected
        The name to be uniquely accessible in the connected soma or synapse.\
        Read more about protected properties in constant-properties-protector package documentation.
    shape : Iterable of int, Protected
        The topology of axons in the population.\
        Read more about protected properties in constant-properties-protector package documentation.
    batch : int, Protected
        Determines the batch size.\
        Read more about protected properties in constant-properties-protector package documentation.
    terminal : Iterable of int, Protected
        The topology of each axon's terminal points.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt: torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    is_excitatory : torch.Tensor[bool], Protected
        Determines which axons have a excitatory role and which have an inhibitory role.\
        Read more about protected properties in constant-properties-protector package documentation.
    response_function : ResponseFunction
        Determines how the response function is calculated.\
        Read more about response functions in Spiral.ResponseFunction module documentation.
    delay : torch.Tensor, Protected
        Determines the time lag of each axon (and even terminals) in active potential transmission in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    action_potential_history : torch.Tensor, Protected
        Indicates the memory of received action potentials.\
        Read more about protected properties in constant-properties-protector package documentation.
    neurotransmitter : torch.Tensor, Protected
        Indicates the amount of neurotransmitter being released at each terminal.\
        Read more about protected properties in constant-properties-protector package documentation.
    is_constructed : bool
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
    name : str, Construction Requirement
        The name to be uniquely accessible in the connected soma or synapse.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the source soma, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    shape : Iterable of int, Construction Requirement
        The topology of axon in the population. Should be same as shape of the connected soma.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the source soma, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    batch : int, Construction Requirement
        Determines the batch size. Should be same as network batch size.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.\
        It will be automatically set based on the source soma, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    terminal : Iterable of int, Optional, default: ()
        The topology of each axon's terminal points.\
        `(*shape, *terminal)` should be same as source shape of the connected synapse.
    response_function : ResponseFunction, Optional, default: ResponseFunction()
        Determines how the response function is calculated.\
        Read more about response functions in Spiral.ResponseFunction module documentation.
    is_excitatory : bool or torch.Tensor[bool], Optional, default: True
        Determines which axons have a excitatory role and which have an inhibitory role.\
        The shape if is_excitatory can be same as axon.shape or (*axon.shape, *axon.terminal)\
        or it can be a single value (for all the population).
    delay : float or torch.Tensor, Optional, default: 0
        Determines the time lag of each axon (and even terminals) in active potential transmission in milliseconds.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds. Should be same as network time step.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.
        It will be automatically set based on the source soma, if you don't set it earlier.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    myelination_model : Myelination, TODO
    myelination : bool, TODO
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
        terminal: Iterable[int] = (),
        response_function: ResponseFunction = None,
        is_excitatory: Union[bool, torch.Tensor] = True,
        delay: Union[float, torch.Tensor] = 0.,
        dt: Union[float, torch.Tensor] = None,
        myelination_model: Myelination = None, # Does nothing now. [TODO]
        myelination: bool = True, # Does nothing now. [TODO]
        analyzable: bool = False,
        construction_permission: bool = True,
    ) -> None:
        torch.nn.Module.__init__(self)
        CPP.protect(self, 'name')
        CPP.protect(self, 'shape')
        CPP.protect(self, 'batch')
        CPP.protect(self, 'terminal')
        CPP.protect(self, 'delay')
        CPP.protect(self, 'dt')
        CPP.protect(self, 'action_potential_history')
        CPP.protect(self, 'neurotransmitter')
        CPP.protect(self, 'is_excitatory')
        self.register_buffer("_is_excitatory", torch.as_tensor(is_excitatory))
        self.response_function = ResponseFunction() if response_function is None else response_function
        Analyzer.__init__(self, analyzable)
        Analyzer.scout(self, state_variables=['neurotransmitter'])
        CRI.__init__(
            self,
            name=name,
            shape=shape,
            batch=batch,
            terminal=terminal,
            delay=delay,
            dt=dt,
            construction_permission=construction_permission,
            ignore_overwrite_error=True,
        )


    def __construct__(
        self,
        name: str,
        shape: Iterable[int],
        batch: int,
        terminal: Iterable[int],
        delay: Union[float, torch.Tensor],
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
        terminal : Iterable of int
            The topology of each axon's terminal points.
        delay : float or torch.Tensor
            Determines the time lag of each axon (and even terminals) in active potential transmission in milliseconds.
        dt : float or torch.Tensor
            Time step in milliseconds.
        
        Returns
        -------
        None
        
        """
        self._name = name
        self._shape = (*shape,)
        self._batch = batch
        self._terminal = terminal

        if self.is_excitatory.numel()>1 and self.is_excitatory.shape!=self.shape and self.is_excitatory.shape!=(*self.shape, *self.terminal):
            raise Exception(f"`is_excitatory` with shape={self.is_excitatory} does not match the axon with shape={self.shape} and terminal={self.terminal}. Expected {self.shape} or {(*self.shape, *self.terminal)} or a single vale.")
        if self.is_excitatory.numel()==1:
            self._is_excitatory = self.is_excitatory.reshape(1, *[1]*len(self.shape), *[1]*len(self.terminal))
        elif self.is_excitatory.shape==self.shape:
            self._is_excitatory = self.is_excitatory.reshape(1, *self.is_excitatory.shape, *[1]*len(self.terminal))
        else:
            self._is_excitatory = self.is_excitatory.reshape(1, *self.is_excitatory.shape)

        self.register_buffer("_dt", torch.as_tensor(dt))
        self.register_buffer("_delay", (torch.as_tensor(delay)//self.dt).type(torch.int64))
        
        terminal_shape = (*self.shape, *self.terminal)
        if (
            len(self.delay.shape)>(len(terminal_shape))
        ) or (
            len(self.delay.shape)!=0 and self.delay.shape!=terminal_shape[-len(self.delay.shape):]
        ) :
            raise Exception(f"Wrong shape for axon delay. Expected {' or '.join([str(terminal_shape[-i:]) for i in range(len(terminal_shape))])} or a single value but got {self.delay.shape}")
        history_length = self.delay.max()+1
        self.register_buffer("_action_potential_history", torch.zeros((history_length, self.batch, *self.shape)))
        self.register_buffer("_neurotransmitter", torch.zeros(self.batch, *self.shape, *self.terminal))
        self.response_function.meet_requirement(shape=(self.batch, *self.shape, *self.terminal))
        self.response_function.meet_requirement(dt=self.dt)


    def _update_action_potential_history(
        self,
        action_potential: torch.Tensor
    ) -> None:
        """
        Inserts the new action potential into memory and removes the oldest.

        Arguments
        ---------
        action_potential : torch.Tensor
            The new action potential.

        Returns
        -------
        None
        
        """
        self._action_potential_history = torch.cat((action_potential.unsqueeze(0), self.action_potential_history[:-1]))


    def _get_delayed_action_potential(
        self
    ) -> torch.Tensor:
        """
        Calculates and returns potentials reached to terminals, taking into account time delays.

        Returns
        -------
        None
        
        """
        if self.delay.numel()==1:
            return self.action_potential_history[self.delay].reshape(
                *self.action_potential_history.shape[1:], *[1]*len(self.terminal)
            ).repeat(
                *[1]*len(self.action_potential_history.shape[1:]), *self.terminal
            )
        else:
            history = self.action_potential_history.reshape(
                *self.action_potential_history.shape, *[1]*len(self.terminal)
            ).repeat(
                *[1]*len(self.action_potential_history.shape), *self.terminal
            )
            delay = self.delay.reshape(
                1, *[1]*(len(self.action_potential_history.shape[1:])-len(self.delay.shape)), *self.delay.shape
            ).repeat(
                1, *history.shape[1:-len(self.delay.shape)], *[1]*len(self.delay.shape)
            )
            return history.gather(dim=0, index=delay)[0]


    @construction_required
    @analysis_point
    def forward(
        self,
        action_potential: torch.Tensor,
    ) -> None:
        """
        Simulate the axon activity for a single step.

        Arguments
        ---------
        action_potential : torch.Tensor
            The new action potential.

        Returns
        -------
        None
        
        """
        self._update_action_potential_history(action_potential=action_potential)
        self._neurotransmitter = self.response_function(action_potential=self._get_delayed_action_potential())


    @construction_required
    def release(
        self
    ) -> torch.Tensor:
        """
        Calculates the output neurotransmitter and consider excitation and inhibition.
        
        Returns
        -------
        neurotransmitter: torch.Tensor
            Output neurotransmitter.
        
        """
        return self.neurotransmitter * (2*self.is_excitatory-1)


    @construction_required
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
        self.response_function.reset()
        if self.analyzable:
            self.monitor.reset()


    @analytics
    def plot_neurotransmitter(
        self,
        axes,
        **kwargs
    ) -> None:
        """
        Draw a plot of neurotransmitter dynamics on `axes`.

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
        y = self.monitor['neurotransmitter'].reshape(self.monitor['neurotransmitter'].shape[0],-1)
        time_range = (0, y.shape[0])
        x = torch.arange(*time_range)*self.dt
        population_alpha = 1/y.shape[1]
        aggregated = y.mean(axis=1)
        axes.plot(x, aggregated, color='cyan', **kwargs)
        axes.plot(x, y, alpha=population_alpha, color='cyan')
        axes.set_ylabel('neurotransmitter concentration')
        axes.set_xlabel('time (ms)')
        axes.set_xlim(time_range)