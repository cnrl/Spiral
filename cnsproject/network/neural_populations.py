"""
Module for neuronal dynamics and populations.
"""

from functools import reduce
from abc import abstractmethod
from operator import mul
from typing import Union, Iterable

import torch


class NeuralPopulation(torch.nn.Module):
    """
    Base class for implementing neural populations.

    Make sure to implement the abstract methods in your child class. Note that this template\
    will give you homogeneous neural populations in terms of excitations and inhibitions. You\
    can modify this by removing `is_inhibitory` and adding another attribute which defines the\
    percentage of inhibitory/excitatory neurons.

    Arguments
    ---------
    shape : Iterable of int
        Define the topology of neurons in the population.
    spike_trace : bool, Optional
        Specify whether to record spike traces. The default is True.
    additive_spike_trace : bool, Optional
        Specify whether to record spike traces additively. The default is True.
    tau_s : float or torch.Tensor, Optional
        Time constant of spike trace decay. The default is 15.0.
    trace_scale : float or torch.Tensor, Optional
        The scaling factor of spike traces. The default is 1.0.
    is_inhibitory : False, Optional
        Whether the neurons are inhibitory or excitatory. The default is False.
    learning : bool, Optional
        Define the training mode. The default is True.

    """

    def __init__(
        self,
        shape: Iterable[int],
        spike_trace: bool = True,
        additive_spike_trace: bool = True,
        tau_s: Union[float, torch.Tensor] = 15.,
        trace_scale: Union[float, torch.Tensor] = 1.,
        is_inhibitory: bool = False,
        learning: bool = True,
        **kwargs
    ) -> None:
        super().__init__()

        self.shape = shape
        self.n = reduce(mul, self.shape)
        self.spike_trace = spike_trace
        self.additive_spike_trace = additive_spike_trace

        if self.spike_trace:
            self.register_buffer("traces", torch.zeros(*self.shape))
            self.register_buffer("tau_s", torch.tensor(tau_s))

            if self.additive_spike_trace:
                self.register_buffer("trace_scale", torch.tensor(trace_scale))

            self.register_buffer("trace_decay", torch.empty_like(self.tau_s))

        self.is_inhibitory = is_inhibitory
        self.learning = learning

        self.register_buffer("s", torch.zeros(*self.shape, dtype=torch.bool))
        self.dt = None

    @abstractmethod
    def forward(self, traces: torch.Tensor) -> None:
        """
        Simulate the neural population for a single step.

        Parameters
        ----------
        traces : torch.Tensor
            Input spike trace.

        Returns
        -------
        None

        """
        if self.spike_trace:
            self.traces *= self.trace_decay

            if self.additive_spike_trace:
                self.traces += self.trace_scale * self.s.float()
            else:
                self.traces.masked_fill_(self.s, 1)

    @abstractmethod
    def compute_potential(self) -> None:
        """
        Compute the potential of neurons in the population.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def compute_spike(self) -> None:
        """
        Compute the spike tensor.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        Refractor and reset the neurons.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def compute_decay(self) -> None:
        """
        Set the decays.

        Returns
        -------
        None

        """
        self.dt = torch.tensor(self.dt)

        if self.spike_trace:
            self.trace_decay = torch.exp(-self.dt/self.tau_s)

    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        self.s.zero_()

        if self.spike_trace:
            self.traces.zero_()

    def train(self, mode: bool = True) -> "NeuralPopulation":
        """
        Set the population's training mode.

        Parameters
        ----------
        mode : bool, optional
            Mode of training. `True` turns on the training while `False` turns\
            it off. The default is True.

        Returns
        -------
        NeuralPopulation

        """
        self.learning = mode
        return super().train(mode)


class InputPopulation(NeuralPopulation):
    """
    Neural population for user-defined spike pattern.

    This class is implemented for future usage. Extend it if needed.

    Arguments
    ---------
    shape : Iterable of int
        Define the topology of neurons in the population.
    spike_trace : bool, Optional
        Specify whether to record spike traces. The default is True.
    additive_spike_trace : bool, Optional
        Specify whether to record spike traces additively. The default is True.
    tau_s : float or torch.Tensor, Optional
        Time constant of spike trace decay. The default is 15.0.
    trace_scale : float or torch.Tensor, Optional
        The scaling factor of spike traces. The default is 1.0.
    learning : bool, Optional
        Define the training mode. The default is True.

    """

    def __init__(
        self,
        shape: Iterable[int],
        spike_trace: bool = True,
        additive_spike_trace: bool = True,
        tau_s: Union[float, torch.Tensor] = 10.,
        trace_scale: Union[float, torch.Tensor] = 1.,
        learning: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            learning=learning,
        )

    def forward(self, traces: torch.Tensor) -> None:
        """
        Simulate the neural population for a single step.

        Parameters
        ----------
        traces : torch.Tensor
            Input spike trace.

        Returns
        -------
        None

        """
        self.s = traces

        super().forward(traces)

    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        super().reset_state_variables()


class LIFPopulation(NeuralPopulation):
    """
    Layer of Leaky Integrate and Fire neurons.

    Implement LIF neural dynamics(Parameters of the model must be modifiable).\
    Follow the template structure of NeuralPopulation class for consistency.
    """

    def __init__(
        self,
        shape: Iterable[int],
        spike_trace: bool = True,
        additive_spike_trace: bool = True,
        tau_s: Union[float, torch.Tensor] = 10.,
        trace_scale: Union[float, torch.Tensor] = 1.,
        is_inhibitory: bool = False,
        learning: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
        )

        """
        TODO.

        1. Add the required parameters.
        2. Fill the body accordingly.
        """

    def forward(self, traces: torch.Tensor) -> None:
        """
        TODO.

        1. Make use of other methods to fill the body. This is the main method\
           responsible for one step of neuron simulation.
        2. You might need to call the method from parent class.
        """
        pass

    def compute_potential(self) -> None:
        """
        TODO.

        Implement the neural dynamics for computing the potential of LIF\
        neurons. The method can either make changes to attributes directly or\
        return the result for further use.
        """
        pass

    def compute_spike(self) -> None:
        """
        TODO.

        Implement the spike condition. The method can either make changes to\
        attributes directly or return the result for further use.
        """
        pass

    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        TODO.

        Implement the refractory and reset conditions. The method can either\
        make changes to attributes directly or return the computed value for\
        further use.
        """
        pass

    @abstractmethod
    def compute_decay(self) -> None:
        """
        TODO.

        Implement the dynamics of decays. You might need to call the method from
        parent class.
        """
        pass


class ELIFPopulation(NeuralPopulation):
    """
    Layer of Exponential Leaky Integrate and Fire neurons.

    Implement ELIF neural dynamics(Parameters of the model must be modifiable).\
    Follow the template structure of NeuralPopulation class for consistency.

    Note: You can use LIFPopulation as parent class as well.
    """

    def __init__(
        self,
        shape: Iterable[int],
        spike_trace: bool = True,
        additive_spike_trace: bool = True,
        tau_s: Union[float, torch.Tensor] = 10.,
        trace_scale: Union[float, torch.Tensor] = 1.,
        is_inhibitory: bool = False,
        learning: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
        )

        """
        TODO.

        1. Add the required parameters.
        2. Fill the body accordingly.
        """

    def forward(self, traces: torch.Tensor) -> None:
        """
        TODO.

        1. Make use of other methods to fill the body. This is the main method\
           responsible for one step of neuron simulation.
        2. You might need to call the method from parent class.
        """
        pass

    def compute_potential(self) -> None:
        """
        TODO.

        Implement the neural dynamics for computing the potential of ELIF\
        neurons. The method can either make changes to attributes directly or\
        return the result for further use.
        """
        pass

    def compute_spike(self) -> None:
        """
        TODO.

        Implement the spike condition. The method can either make changes to
        attributes directly or return the result for further use.
        """
        pass

    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        TODO.

        Implement the refractory and reset conditions. The method can either\
        make changes to attributes directly or return the computed value for\
        further use.
        """
        pass

    @abstractmethod
    def compute_decay(self) -> None:
        """
        TODO.

        Implement the dynamics of decays. You might need to call the method from
        parent class.
        """
        pass


class AELIFPopulation(NeuralPopulation):
    """
    Layer of Adaptive Exponential Leaky Integrate and Fire neurons.

    Implement adaptive ELIF neural dynamics(Parameters of the model must be\
    modifiable). Follow the template structure of NeuralPopulation class for\
    consistency.

    Note: You can use ELIFPopulation as parent class as well.
    """

    def __init__(
        self,
        shape: Iterable[int],
        spike_trace: bool = True,
        additive_spike_trace: bool = True,
        tau_s: Union[float, torch.Tensor] = 10.,
        trace_scale: Union[float, torch.Tensor] = 1.,
        is_inhibitory: bool = False,
        learning: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            shape=shape,
            spike_trace=spike_trace,
            additive_spike_trace=additive_spike_trace,
            tau_s=tau_s,
            trace_scale=trace_scale,
            is_inhibitory=is_inhibitory,
            learning=learning,
        )

        """
        TODO.

        1. Add the required parameters.
        2. Fill the body accordingly.
        """

    def forward(self, traces: torch.Tensor) -> None:
        """
        TODO.

        1. Make use of other methods to fill the body. This is the main method\
           responsible for one step of neuron simulation.
        2. You might need to call the method from parent class.
        """
        pass

    def compute_potential(self) -> None:
        """
        TODO.

        Implement the neural dynamics for computing the potential of adaptive\
        ELIF neurons. The method can either make changes to attributes directly\
        or return the result for further use.
        """
        pass

    def compute_spike(self) -> None:
        """
        TODO.

        Implement the spike condition. The method can either make changes to\
        attributes directly or return the result for further use.
        """
        pass

    @abstractmethod
    def refractory_and_reset(self) -> None:
        """
        TODO.

        Implement the refractory and reset conditions. The method can either\
        make changes to attributes directly or return the computed value for\
        further use.
        """
        pass

    @abstractmethod
    def compute_decay(self) -> None:
        """
        TODO.

        Implement the dynamics of decays. You might need to call the method from
        parent class.
        """
        pass
