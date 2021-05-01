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
    percentage of inhibitory/excitatory neurons or use a boolean tensor with the same shape as\
    the population, defining which neurons are inhibitory.

    The most important attribute of each neural population is its `shape` which indicates the\
    number and/or architecture of the neurons in it. When there are connected populations, each\
    pre-synaptic population will have an impact on the post-synaptic one in case of spike. This\
    spike might be persistent for some duration of time and with some decaying magnitude. To\
    handle this coincidence, four attributes are defined:
    - `spike_trace` is a boolean indicating whether to record the spike trace in each time step.
    - `additive_spike_trace` would indicate whether to save the accumulated traces up to the\
        current time step.
    - `tau_s` will show the duration by which the spike trace persists by a decaying manner.
    - `trace_scale` is responsible for the scale of each spike at the following time steps.\
        Its value is only considered if `additive_spike_trace` is set to `True`.

    Make sure to call `reset_state_variables` before starting the simulation to allocate\
    and/or reset the state variables such as `s` (spikes tensor) and `traces` (trace of spikes).\
    Also do not forget to set the time resolution (dt) for the simulation.

    Each simulation step is defined in `forward` method. You can use the utility methods (i.e.\
    `compute_potential`, `compute_spike`, `refractory_and_reset`, and `compute_decay`) to break\
    the differential equations into smaller code blocks and call them within `forward`. Make\
    sure to call methods `forward` and `compute_decay` of `NeuralPopulation` in child class\
    methods; As it provides the computation of spike traces (not necessary if you are not\
    considering the traces). The `forward` method can either work with current or spike trace.\
    You can easily work with any of them you wish. When there are connected populations, you\
    might need to consider how to convert the pre-synaptic spikes into current or how to\
    change the `forward` block to support spike traces as input.

    There are some more points to be considered further:
    - Note that parameters of the neuron are not specified in child classes. You have to\
        define them as attributes of the corresponding class (i.e. in __init__) with suitable\
        naming.
    - In case you want to make simulations on `cuda`, make sure to transfer the tensors\
        to the desired device by defining a `device` attribute or handling the issue from\
        upstream code.
    - Almost all variables, parameters, and arguments in this file are tensors with a\
        single value or tensors of the shape equal to population`s shape. No extra\
        dimension for time is needed. The time dimension should be handled in upstream\
        code and/or monitor objects.

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
        is_excitatory: Union[bool, torch.Tensor] = True,
        learning: bool = True,
        dt: float = None,
        **kwargs
    ) -> None:
        super().__init__()

        self.shape = shape
        self.n = reduce(mul, self.shape)
        self.spike_trace = spike_trace
        self.additive_spike_trace = additive_spike_trace
        self.register_buffer("is_excitatory", torch.tensor(is_excitatory))

        if self.spike_trace:
            # You can use `torch.Tensor()` instead of `torch.zeros(*shape)` if `reset_state_variables`
            # is intended to be called before every simulation.
            self.register_buffer("traces", torch.zeros(*self.shape))
            self.register_buffer("tau_s", torch.tensor(tau_s))

            if self.additive_spike_trace:
                self.register_buffer("trace_scale", torch.tensor(trace_scale))

            self.register_buffer("trace_decay", torch.empty_like(self.tau_s))

        self.learning = learning

        # You can use `torch.Tensor()` instead of `torch.zeros(*shape, dtype=torch.bool)` if \
        # `reset_state_variables` is intended to be called before every simulation.
        self.register_buffer("s", torch.zeros(*self.shape, dtype=torch.bool))
        self.set_dt(dt)

    def set_dt(self, dt:float):
        self.dt = torch.tensor(dt) if dt is not None else dt

    @abstractmethod
    def forward(self, I: torch.Tensor) -> None:
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

        self.compute_trace_decay()

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
    def reset(self) -> None:
        """
        Refractor and reset the neurons.

        Returns
        -------
        None

        """
        self.s.zero_()

        if self.spike_trace:
            self.traces.zero_()

    @abstractmethod
    def compute_trace_decay(self) -> None:
        """
        Set the trace_decay.

        Returns
        -------
        None

        """

        if self.spike_trace:
            self.trace_decay = torch.exp(-self.dt/self.tau_s)

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

    def forward(self, I: torch.Tensor) -> None:
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

    def reset(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        super().reset()


class LIFPopulation(NeuralPopulation):
    """
    Layer of Leaky Integrate and Fire neurons.

    Implement LIF neural dynamics(Parameters of the model must be modifiable).\
    Follow the template structure of NeuralPopulation class for consistency.
    """

    def __init__(
        self,
        shape: Iterable[int],
        tau: Union[float, torch.Tensor] = 20.,  #ms
        R: Union[float, torch.Tensor] = 1., #Ohm
        resting_potential: Union[float, torch.Tensor] = -70.6, #mV
        spike_threshold: Union[float, torch.Tensor] = -40., #mV
        **kwargs
    ) -> None:
        super().__init__(
            shape=shape,
            **kwargs
        )
        self.register_buffer("tau", torch.tensor(tau))
        self.register_buffer("R", torch.tensor(R))
        self.register_buffer("u_rest", torch.tensor(resting_potential))
        self.register_buffer("spike_threshold", torch.tensor(spike_threshold))
        self.register_buffer("u", torch.zeros(self.shape))
        self.u += self.u_rest
        self.register_buffer("s", torch.zeros(self.shape, dtype=torch.bool))


    def forward(self, I: torch.Tensor) -> None: # I: mA
        self.compute_potential(I)
        self.compute_spike()
        super().forward(I)

    def compute_potential(self, I: torch.Tensor) -> None:
        self.u -= self.dt/self.tau * ((self.u-self.u_rest) - self.R*I)

    def compute_spike(self) -> None:
        self.s = (self.u > self.spike_threshold)
        self.u *= ~self.s
        self.u += self.s*self.u_rest

    @abstractmethod
    def reset(self) -> None:
        self.u.zero_()
        self.u += self.u_rest
        super().reset()


class ELIFPopulation(LIFPopulation):
    """
    Layer of Exponential Leaky Integrate and Fire neurons.

    Implement ELIF neural dynamics(Parameters of the model must be modifiable).\
    Follow the template structure of NeuralPopulation class for consistency.

    Note: You can use LIFPopulation as parent class as well.
    """

    def __init__(
        self,
        shape: Iterable[int],
        sharpness: Union[float, torch.Tensor] = 2.,
        firing_threshold: Union[float, torch.Tensor] = -50.4, #mV
        spike_threshold: Union[float, torch.Tensor] = -40., #mV
        **kwargs
    ) -> None:
        super().__init__(
            shape=shape,
            spike_threshold=spike_threshold,
            **kwargs
        )
        self.register_buffer("sharpness", torch.tensor(sharpness))
        self.register_buffer("firing_threshold", torch.tensor(firing_threshold))

    def compute_potential(self, I) -> None:
        firing_term = self.sharpness*torch.exp((self.u-self.firing_threshold)/self.sharpness)
        super().compute_potential(I)
        self.u += self.dt/self.tau * firing_term


class AELIFPopulation(ELIFPopulation):
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
        a_w: Union[float, torch.Tensor] = 4.,
        b_w: Union[float, torch.Tensor] = .0805,
        tau_w: Union[float, torch.Tensor] = 144.,
        **kwargs
    ) -> None:
        super().__init__(
            shape=shape,
            **kwargs
        )
        self.register_buffer("a_w", torch.tensor(a_w))
        self.register_buffer("b_w", torch.tensor(b_w))
        self.register_buffer("tau_w", torch.tensor(tau_w))
        self.register_buffer("w", torch.zeros(self.shape))

    def forward(self, I: torch.Tensor) -> None: # I: mA
        super().forward(I)
        self.compute_w()

    def compute_potential(self, I) -> None:
        super().compute_potential(I)
        self.u -= self.R*self.w

    def compute_w(self) -> None:
        self.w += self.dt/self.tau_w * (self.a_w*(self.u-self.u_rest) - self.w + self.b_w*self.tau_w*self.s)

    @abstractmethod
    def reset(self) -> None:
        self.w.zero_()
        super().reset()
