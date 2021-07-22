"""
The integrate & fire neuron model describes the membrane potential of a neuron in terms\
of the synaptic inputs and the injected current that it receives.\
An action potential (spike) is generated when the membrane potential reaches a threshold.
"""


from add_on_class import AOC
from typing import Union, Iterable
import torch
from constant_properties_protector import CPP
from spiral.analysis import Analyzer, analytics
from typeguard import typechecked
from construction_requirements_integrator import construction_required
from ..spiking_soma import SpikingSoma




@typechecked
class IntegrateAndFireSoma(SpikingSoma):
    """
    Class presenting integrate & fire neuron model is a spiking soma.

    This module keeps and updates dynamic of membrane potential\
    and compute spikes (action potentials) based on the membrane potential.\
    As it is inherited from spiking soma, it will propagates spikes on its axons as output signals.
    In each time step, when `progress()` is called, the sum of currents from\
    dendrites or direct inputs will be calculated in milliamperes by `integrate_inputs()`.
    Then `process()` will update membrane potential.\
    In this process, repolarization will be applied by `repolarization()` first.\
    After that, the effect of the inputs will be considered by `update_potential()`.\
    It will be the end of `process()`.\
    Then action potentials will be calculated by `fire_axon_hillock()` and\
    will be propagated by `propagate_spike()`.

    Properties
    ----------
    name : str, Protected
        The name to be uniquely accessible in Spiral network.\
        Read more about protected properties in constant-properties-protector package documentation.
    shape: Iterable of int, Protected
        The topology of somas in the population.\
        Read more about protected properties in constant-properties-protector package documentation.
    batch : int, Protected
        Determines the batch size.\
        Read more about protected properties in constant-properties-protector package documentation.
    spike: torch.Tensor[bool], Protected
        Indicates which neurons are firing.\
        Read more about protected properties in constant-properties-protector package documentation.
    potential: torch.Tensor, Protected
        The membrane potentials in millivolts.\
        Read more about protected properties in constant-properties-protector package documentation.
    tau: torch.Tensor
        The time constant of membrane potential dynamics in milliseconds.
    R : torch.Tensor
        The membrane electrical resistance in ohms.
    resting_potential : torch.Tensor
        The resting potential of the membrane in millivolts.
    firing_threshold : torch.Tensor
        The firing threshold for membrane potential in millivolts.
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
    shape : Iterable of int, Construction Requirement
        Defines the topology of somas in the population.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.
        Read more about construction requirement in construction-requirements-integrator package documentation.
    batch : int, Construction Requirement
        Determines the batch size.\
        Read more about construction requirement in construction-requirements-integrator package documentation.
    tau : float or torch.Tensor, Optional, default: 20.0
        The time constant of membrane potential dynamics in milliseconds.
    R : float or torch.Tensor, Optional, default: 1.0
        The membrane electrical resistance in ohms.
    resting_potential : float or torch.Tensor, Optional, default: -70.6
        The resting potential of the membrane in millivolts.
    firing_threshold : float or torch.Tensor, Optional, default: -40.0
        The firing threshold for membrane potential in millivolts.
    dt : float or torch.Tensor, Construction Requirement
        Time step in milliseconds.\
        It is necessary for construction, but you can determine it with a delay after the initial construction and complete the construction process.
        Read more about construction requirement in construction-requirements-integrator package documentation.
    analyzable: bool, Optional, default: False
        If it is `True`, it will record its behavior and provide you with functions for drawing plots.\
        You can also get a recorded sequence of important features through the `monitor` variable it creates.
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
        batch: int = None,
        tau: Union[float, torch.Tensor] = 20.,
        R: Union[float, torch.Tensor] = 1.,
        resting_potential: Union[float, torch.Tensor] = -70.6,
        firing_threshold: Union[float, torch.Tensor] = -40.,
        dt: Union[float, torch.Tensor] = None,
        analyzable: bool = False,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            shape=shape,
            batch=batch,
            dt=dt,
            analyzable=analyzable,
            construction_permission=False,
        )
        self.register_buffer("tau", torch.as_tensor(tau))
        self.register_buffer("R", torch.as_tensor(R))
        self.register_buffer("resting_potential", torch.as_tensor(resting_potential))
        self.register_buffer("firing_threshold", torch.as_tensor(firing_threshold))
        CPP.protect(self, 'potential')
        self.set_construction_permission(construction_permission)
        Analyzer.scout(self, state_variables=['potential'])


    def __construct__(
        self,
        shape: Iterable[int],
        batch: int,
        dt: Union[float, torch.Tensor],
    ) -> None:
        """
        This function is related to the completion of the construction process.\
        Read more about `__construct__` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        shape : Iterable of int
            Defines the topology of somas in the population.
        batch : int
            Determines the batch size.
        dt : float or torch.Tensor
            Time step in milliseconds.
        
        Returns
        -------
        None
        
        """
        super().__construct__(
            shape=shape,
            batch=batch,
            dt=dt
        )
        self.register_buffer("_potential", torch.zeros((self.batch, *self.shape)))
        self._potential += self.resting_potential


    def _repolarization(
        self,
    ) -> None:
        """
        Simulates the repolarization process.\
        Repolarization is the process of returning the membrane potential\
        of neurons that have recently spiked to their own resting potential.

        Returns
        -------
        None
        
        """
        self._potential *= ~self.spike
        self._potential += self.spike*self.resting_potential


    def _update_potential(
        self,
        current: torch.Tensor,
    ) -> None:
        """
        Updates membrane potential based on input currents.
        
        Arguments
        ---------
        current : torch.Tensor
            Input current in milliamperes.

        Returns
        -------
        None
        
        """
        self._potential += self.R * current * self.dt / self.tau


    def _process(
        self,
        inputs: torch.Tensor,
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
        self._repolarization()
        self._update_potential(current=inputs)
        super()._process(inputs=inputs)


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
        self._spike = (self.potential > self.firing_threshold)
        super()._fire_axon_hillock(clamps=clamps, unclamps=unclamps)


    @construction_required
    def reset(
        self,
    ) -> None:
        """
        Refractor and reset the somas and connected organs.
        
        Returns
        -------
        None
        
        """
        self._potential.zero_()
        self._potential += self.resting_potential
        super().reset()


    @analytics
    def plot_potential(
        self,
        axes,
        **kwargs
    ):
        """
        Draw a plot of neuron potentials on `axes`.

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
        y = self.monitor['potential'].reshape(self.monitor['potential'].shape[0],-1)
        time_range = (0, y.shape[0])
        x = torch.arange(*time_range)*self.dt
        population_alpha = 1/y.shape[1]
        aggregated = y.mean(axis=1)
        axes.plot(x, aggregated, color='green', label='potential', **kwargs)
        axes.plot(x, y, alpha=population_alpha, color='green')
        axes.plot(x, torch.ones((len(x), *self.resting_potential.shape))*self.resting_potential,
            'k-.', label='resting potential')
        axes.plot(x, torch.ones((len(x), *self.firing_threshold.shape))*self.firing_threshold,
            'b--', label='firing threshold')
        axes.set_ylabel('potential (mV)')
        axes.set_xlabel('time (ms)')
        axes.set_xlim(time_range)
        diff = (self.firing_threshold-self.resting_potential).max()
        axes.set_ylim((self.resting_potential.min()-diff/2, self.firing_threshold.max()+diff/2))
        axes.legend()




@typechecked
class LeakyMembrane(AOC):
    """
    Add-on class to add leakage to membrane potential dynamics in an integrate & fire neuron model.

    This module adds leakage to membrane potential dynamics in an integrate & fire neuron model.\
    Read more about add-on classes in add-on-class package documentation.\
    Because this is an add-on class, we will only introduce the properties and functions it adds.\
    It will override `update_potential()`.\
    First, it computs membrane potential leakage by `compute_leakage()`.\
    Then will call original `update_potential()`.\
    At the end, it will applies computed leakage on membrane potential by `leak()`.
    """

    def __pre_init__(
        self,
    ) -> None:
        """
        This function will be called before the original `__init__()` call\
        and ensures that the core is an integrate & fire model.

        Returns
        -------
        None
        
        """
        if not issubclass(self.__core, IntegrateAndFireSoma):
            raise Exception("LeakyMembrane can only be added to IntegrateAndFireSoma or IntegrateAndFireSoma subclasses.")


    def __compute_leakage(
        self
    ) -> torch.Tensor:
        """
        Computes membrane potential leakage.

        Returns
        -------
        leakage : torch.Tensor
            The membrane potential leakage.
        
        """
        return ((self.potential-self.resting_potential) * self.dt / self.tau)


    def __leak(
        self,
        leakage: torch.Tensor,
    ) -> None:
        """
        It will applies computed leakage on membrane potential.

        Arguments
        ---------
        leakage : torch.Tensor
            The membrane potential leakage.

        Returns
        -------
        None
        
        """
        self._potential -= leakage


    def _update_potential(
        self,
        current: torch.Tensor
    ) -> None:
        """
        Updates membrane potential based on input currents.
        
        Arguments
        ---------
        current : torch.Tensor
            Input current in milliamperes.

        Returns
        -------
        None
        
        """
        leakage = self.__compute_leakage()
        self.__core._update_potential(self, current=current)
        self.__leak(leakage)




@typechecked
class ExponentialDepolaristicMembrane(AOC):
    """
    Add-on class to add depolarisation to membrane potential dynamics in an integrate & fire neuron model.

    This module adds depolarisation to membrane potential dynamics in an integrate & fire neuron model.\
    Read more about add-on classes in add-on-class package documentation.\
    Because this is an add-on class, we will only introduce the properties and functions it adds.\
    It will override `update_potential()`.\
    First, it computs membrane potential depolarisation by `compute_depolarisation()`.\
    Then will call original `update_potential()`.\
    At the end, it will applies computed depolarisation on membrane potential by `depolarize()`.

    Add-On Properties
    -----------------
    sharpness : torch.Tensor
        The sharpness of the depolarisation process.
    depolarization_threshold: torch.Tensor
        The membrane potential threshold of the depolarisation process in millivolts.

    Add-On Arguments
    ----------------
    sharpness : float or torch.Tensor, Optional, default: 2.0
        Determines the sharpness of the depolarisation process.
    depolarization_threshold : float or torch.Tensor, Optional, default: -50.4
        Determines the membrane potential threshold of the depolarisation process in millivolts.
    """

    def __pre_init__(
        self,
    ) -> None:
        """
        This function will be called before the original `__init__()` call\
        and ensures that the core is an integrate & fire model.

        Returns
        -------
        None
        
        """
        if not issubclass(self.__core, IntegrateAndFireSoma):
            raise Exception("ExponentialDepolaristicMembrane can only be added to IntegrateAndFireSoma or IntegrateAndFireSoma subclasses.")


    def __post_init__(
        self,
        sharpness: Union[float, torch.Tensor] = 2.,
        depolarization_threshold: Union[float, torch.Tensor] = -50.4,
    ) -> None:
        """
        This function will be called after the original `__init__()` call\
        initializes add-on properties.

        Arguments
        ---------
        sharpness : float or torch.Tensor, Optional, default: 2.0
            Determines the sharpness of the depolarisation process.
        depolarization_threshold : float or torch.Tensor, Optional, default: -50.4
            Determines the membrane potential threshold of the depolarisation process in millivolts.

        Returns
        -------
        None
        
        """
        self.register_buffer("sharpness", torch.as_tensor(sharpness))
        self.register_buffer("depolarization_threshold", torch.as_tensor(depolarization_threshold))


    def __compute_depolarisation(
        self,
    ) -> torch.Tensor:
        """
        Computes membrane potential depolarisation.

        Returns
        -------
        depolarisation : torch.Tensor
            The membrane potential depolarisation.
        
        """
        return self.sharpness * torch.exp((self.potential-self.depolarization_threshold)/self.sharpness) * self.dt / self.tau


    def __depolarize(
        self,
        depolarisation,
    ) -> None:
        """
        It will applies computed depolarisation on membrane potential.

        Arguments
        ---------
        depolarisation : torch.Tensor
            The membrane potential depolarisation.

        Returns
        -------
        None
        
        """
        self._potential += depolarisation


    def _update_potential(
        self,
        current: torch.Tensor,
    ) -> None:
        """
        Updates membrane potential based on input currents.
        
        Arguments
        ---------
        current : torch.Tensor
            Input current in milliamperes.

        Returns
        -------
        None
        
        """
        depolarisation = self.__compute_depolarisation()
        self.__core._update_potential(self, current=current)
        self.__depolarize(depolarisation)




@typechecked
class AdaptiveMembrane(AOC):
    """
    Add-on class to add adaptation to membrane potential dynamics in an integrate & fire neuron model.

    This module adds adaptation to membrane potential dynamics in an integrate & fire neuron model.\
    Read more about add-on classes in add-on-class package documentation.\
    Because this is an add-on class, we will only introduce the properties and functions it adds.\
    At the end of each `process()` call, it will update adaptation current for the next turn by\
    `update_adaptation_current`.\
    Also, at the end of `update_potential()` call, applies last updated adaptation current on\
    membrane potential by `adapt()`.\
    This module also provides `plot_adaptation_current()`. This function draws adaptation current dynamics.

    Add-On Properties
    -----------------
    subthreshold_adaptation : torch.Tensor
        The degree of adaptability of the membrane to strong steady flow.
    spike_triggered_adaptation: torch.Tensor
        The degree of adaptability of the membrane to frequent action potentials.
    tau_adaptation: torch.Tensor
        The time constant of membrane adaptation current dynamics in milliseconds.
    adaptation_current: torch.Tensor, Protected
        The membrane adaptation current in milliamperes.\
        Read more about protected properties in constant-properties-protector package documentation.

    Add-On Arguments
    ----------------
    subthreshold_adaptation : float or torch.Tensor, Optional, default: 4.0
        Determines the degree of adaptability of the membrane to strong steady flow.
    spike_triggered_adaptation : float or torch.Tensor, Optional, default: 0.0805
        Determines the degree of adaptability of the membrane to frequent action potentials.
    tau_adaptation : float or torch.Tensor, Optional, default: 144.0
        Determines time constant of membrane adaptation current dynamics in milliseconds.
    """

    def __pre_init__(
        self,
    ) -> None:
        """
        This function will be called before the original `__init__()` call\
        and ensures that the core is an integrate & fire model.

        Returns
        -------
        None
        
        """
        if not issubclass(self.__core, IntegrateAndFireSoma):
            raise Exception("AdaptiveMembrane can only be added to IntegrateAndFireSoma or IntegrateAndFireSoma subclasses.")


    def __post_init__(
        self,
        subthreshold_adaptation: Union[float, torch.Tensor] = 4.,
        spike_triggered_adaptation: Union[float, torch.Tensor] = .0805,
        tau_adaptation: Union[float, torch.Tensor] = 144.,
    ) -> None:
        """
        This function will be called after the original `__init__()` call\
        initializes add-on properties.

        Arguments
        ---------
        subthreshold_adaptation : float or torch.Tensor, Optional, default: 4.0
            Determines the degree of adaptability of the membrane to strong steady flow.
        spike_triggered_adaptation : float or torch.Tensor, Optional, default: 0.0805
            Determines the degree of adaptability of the membrane to frequent action potentials.
        tau_adaptation : float or torch.Tensor, Optional, default: 144.0
            Determines time constant of membrane adaptation current dynamics in milliseconds.

        Returns
        -------
        None
        
        """
        self.register_buffer("subthreshold_adaptation", torch.as_tensor(subthreshold_adaptation))
        self.register_buffer("spike_triggered_adaptation", torch.as_tensor(spike_triggered_adaptation))
        self.register_buffer("tau_adaptation", torch.as_tensor(tau_adaptation))
        CPP.protect(self, 'adaptation_current')
        Analyzer.scout(self, state_variables=['adaptation_current'])


    def __construct__(
        self,
        shape: Iterable[int],
        batch: int,
        dt: Union[float, torch.Tensor],
    ) -> None:
        """
        This function is related to the completion of the construction process.\
        Read more about `__construct__` in construction-requirements-integrator package documentation.
        
        Arguments
        ---------
        shape : Iterable of int
            Defines the topology of somas in the population.
        batch : int, Construction Requirement, Optional, default: 1
            Determines the batch size.\
            Will be added to the top of the topology shape.
        dt : float or torch.Tensor
            Time step in milliseconds.
        
        Returns
        -------
        None
        
        """
        self.__core.__construct__(
            self,
            shape=shape,
            batch=batch,
            dt=dt
        )
        self.register_buffer("_adaptation_current", torch.zeros((self.batch, *self.shape)))


    def __update_adaptation_current(
        self,
    ) -> None:
        """
        Updates membrane adaptation current based on membrane potential.
        
        Returns
        -------
        None
        
        """
        self._adaptation_current += (
            self.subthreshold_adaptation*(self.potential-self.resting_potential)\
          - self.adaptation_current\
          + self.spike_triggered_adaptation*self.tau_adaptation*self.spike
        ) * self.dt / self.tau_adaptation


    def __adapt(
        self,
    ) -> None:
        """
        It will applies last updated adaptation current on membrane potential.

        Returns
        -------
        None
        
        """
        self._potential -= self.R*self.adaptation_current


    def _update_potential(
        self,
        current: torch.Tensor,
    ) -> None:
        """
        Updates membrane potential based on input currents.
        
        Arguments
        ---------
        current : torch.Tensor
            Input current in milliamperes.

        Returns
        -------
        None
        
        """
        self.__core._update_potential(self, current=current)
        self.__adapt()


    def _process(
        self,
        inputs: torch.Tensor,
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
        self.__core._process(self, inputs=inputs)
        self.__update_adaptation_current()


    @construction_required
    def reset(
        self,
    ) -> None:
        """
        Refractor and reset the somas and connected organs.
        
        Returns
        -------
        None
        
        """
        self._adaptation_current.zero_()
        self.__core.reset(self)


    @analytics
    def plot_adaptation_current(
        self,
        axes,
        **kwargs
    ):
        """
        Draw a plot of adaptation current on `axes`.

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
        y = self.monitor['adaptation_current'].reshape(self.monitor['adaptation_current'].shape[0],-1)
        time_range = (0, y.shape[0])
        x = torch.arange(*time_range)*self.dt
        population_alpha = 1/y.shape[1]
        aggregated = y.mean(axis=1)
        axes.plot(x, aggregated, color='red', **kwargs)
        axes.plot(x, y, alpha=population_alpha, color='red')
        axes.set_ylabel('adaptation current (mA)')
        axes.set_xlabel('time (ms)')
        axes.set_xlim(time_range)