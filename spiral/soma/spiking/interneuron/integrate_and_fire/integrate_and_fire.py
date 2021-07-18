"""

"""

from abstract_additive_class import AAD
from typing import Union, Iterable
import torch
from constant_properties_protector import CPP
from spiral.analysis import Analyzer, analytics
from ..interneuron_spiking_soma import InterneuronSpikingSoma




class IntegrateAndFireSoma(InterneuronSpikingSoma):
    def __init__(
        self,
        name: str,
        shape: Iterable[int] = None,
        tau: Union[float, torch.Tensor] = 20.,  #ms
        R: Union[float, torch.Tensor] = 1., #Ohm
        resting_potential: Union[float, torch.Tensor] = -70.6, #mV
        firing_threshold: Union[float, torch.Tensor] = -40., #mV
        dt: Union[float, torch.Tensor] = None,
        analyzable: bool = False,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            shape=shape,
            dt=dt,
            analyzable=analyzable,
            construction_permission=False,
        )
        self.register_buffer("tau", torch.tensor(tau))
        self.register_buffer("R", torch.tensor(R))
        self.register_buffer("resting_potential", torch.tensor(resting_potential))
        self.register_buffer("firing_threshold", torch.tensor(firing_threshold))
        CPP.protect(self, 'potential')
        self.set_construction_permission(construction_permission)
        Analyzer.scout(self, state_variables=['potential'])


    def __construct__(
        self,
        shape: Iterable[int],
        dt: Union[float, torch.Tensor],
    ) -> None:
        super().__construct__(
            shape=shape,
            dt=dt
        )
        self.register_buffer("_potential", torch.zeros(self.shape))
        self._potential += self.resting_potential


    def _repolarization(
        self,
    ) -> None:
        self._potential *= ~self.spike
        self._potential += self.spike*self.resting_potential


    def _update_potential(
        self,
        current: torch.Tensor,
    ) -> None:
        self._potential += self.R * current * self.dt / self.tau


    def _process(
        self,
        inputs: torch.Tensor,
    ) -> None:
        self._repolarization()
        self._update_potential(current=inputs)


    def _fire_axon_hillock(
        self,
        clamps: torch.Tensor = torch.tensor(False),
        unclamps: torch.Tensor = torch.tensor(False),
    ) -> None:
        self._spike = (self.potential > self.firing_threshold)
        super()._fire_axon_hillock(clamps=clamps, unclamps=unclamps)


    def reset(
        self,
    ) -> None:
        self._potential.zero_()
        self._potential += self.resting_potential
        super().reset()


    @analytics
    def plot_potential(
        self,
        axes,
        color='green',
        alpha: float = 1.,
        label='potential',
        **kwargs
    ):
        y = self.monitor['potential'].reshape(self.monitor['potential'].shape[0],-1)
        time_range = (0, y.shape[0])
        x = torch.arange(*time_range)*self.dt
        population_alpha = alpha/y.shape[1]
        aggregated = y.mean(axis=1)
        axes.plot(x, aggregated, alpha=alpha, color=color, label=label, **kwargs)
        axes.plot(x, y, alpha=population_alpha, color=color)
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




# Covers a IntegrateAndFireSoma core class
class LeakyMembrane(AAD):
    def __compute_leakage(self) -> torch.Tensor:
        return ((self.potential-self.resting_potential) * self.dt / self.tau)


    def __leak(
        self,
        leakage
    ) -> None:
        self._potential -= leakage


    def _update_potential(
        self,
        current: torch.Tensor
    ) -> None:
        leakage = self.__compute_leakage()
        self.__core._update_potential(self, current=current)
        self.__leak(leakage)




# Covers a IntegrateAndFireSoma core class
class ExponentialDepolaristicMembrane(AAD):
    def __post_init__(
        self,
        sharpness: Union[float, torch.Tensor] = 2.,
        depolarization_threshold: Union[float, torch.Tensor] = -50.4, #mV
    ) -> None:
        self.register_buffer("sharpness", torch.tensor(sharpness))
        self.register_buffer("depolarization_threshold", torch.tensor(depolarization_threshold))


    def __compute_depolarisation(
        self,
    ) -> torch.Tensor:
        return self.sharpness * torch.exp((self.potential-self.depolarization_threshold)/self.sharpness) * self.dt / self.tau


    def __depolarize(
        self,
        depolarisation,
    ) -> None:
        self._potential += depolarisation


    def _update_potential(
        self,
        current: torch.Tensor,
    ) -> None:
        depolarisation = self.__compute_depolarisation()
        self.__core._update_potential(self, current=current)
        self.__depolarize(depolarisation)




# Covers a IntegrateAndFireSoma core class
class AdaptiveMembrane(AAD):
    def __post_init__(
        self,
        subthreshold_adaptation: Union[float, torch.Tensor] = 4.,
        spike_triggered_adaptation: Union[float, torch.Tensor] = .0805,
        tau_adaptation: Union[float, torch.Tensor] = 144.,
    ) -> None:
        self.register_buffer("subthreshold_adaptation", torch.tensor(subthreshold_adaptation))
        self.register_buffer("spike_triggered_adaptation", torch.tensor(spike_triggered_adaptation))
        self.register_buffer("tau_adaptation", torch.tensor(tau_adaptation))
        CPP.protect(self, 'adaptation_current')
        Analyzer.scout(self, state_variables=['adaptation_current'])


    def __construct__(
        self,
        shape: Iterable[int],
        dt: Union[float, torch.Tensor],
    ) -> None:
        self.__core.__construct__(
            self,
            shape=shape,
            dt=dt
        )
        self.register_buffer("_adaptation_current", torch.zeros(self.shape))


    def __update_adaptation_current(
        self,
    ) -> None:
        self._adaptation_current += (
            self.subthreshold_adaptation*(self.potential-self.resting_potential)\
          - self.adaptation_current\
          + self.spike_triggered_adaptation*self.tau_adaptation*self.spike
        ) * self.dt / self.tau_adaptation


    def __adapt(
        self,
    ) -> None:
        self._potential -= self.R*self.adaptation_current


    def _update_potential(
        self,
        current: torch.Tensor,
    ) -> None:
        self.__core._update_potential(self, current=current)
        self.__adapt()


    def _process(
        self,
        inputs: torch.Tensor,
    ) -> None:
        self.__core._process(self, inputs=inputs)
        self.__update_adaptation_current()


    def reset(
        self,
    ) -> None:
        self._adaptation_current.zero_()
        self.__core.reset(self)


    @analytics
    def plot_adaptation_current(
        self,
        axes,
        color='red',
        alpha: float = 1.,
        **kwargs
    ):
        y = self.monitor['adaptation_current'].reshape(self.monitor['adaptation_current'].shape[0],-1)
        time_range = (0, y.shape[0])
        x = torch.arange(*time_range)*self.dt
        population_alpha = alpha/y.shape[1]
        aggregated = y.mean(axis=1)
        axes.plot(x, aggregated, alpha=alpha, color=color, **kwargs)
        axes.plot(x, y, alpha=population_alpha, color=color)
        axes.set_ylabel('adaptation current (mA)')
        axes.set_xlabel('time (ms)')
        axes.set_xlim(time_range)