"""

"""

from abc import abstractmethod
from abstract_object_decorator import AOD
from typing import Union, Iterable, Callable
import torch
from .interneuron_soma import InterneuronSoma




class IntegrateAndFireSoma(InterneuronSoma):
    def __init__(
        self,
        name: str,
        tau: Union[float, torch.Tensor] = 20.,  #ms
        R: Union[float, torch.Tensor] = 1., #Ohm
        resting_potential: Union[float, torch.Tensor] = -70.6, #mV
        firing_threshold: Union[float, torch.Tensor] = -40., #mV
        **kwargs
    ) -> None:
        super().__init__(
            name=name,
            **kwargs
        )
        self.register_buffer("tau", torch.tensor(tau))
        self.register_buffer("R", torch.tensor(R))
        self.register_buffer("resting_potential", torch.tensor(resting_potential))
        self.register_buffer("firing_threshold", torch.tensor(firing_threshold))
        self.protect_properties(['potential'])


    def __construct__(self, shape: Iterable[int], **kwargs) -> None:
        super().__construct__(shape=shape, **kwargs)
        self.register_buffer("_potential", torch.zeros(self.shape))
        self._potential += self.resting_potential


    def _repolarization(self):
        self._potential *= ~self.spike
        self._potential += self.spike*self.resting_potential


    def _update_potential(self, current: torch.Tensor) -> None:
        self._potential += self.R * current * self.dt / self.tau


    def _process(self, **kwargs) -> None:
        self._repolarization()
        self._update_potential()


    def _fire_axon_hillock(self, **kwargs) -> None:
        self._spike = (self.potential > self.firing_threshold)
        super()._fire_axon_hillock(**kwargs)


    def reset(self) -> None:
        self._potential.zero_()
        self._potential += self.u_rest
        super().reset()




class LeakyMembrane(AOD, IntegrateAndFireSoma):
    def __init__(
        self,
        obj: IntegrateAndFireSoma,
    ) -> None:
        AOD.__init__(self, obj=obj)


    def __compute_leakage(self) -> torch.Tensor:
        return ((self.potential-self.resting_potential) * self.dt / self.tau)


    def __leak(self, leakage) -> None:
        self._potential -= leakage


    def _update_potential(self, **kwargs) -> None:
        leakage = self.__compute_leakage()
        self.obj._update_potential(**kwargs)
        self.__leak(leakage)




class ExponentialDepolaristicMembrane(AOD, IntegrateAndFireSoma):
    def __init__(
        self,
        obj: IntegrateAndFireSoma,
        sharpness: Union[float, torch.Tensor] = 2.,
        depolarization_threshold: Union[float, torch.Tensor] = -50.4, #mV
    ) -> None:
        AOD.__init__(self, obj=obj)
        self.register_buffer("sharpness", torch.tensor(sharpness))
        self.register_buffer("depolarization_threshold", torch.tensor(depolarization_threshold))


    def __compute_depolarisation(self) -> torch.Tensor:
        return self.sharpness * torch.exp((self.potential-self.depolarization_threshold)/self.sharpness) * self.dt / self.tau


    def __depolarize(self, depolarisation) -> None:
        self._potential += depolarisation


    def _update_potential(self, **kwargs) -> None:
        depolarisation = self.__compute_depolarisation()
        self.obj._update_potential(**kwargs)
        self.__depolarize(depolarisation)




class AdaptiveMembrane(IntegrateAndFireSoma, AOD):
    def __init__(
        self,
        obj: IntegrateAndFireSoma,
        subthreshold_adaptation: Union[float, torch.Tensor] = 4.,
        spike_triggered_adaptation: Union[float, torch.Tensor] = .0805,
        tau_adaptation: Union[float, torch.Tensor] = 144.,
    ) -> None:
        AOD.__init__(self, obj=obj)
        self.register_buffer("subthreshold_adaptation", torch.tensor(subthreshold_adaptation))
        self.register_buffer("spike_triggered_adaptation", torch.tensor(spike_triggered_adaptation))
        self.register_buffer("tau_adaptation", torch.tensor(tau_adaptation))
        self.obj.protect_properties(['adaptation_current'])


    def __construct__(self, shape: Iterable[int], **kwargs) -> None:
        self.obj.__construct__(shape=shape, **kwargs)
        self.register_buffer("_adaptation_current", torch.zeros(self.shape))


    def __update_adaptation_current(self) -> None:
        self._adaptation_current += (
            self.subthreshold_adaptation*(self.potential-self.resting_potential)\
          - self.adaptation_current\
          + self.spike_triggered_adaptation*self.tau_adaptation*self.spike
        ) * self.dt / self.tau_adaptation


    def __adapt(self) -> None:
        self._potential -= self.R*self.adaptation_current


    def _update_potential(self, **kwargs) -> None:
        self.obj._update_potential(**kwargs)
        self.__adapt()


    def _process(self, **kwargs) -> None:
        self.obj._process(**kwargs)
        self.__update_adaptation_current()


    def reset(self) -> None:
        self._adaptation_current.zero_()
        self.obj.reset()
