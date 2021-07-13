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
        **args
    ) -> None:
        super().__init__(
            name=name,
            **args
        )
        self.register_buffer("tau", torch.tensor(tau))
        self.register_buffer("R", torch.tensor(R))
        self.register_buffer("resting_potential", torch.tensor(resting_potential))
        self.register_buffer("firing_threshold", torch.tensor(firing_threshold))


    def __construct__(self, shape: Iterable[int], **args) -> None:
        super().__construct__(shape=shape, **args)
        self.register_buffer("potential", torch.zeros(self.shape))
        self.potential += self.resting_potential


    def repolarization(self):
        self.potential *= ~self.spike
        self.potential += self.spike*self.resting_potential


    def update_potential(self, current: torch.Tensor) -> None:
        self.potential += self.R * current * self.dt / self.tau


    def process(self, **args) -> None:
        self.repolarization()
        self.update_potential()


    def fire_axon_hillock(self, **args) -> None:
        self.spike = (self.potential > self.firing_threshold)
        super().fire_axon_hillock(**args)


    def reset(self) -> None:
        self.potential.zero_()
        self.potential += self.u_rest
        super().reset()




class LeakyMembrane(IntegrateAndFireSoma, AOD):
    def __init__(
        self,
        obj: IntegrateAndFireSoma,
    ) -> None:
        AOD.__init__(self, obj=obj)


    def compute_leakage(self) -> torch.Tensor:
        return ((self.potential-self.resting_potential) * self.dt / self.tau)


    def leak(self, leakage) -> None:
        self.potential -= leakage


    def update_potential(self, **args) -> None:
        leakage = self.compute_leakage()
        self.obj.update_potential(**args)
        self.leak(leakage)




class ExponentialDepolaristicMembrane(IntegrateAndFireSoma, AOD):
    def __init__(
        self,
        obj: IntegrateAndFireSoma,
        sharpness: Union[float, torch.Tensor] = 2.,
        depolarization_threshold: Union[float, torch.Tensor] = -50.4, #mV
    ) -> None:
        AOD.__init__(self, obj=obj)
        self.register_buffer("sharpness", torch.tensor(sharpness))
        self.register_buffer("depolarization_threshold", torch.tensor(depolarization_threshold))


    def compute_depolarisation(self) -> torch.Tensor:
        depolarisation = self.sharpness * torch.exp((self.potential-self.depolarization_threshold)/self.sharpness) * self.dt / self.tau


    def depolarize(self, depolarisation) -> None:
        self.potential += depolarisation


    def update_potential(self, **args) -> None:
        depolarisation = self.compute_depolarisation()
        self.obj.update_potential(**args)
        self.depolarize(depolarisation)




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


    def __construct__(self, shape: Iterable[int], **args) -> None:
        self.obj.__construct__(shape=shape, **args)
        self.register_buffer("adaptation_current", torch.zeros(self.shape))


    def update_adaptation_current(self) -> None:
        self.adaptation_current += (
            self.subthreshold_adaptation*(self.potential-self.resting_potential)\
          - self.adaptation_current\
          + self.spike_triggered_adaptation*self.tau_adaptation*self.spike
        ) * self.dt / self.tau_adaptation


    def adapt(self) -> None:
        self.potential -= self.R*self.adaptation_current


    def update_potential(self, **args) -> None:
        self.obj.update_potential(**args)
        self.adapt()


    def process(self, **args) -> None:
        self.obj.process(**args)
        self.update_adaptation_current()


    def reset(self) -> None:
        self.adaptation_current.zero_()
        self.obj.reset()
