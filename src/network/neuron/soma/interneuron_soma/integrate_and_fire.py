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


    def main_process(self, current: torch.Tensor) -> None:
        self.potential += self.R * current * self.dt / self.tau


    def fire_axon_hillock(self, **args) -> None:
        self.spike = (self.potential > self.firing_threshold)
        super().fire_axon_hillock(**args)


    def progress(self, **args) -> None:
        self.repolarization()
        super().progress(**args)


    def reset(self) -> None:
        self.potential.zero_()
        self.potential += self.u_rest
        super().reset()




class LeakyMembrane(InterneuronSoma, AOD):
    def __init__(
        self,
        obj: InterneuronSoma,
    ) -> None:
        AOD.__init__(self, obj=obj)


    def compute_leakage(self) -> torch.Tensor:
        return (self.potential-self.resting_potential) * self.dt / self.tau


    def leak(self, leakage) -> None:
        self.potential -= leakage


    def main_process(self, **args) -> None:
        leakage = self.compute_leakage()
        self.obj.main_process(**args)
        self.leak(leakage)




class ExponentialDepolaristicMembrane(InterneuronSoma, AOD):
    def __init__(
        self,
        obj: InterneuronSoma,
        sharpness: Union[float, torch.Tensor] = 2.,
        depolarization_threshold: Union[float, torch.Tensor] = -50.4, #mV
    ) -> None:
        AOD.__init__(self, obj=obj)
        self.register_buffer("sharpness", torch.tensor(sharpness))
        self.register_buffer("depolarization_threshold", torch.tensor(depolarization_threshold))


    def depolarize(self) -> None:
        depolarisation = self.sharpness*torch.exp((self.potential-self.depolarization_threshold)/self.sharpness)
        self.potential += self.dt/self.tau * depolarisation


    def main_process(self, **args) -> None:
        self.obj.main_process(**args)
        self.depolarize()




class AdaptiveMembrane(InterneuronSoma, AOD):
    def __init__(
        self,
        obj: InterneuronSoma,
        a_w: Union[float, torch.Tensor] = 4.,
        b_w: Union[float, torch.Tensor] = .0805,
        tau_w: Union[float, torch.Tensor] = 144.,
    ) -> None:
        AOD.__init__(self, obj=obj)
        self.register_buffer("a_w", torch.tensor(a_w))
        self.register_buffer("b_w", torch.tensor(b_w))
        self.register_buffer("tau_w", torch.tensor(tau_w))


    def __construct__(self, shape: Iterable[int], **args) -> None:
        self.obj.__construct__(shape=shape, **args)
        self.register_buffer("adaptivity", torch.zeros(self.shape))


    def update_adaptivity(self) -> None:
        self.adaptivity += self.dt/self.tau_w * (self.a_w*(self.potential-self.resting_potential) - self.adaptivity + self.b_w*self.tau_w*self.spike)


    def adapt(self) -> None:
        self.potential -= self.R*self.adaptivity


    def main_process(self, **args) -> None:
        self.update_adaptivity()
        self.obj.main_process(**args)
        self.adapt()


    def reset(self) -> None:
        self.adaptivity.zero_()
        self.obj.reset()
