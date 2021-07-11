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


    def refactoriness(self):
        self.potential *= ~self.spike
        self.potential += self.spike*self.resting_potential


    def main_process(self, current: torch.Tensor) -> None:
        self.potential += self.R * current * self.dt / self.tau


    def fire_axon_hillock(self, **args) -> None:
        self.spike = (self.potential > self.firing_threshold)
        super().fire_axon_hillock(**args)


    def progress(self, **args) -> None:
        self.refactoriness()
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


    def main_process(self, **args) -> None:
        leaky_term = self.potential-self.resting_potential
        self.obj.main_process(**args)
        self.potential -= leaky_term * self.dt / self.tau




class ExponentialMembrane(InterneuronSoma, AOD):
    def __init__(
        self,
        obj: InterneuronSoma,
        sharpness: Union[float, torch.Tensor] = 2.,
        action_threshold: Union[float, torch.Tensor] = -50.4, #mV
    ) -> None:
        AOD.__init__(self, obj=obj)
        self.register_buffer("sharpness", torch.tensor(sharpness))
        self.register_buffer("action_threshold", torch.tensor(action_threshold))


    def main_process(self, **args) -> None:
        self.obj.main_process(**args)
        action_term = self.sharpness*torch.exp((self.potential-self.action_threshold)/self.sharpness)
        self.potential += self.dt/self.tau * action_term




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
        self.register_buffer("w", torch.zeros(self.shape))


    def compute_w(self) -> None:
        self.w += self.dt/self.tau_w * (self.a_w*(self.potential-self.resting_potential) - self.w + self.b_w*self.tau_w*self.spike)


    def main_process(self, **args) -> None:
        self.compute_w()
        self.obj.main_process(**args)
        self.potential -= self.R*self.w


    def reset(self) -> None:
        self.w.zero_()
        self.obj.reset()
