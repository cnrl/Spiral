"""
"""


from abc import ABC, abstractmethod
from construction_requirements_integrator import CRI, construction_required
from constant_properties_protector import CPP
from typing import Union, Iterable
from typeguard import typechecked
import torch




@typechecked
class SynapticPlasticityRate(torch.nn.Module, CRI):
    def __init__(
        self,
        rate: Union[float, torch.Tensor],
        dt: Union[float, torch.Tensor] = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.register_buffer("rate", torch.as_tensor(rate))
        CPP.protect(self, 'dt')
        CRI.__init__(
            self,
            dt=dt,
            ignore_overwrite_error=True,
        )


    def __construct__(
        self,
        dt: Union[float, torch.Tensor],
    ) -> None:
        self.register_buffer("_dt", torch.as_tensor(dt))


    @construction_required
    def __call__(
        self,
        synaptic_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.rate


    def reset(
        self
    ) -> None:
        pass




@typechecked
class WeightDependentRate(SynapticPlasticityRate):
    def __init__(
        self,
        rate: Union[float, torch.Tensor],
        minimum: Union[float, torch.Tensor] = 0.,
        maximum: Union[float, torch.Tensor] = 1.,
        dt: Union[float, torch.Tensor] = None,
    ) -> None:
        super().__init__(
            rate=rate,
            dt=dt
        )
        self.register_buffer("min", torch.as_tensor(minimum))
        self.register_buffer("max", torch.as_tensor(maximum))


    @construction_required
    def __call__(
        self,
        synaptic_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.rate * (self.max-synaptic_weights) * (synaptic_weights-self.min)




@typechecked
class DescendingSynapticPlasticityRate(SynapticPlasticityRate):
    def __init__(
        self,
        rate: Union[float, torch.Tensor],
        tau: Union[float, torch.Tensor] = 1000.,
        dt: Union[float, torch.Tensor] = None,
    ) -> None:
        super().__init__(
            rate=rate,
            dt=dt
        )
        self.register_buffer("tau", torch.as_tensor(tau))
        self.register_buffer("current_rate", self.rate.detach().clone())


    @construction_required
    def __call__(
        self,
        synaptic_weights: torch.Tensor,
    ) -> torch.Tensor:
        learning_rate = self.current_rate.detach().clone()
        self.current_rate -= self.current_rate * self.dt / self.tau
        return learning_rate


    def reset(
        self
    ) -> None:
        self.current_rate = self.rate.detach().clone()