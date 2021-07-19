"""
"""


from abc import ABC, abstractmethod
from construction_requirements_integrator import CRI, construction_required
from constant_properties_protector import CPP
from typing import Union, Iterable
from typeguard import typechecked
import torch




@typechecked
class ResponseFunction(torch.nn.Module, CRI, ABC):
    def __init__(
        self,
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        CPP.protect(self, 'shape')
        CPP.protect(self, 'dt')
        CRI.__init__(
            self,
            shape=shape,
            dt=dt,
            ignore_overwrite_error=True,
        )


    def __construct__(
        self,
        shape: Iterable[int],
        dt: Union[float, torch.Tensor],
    ) -> None:
        self._shape = shape
        self.register_buffer("_dt", torch.tensor(dt))


    @construction_required
    def __call__(
        self,
        action_potential: torch.Tensor,
    ) -> torch.Tensor:
        return action_potential.float()


    @construction_required
    def reset(
        self
    ) -> None:
        pass




@typechecked
class CompositeResponseFunction(ResponseFunction):
    def __init__(
        self,
        response_functions: Iterable[ResponseFunction],
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None,
    ) -> None:
        super().__init__(
            shape=shape,
            dt=dt,
        )
        self.response_functions = response_functions


    def __construct__(
        self,
        shape: Iterable[int],
        dt: Union[float, torch.Tensor],
    ) -> None:
        super(
            shape=shape,
            dt=dt,
        )
        for response_function in response_functions:
            response_function.meet_requirement(shape=shape)
            response_function.meet_requirement(dt=dt)


    @construction_required
    def __call__(
        self,
        action_potential: torch.Tensor,
    ) -> torch.Tensor:
        for response_function in response_functions:
            action_potential = response_function(action_potential)
        return action_potential


    @construction_required
    def reset(
        self
    ) -> None:
        for response_function in response_functions:
            response_function.reset()




@typechecked
class ScalingResponseFunction(ResponseFunction):
    def __init__(
        self,
        scale: Union[float, torch.Tensor],
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None,
    ) -> None:
        super().__init__(
            shape=shape,
            dt=dt,
        )
        self.register_buffer("scale", torch.tensor(scale))


    @construction_required
    def __call__(
        self,
        action_potential: torch.Tensor,
    ) -> torch.Tensor:
        return action_potential*self.scale




@typechecked
class LeakyResponseFunction(ResponseFunction):
    def __init__(
        self,
        tau: Union[float, torch.Tensor] = 10.,
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None,
    ) -> None:
        super().__init__(
            shape=shape,
            dt=dt,
        )
        self.register_buffer("tau", torch.tensor(tau))
        CPP.protect(self, 'response')


    def __construct__(
        self,
        shape: Iterable[int],
        dt: Union[float, torch.Tensor],
    ) -> None:
        super().__construct__(
            shape=shape,
            dt=dt,
        )
        self.register_buffer("_response", torch.zeros(*self.shape))


    @construction_required
    def __call__(
        self,
        action_potential: torch.Tensor,
    ) -> torch.Tensor:
        self._response += action_potential - self.response * self.dt / self.tau
        return self.response


    @construction_required
    def reset(
        self,
    ) -> None:
        self._response.zero_()