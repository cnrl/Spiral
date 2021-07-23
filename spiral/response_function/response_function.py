"""
"""


import torch
from typing import Union, Iterable
from typeguard import typechecked
from constant_properties_protector import CPP
from construction_requirements_integrator import CRI, construction_required
from spiral.analysis import Analyzer, analysis_point, analytics




@typechecked
class ResponseFunction(torch.nn.Module, CRI):
    def __init__(
        self,
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        torch.nn.Module.__init__(self)
        CPP.protect(self, 'shape')
        CPP.protect(self, 'dt')
        CRI.__init__(
            self,
            shape=shape,
            dt=dt,
            ignore_overwrite_error=True,
            construction_permission=construction_permission,
        )


    def __construct__(
        self,
        shape: Iterable[int],
        dt: Union[float, torch.Tensor],
    ) -> None:
        self._shape = shape
        self.register_buffer("_dt", torch.as_tensor(dt))


    @construction_required
    def __call__(
        self,
        action_potential: torch.Tensor,
    ) -> torch.Tensor:
        return action_potential.float()


    def reset(
        self
    ) -> None:
        pass




@typechecked
class CompositeResponseFunction(ResponseFunction): #In order
    def __init__(
        self,
        response_functions: Iterable[ResponseFunction],
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            shape=shape,
            dt=dt,
            construction_permission=False,
        )
        self.response_functions = response_functions
        self.set_construction_permission(construction_permission)


    def __construct__(
        self,
        shape: Iterable[int],
        dt: Union[float, torch.Tensor],
    ) -> None:
        super().__construct__(
            shape=shape,
            dt=dt,
        )
        for response_function in self.response_functions:
            response_function.meet_requirement(shape=shape)
            response_function.meet_requirement(dt=dt)


    @construction_required
    def __call__(
        self,
        action_potential: torch.Tensor,
    ) -> torch.Tensor:
        for response_function in self.response_functions:
            action_potential = response_function(action_potential=action_potential)
        return action_potential


    def reset(
        self
    ) -> None:
        for response_function in self.response_functions:
            response_function.reset()




@typechecked
class ScalingResponseFunction(ResponseFunction):
    def __init__(
        self,
        scale: Union[float, torch.Tensor],
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            shape=shape,
            dt=dt,
            construction_permission=construction_permission,
        )
        self.register_buffer("scale", torch.as_tensor(scale))


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
        tau: Union[float, torch.Tensor] = 15.,
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None,
        analyzable: bool = False,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            shape=shape,
            dt=dt,
            construction_permission=construction_permission,
        )
        self.register_buffer("tau", torch.as_tensor(tau))
        CPP.protect(self, 'response')
        Analyzer.__init__(self, analyzable)
        Analyzer.scout(self, state_variables=['response'])


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
    @analysis_point
    def __call__(
        self,
        action_potential: torch.Tensor,
    ) -> torch.Tensor:
        self._response += action_potential.float() - self.response * self.dt / self.tau
        return self.response


    @construction_required
    def reset(
        self,
    ) -> None:
        self._response.zero_()
        if self.analyzable:
            self.monitor.reset()


    @analytics
    def plot_response(
        self,
        axes,
        **kwargs
    ) -> None:
        """
        Draw a plot of response value on `axes`.

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
        y = self.monitor['response'].reshape(self.monitor['response'].shape[0],-1)
        time_range = (0, y.shape[0])
        x = torch.arange(*time_range)*self.dt
        population_alpha = 1/y.shape[1]
        aggregated = y.mean(axis=1)
        axes.plot(x, aggregated, color='black', **kwargs)
        axes.plot(x, y, alpha=population_alpha, color='black')
        axes.set_ylabel('response')
        axes.set_xlabel('time (ms)')
        axes.set_xlim(time_range)




@typechecked
class FlatResponseFunction(ResponseFunction):
    def __init__(
        self,
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            shape=shape,
            dt=dt,
            construction_permission=construction_permission,
        )
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
        self._response += action_potential
        return self.response


    @construction_required
    def reset(
        self,
    ) -> None:
        self._response.zero_()




@typechecked
class LimitedFlatResponseFunction(ResponseFunction):
    def __init__(
        self,
        duration: Union[float, torch.Tensor] = 10.,
        shape: Iterable[int] = None,
        dt: Union[float, torch.Tensor] = None,
        construction_permission: bool = True,
    ) -> None:
        super().__init__(
            shape=shape,
            dt=dt,
            construction_permission=False,
        )
        self.add_to_construction_requirements(duration=duration)
        CPP.protect(self, 'action_potential_history')
        CPP.protect(self, 'duration')
        self.set_construction_permission(construction_permission)


    def __construct__(
        self,
        shape: Iterable[int],
        dt: Union[float, torch.Tensor],
        duration: Union[float, torch.Tensor],
    ) -> None:
        super().__construct__(
            shape=shape,
            dt=dt,
        )
        self.register_buffer("_duration", (torch.as_tensor(duration)//self.dt).type(torch.int64))
        if len(self.duration.shape)!=0 and self.duration.shape!=self.shape:
            raise Exception(f"Wrong shape for response limited flat function duration. Expected {self.shape} or a single value but got {self.duration.shape}")
        history_length = self.duration.max()+1
        self.register_buffer("_action_potential_history", torch.zeros((history_length,*self.shape)))


    @construction_required
    def __call__(
        self,
        action_potential: torch.Tensor,
    ) -> torch.Tensor:
        self._action_potential_history = torch.cat([action_potential.unsqueeze(0), self.action_potential_history])
        self._action_potential_history = self.action_potential_history.scatter(
            dim=0, index=self.duration.unsqueeze(0)+1, src=torch.zeros_like(self.duration).unsqueeze(0)
        )
        self._action_potential_history = self.action_potential_history[:-1]
        return self.action_potential_history.sum(axis=0)


    @construction_required
    def reset(
        self,
    ) -> None:
        self._action_potential_history.zero_()