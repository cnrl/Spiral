"""
"""


from construction_requirements_integrator import CRI, construction_required
from constant_properties_protector import CPP
from typing import Union, Iterable
from typeguard import typechecked
import torch
from spiral.response_function import ResponseFunction
from spiral.analysis import Analyzer, analysis_point, analytics




@typechecked
class Axon(torch.nn.Module, CRI):
    def __init__(
        self,
        name: str = None,
        shape: Iterable[int] = None,
        terminal: Iterable[int] = (),
        response_function: ResponseFunction = None,
        is_excitatory: Union[bool, torch.Tensor] = True,
        delay: Union[float, torch.Tensor] = 0.,
        dt: Union[float, torch.Tensor] = None,
        analyzable: bool = False,
        construction_permission: bool = True,
    ) -> None:
        torch.nn.Module.__init__(self)
        CPP.protect(self, 'name')
        CPP.protect(self, 'shape')
        CPP.protect(self, 'terminal')
        CPP.protect(self, 'delay')
        CPP.protect(self, 'dt')
        CPP.protect(self, 'action_potential_history')
        CPP.protect(self, 'neurotransmitter')
        CPP.protect(self, 'is_excitatory')
        CRI.__init__(
            self,
            name=name,
            shape=shape,
            terminal=terminal,
            delay=delay,
            dt=dt,
            construction_permission=construction_permission,
            ignore_overwrite_error=True,
        )
        self.register_buffer("_is_excitatory", torch.tensor(is_excitatory))
        self.response_function = ResponseFunction() if response_function is None else response_function
        Analyzer.__init__(self, analyzable)
        Analyzer.scout(self, state_variables=['neurotransmitter'])


    def __construct__(
        self,
        name: str,
        shape: Iterable[int],
        terminal: Iterable[int],
        delay: Union[float, torch.Tensor],
        dt: Union[float, torch.Tensor],
    ) -> None:
        self._name = name
        self._shape = shape
        self._terminal = terminal
        self._is_excitatory = self.is_excitatory.reshape(
            *self.is_excitatory.shape, *[1]*(len(self.shape)+len(self.terminal)-len(self.is_excitatory.shape))
        )
        self.register_buffer("_dt", torch.tensor(dt))
        self.register_buffer("_delay", (torch.tensor(delay)//self.dt).type(torch.int64))
        history_length = self.delay.max()+1
        self.register_buffer("_action_potential_history", torch.zeros((history_length,*self.shape)))
        self.register_buffer("_neurotransmitter", torch.zeros(*self.shape, *self.terminal))
        self.response_function.meet_requirement(shape=(*self.shape, *self.terminal))
        self.response_function.meet_requirement(dt=self.dt)


    def _update_action_potential_history(
        self,
        action_potential: torch.Tensor
    ) -> None:
        self._action_potential_history = torch.cat((action_potential.unsqueeze(0), self.action_potential_history[:-1]))


    def _get_delayed_action_potential(
        self
    ) -> torch.Tensor:
        if self.delay.numel()==1:
            return self.action_potential_history[self.delay]
        else:
            history = self.action_potential_history.reshape(
                *self.action_potential_history.shape, *[1]*len(self.terminal)
            ).repeat(
                *[1]*len(self.action_potential_history.shape), *self.terminal
            )
            delay = self.delay.reshape(
                1, self.delay.shape, *[1]*(len(self.shape)+len(self.terminal)-len(self.delay.shape))
            ).repeat(
                1, *[1]*len(self.delay.shape), *self.action_potential_history.shape[-1-len(self.delay.shape):]
            )
            return torch.gather(history, dim=0, index=delay)


    @construction_required
    @analysis_point
    def forward(
        self,
        action_potential: torch.Tensor,
    ) -> None:
        self._update_action_potential_history(action_potential=action_potential)
        self._neurotransmitter = self.response_function(action_potential=self._get_delayed_action_potential())


    @construction_required
    def release(self) -> torch.Tensor:
        return self.neurotransmitter * (2*self.is_excitatory-1)


    @construction_required
    def reset(
        self
    ) -> None:
        self._neurotransmitter.zero_()
        self.response_function.reset()


    @analytics
    def plot__neurotransmitter(
        self,
        axes,
        **kwargs
    ) -> None:
        """
        Draw a plot of neurotransmitter dynamics on `axes`.

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
        y = self.monitor['neurotransmitter'].reshape(self.monitor['neurotransmitter'].shape[0],-1)
        time_range = (0, y.shape[0])
        x = torch.arange(*time_range)*self.dt
        population_alpha = 1/y.shape[1]
        aggregated = y.mean(axis=1)
        axes.plot(x, aggregated, color='cyan', **kwargs)
        axes.plot(x, y, alpha=population_alpha, color='cyan')
        axes.set_ylabel('neurotransmitter concentration')
        axes.set_xlabel('time (ms)')
        axes.set_xlim(time_range)