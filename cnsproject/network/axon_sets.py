"""
"""

from abc import ABC, abstractmethod
from typing import Union, Iterable
from ..utils import masked_shift
import torch

class AbstractAxonSet(ABC, torch.nn.Module):
    def __init__(
        self,
        population: Iterable[int],
        terminal: Iterable[int] = (),
        is_excitatory: Union[bool, torch.Tensor] = True,
        dt: float = None,
        **kwargs
    ) -> None:
        super().__init__()

        self.population_shape = population
        self.terminal_shape = terminal
        self.shape = (*self.population_shape,*self.terminal_shape)
        self.register_buffer("is_excitatory", self.to_singlton_terminal_shape(torch.tensor(is_excitatory)))
        self.register_buffer("e", torch.zeros(self.shape) * 0.)
        self.set_dt(dt)


    def set_dt(self, dt:float):
        self.dt = torch.tensor(dt) if dt is not None else dt


    def to_singlton_terminal_shape(self, tensor: torch.Tensor):
        if tensor.numel()==1 or tensor.shape==self.shape:
            return tensor
        else:
            return tensor.reshape((*tensor.shape,*[1]*(len(self.shape)-len(tensor.shape))))


    @abstractmethod
    def forward(self, spikes: torch.Tensor) -> None: #s: spike in shape *self.population_shape
        self.s = spikes


    @abstractmethod
    def reset(self) -> None:
        self.e.zero_()


    @abstractmethod
    def neurotransmitters(self) -> torch.Tensor: # in shape (*self.population_shape,*self.terminal_shape)
        return self.e * (2*self.to_singlton_terminal_shape(self.is_excitatory)-1)


    def spikes(self) -> torch.Tensor: # in shape (*self.population_shape)
        return self.s




class SimpleAxonSet(AbstractAxonSet):
    def __init__(
        self,
        scale: Union[float, torch.Tensor] = 1.,
        delay: Union[int, torch.Tensor] = 0., #ms
        dt: float = None,
        **kwargs
    ) -> None:
        super().__init__(dt=None, **kwargs)
        self.register_buffer("delay_time", torch.tensor(delay))
        self.register_buffer("scale", self.to_singlton_terminal_shape(torch.tensor(scale)))
        self.set_dt(dt)


    def set_dt(self, dt:float):
        super().set_dt(dt)
        if self.dt is not None:
            self.register_buffer("delay", self.delay_time//self.dt)
            self.delay = self.delay.type(torch.int64)
            self.max_delay = self.delay.max()
            self.register_buffer("spike_history", torch.zeros((self.max_delay,*self.population_shape), dtype=torch.bool))


    def forward(self, spikes: torch.Tensor) -> None:
        self.update_spike_history(spikes)
        s = self.get_delayed_spikes().clone()
        self.compute_response(s)
        self.s = self.spike_history[0]
        self.minimize_spike_history()


    def compute_response(self, s: torch.Tensor) -> None:
        self.e = s * 1.
        self.e = self.to_singlton_terminal_shape(self.e)
        self.e = self.e.repeat(*[1]*len(s.shape), *self.shape[len(s.shape):])


    def update_spike_history(self, s: torch.Tensor) -> None:
        self.spike_history = torch.cat((s.unsqueeze(0), self.spike_history))


    def minimize_spike_history(self) -> None:
        self.spike_history = self.spike_history[:self.max_delay]


    def get_delayed_spikes(self) -> torch.Tensor:
        if self.delay.numel()==1:
            return self.spike_history[self.delay]
        else: # delay shape can be like (*spike shape, ...)
            spike_shape = self.spike_history[0].shape
            diff_shape = self.delay.shape[len(spike_shape):]
            repeating_shape = self.spike_history.shape+tuple(1 for i in diff_shape)
            output = self.spike_history.reshape(repeating_shape)
            repeating = (1,)+tuple(1 for i in spike_shape)+diff_shape
            output = output.repeat(repeating)
            output = output.reshape(*self.spike_history.shape, *diff_shape)
            output = torch.gather(output, dim=0, index=self.delay.unsqueeze(0))
            return output[0]


    def reset(self) -> None:
        self.spike_history.zero_()
        super().reset()


    def neurotransmitters(self) -> None:
        return self.scale * super().neurotransmitters()




class SRFAxonSet(SimpleAxonSet): #Spike response function
    def __init__(
        self,
        tau: Union[float, torch.Tensor] = 10.,
        max_spikes_at_the_same_time: int = 10,
        dt: float = None,
        **kwargs
    ) -> None:
        super().__init__(dt=dt, **kwargs)
        self.register_buffer("tau", self.to_singlton_terminal_shape(torch.tensor(tau)))
        self.max_spikes_at_the_same_time = max_spikes_at_the_same_time
        self.set_dt(dt)

    def set_dt(self, dt:float):
        super().set_dt(dt)
        if self.dt is not None:
            self.register_buffer("lstd", torch.ones(self.max_spikes_at_the_same_time, *self.population_shape, *self.delay.shape[len(self.population_shape):])*float("Inf")) #last spike time difference
            self.register_buffer("sub_e", torch.zeros(*self.lstd.shape, *self.tau.shape[len(self.lstd.shape)-1:]))


    def compute_remained_spike_effect(self):
        if len(self.tau.shape)<len(self.lstd.shape):
            return 1 - self.lstd/self.tau
        else:
            return 1 - self.lstd.reshape(*self.lstd.shape, *[1]*(len(self.tau.shape)-len(self.lstd.shape)+1)) / self.tau


    def compute_response(self, s: torch.Tensor) -> None:
        self.lstd += self.dt
        self.lstd = masked_shift(self.lstd, s)
        self.sub_e = masked_shift(self.sub_e, s)
        remained_spike_effect = self.compute_remained_spike_effect()
        d_epsilon = self.dt * remained_spike_effect * torch.exp(remained_spike_effect) / self.tau
        d_epsilon[torch.isnan(d_epsilon)] = 0
        self.sub_e += d_epsilon
        e = self.sub_e.sum(axis=0)
        super().compute_response(e)


    def reset(self) -> None:
        self.lstd = torch.ones(self.lstd.shape)*float("Inf")
        self.sub_e.zero_()
        super().reset()
