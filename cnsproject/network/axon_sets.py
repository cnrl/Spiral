"""
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Callable

import torch

class AbstractAxonSet(ABC, torch.nn.Module):
    def __init__(
        self,
        population_shape: Iterable[int],
        terminal_shape: Iterable[int] = (),
        is_excitatory: Union[bool, torch.Tensor] = True,
        dt: float = None,
        **kwargs
    ) -> None:
        super().__init__()

        self.population_shape = population_shape
        self.terminal_shape = terminal_shape
        self.shape = (*self.population_shape,*self.terminal_shape)
        self.register_buffer("is_excitatory", add_terminal_shape(torch.tensor(is_excitatory)))
        self.register_buffer("e", torch.zeros(self.shape))
        self.set_dt(dt)


    def set_dt(self, dt:float):
        self.dt = torch.tensor(dt) if dt is not None else dt


    def add_terminal_shape(self, tensor: torch.Tensor):
        if tensor.numel()>1:
            return tensor.reshape((*self.population_shape,*[1]*len(self.terminal_shape)))
        else:
            return tensor


    @abstractmethod
    def forward(self, s: torch.Tensor) -> None: #s: spike in shape *self.population_shape
        pass


    @abstractmethod
    def reset(self) -> None:
        self.e.zero_()


    @abstractmethod
    def get_output(self) -> torch.Tensor: # in shape (*self.population_shape,*self.terminal_shape)
        return e.clone()




class SimpleAxonSet(AbstractAxonSet):
    def __init__(
        self,
        scale: Union[float, torch.Tensor] = 1.,
        delay: Union[int, torch.Tensor] = 0, #dt
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.register_buffer("scale", add_terminal_shape(torch.tensor(scale)))
        self.register_buffer("delay", add_terminal_shape(torch.tensor(delay)))
        self.max_delay = self.delay.max()
        self.register_buffer("spike_history", torch.zeros((self.max_delay,*self.population_shape), dtype=torch.bool))


    def forward(self, s: torch.Tensor) -> None:
        self.update_spike_history(s)
        s = self.get_delayed_spikes().clone()
        self.compute_response(s)
        self.minimize_spike_history()


    def compute_response(self, s: torch.Tensor) -> None:
        self.e = s


    def update_spike_history(self, s: torch.Tensor) -> None:
        self.spike_history = torch.cat((s.reshape(1,-1), self.spike_history))


    def minimize_spike_history(self) -> None:
        self.spike_history = self.spike_history[:self.max_delay]


    def get_delayed_spikes(self) -> torch.Tensor:
        if self.delay.numel()==1:
            return self.spike_history[self.delay]
        else: # delay shape can be like (*spike shape, ...)
            spike_shape = self.spike_history.shape[1:]
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


    def get_output(self) -> None:
        return self.e * self.scale * (2*self.is_excitatory-1)




class SRFAxonSet(SimpleAxonSet): #Spike response function
    def __init__(
        self,
        tau: Union[float, torch.Tensor] = 15.,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.register_buffer("tau", add_terminal_shape(torch.tensor(tau)))
        self.register_buffer("lstd", torch.ones(self.population_shape)*float("Inf")) #last spike time difference
        

    def compute_response(self, s: torch.Tensor) -> None:
        self.lstd *= ~s
        temp = 1 - self.lstd/self.tau
        d_epsilon = self.dt * temp * torch.exp(temp) / self.tau
        self.e = torch.abs(self.e) + d_epsilon


    def reset(self) -> None:
        self.lstd = torch.ones(self.population_shape)*float("Inf")
        super().reset()
