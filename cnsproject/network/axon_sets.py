"""
"""

from abc import ABC, abstractmethod
from typing import Union, Iterable
from ..utils import masked_shift
import torch

class AbstractAxonSet(ABC, torch.nn.Module):
    def __init__(
        self,
        name: str = None,
        population: Iterable[int] = None,
        terminal: Iterable[int] = (),
        is_excitatory: Union[bool, torch.Tensor] = True,
        dt: float = None,
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__()

        self.name = None
        self.set_name(name)
        self.register_buffer("is_excitatory", torch.tensor(is_excitatory))
        self.population_shape = None
        self.terminal_shape = None
        self.dt = None
        self.configed = False
        self.config_prohibit = config_prohibit
        self.set_dt(dt)
        self.set_terminal_shape(terminal)
        self.set_population_shape(population)


    def config_permit(self):
        return (
            self.population_shape is not None and
            self.terminal_shape is not None and
            self.dt is not None and
            not self.config_prohibit and
            not self.configed
        )


    def config(self) -> bool:
        if not self.config_permit():
            return False
        self.shape = (*self.population_shape,*self.terminal_shape)
        self.register_buffer("e", torch.zeros(self.shape) * 0.)
        self.configed = True
        return True


    def set_name(self, name:str=None, soft=False) -> None:
        if self.name is not None and soft:
            return
        self.name = name


    def set_dt(self, dt:float) -> bool:
        if self.configed:
            return False
        self.dt = torch.tensor(dt) if dt is not None else dt
        self.config()
        return True


    def set_population_shape(self, population: Iterable[int]) -> bool:
        if self.configed:
            return False
        self.population_shape = population
        self.config()
        return True


    def set_terminal_shape(self, terminal: Iterable[int] = ()) -> bool:
        if self.configed:
            return False
        self.terminal_shape = terminal
        self.config()
        return True


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


    def __str__(self):
        if self.configed:
            return self.name
        else:
            return f"{self.name}(X)"




class SimpleAxonSet(AbstractAxonSet):
    def __init__(
        self,
        name: str = None,
        scale: Union[float, torch.Tensor] = 1.,
        delay: Union[int, torch.Tensor] = 0., #ms
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__(config_prohibit=True, name=name, **kwargs)
        self.register_buffer("delay_time", torch.tensor(delay))
        self.register_buffer("scale", torch.tensor(scale))
        self.config_prohibit = config_prohibit
        self.config()


    def config(self) -> bool:
        if not super().config():
            return False
        self.register_buffer("delay", self.delay_time//self.dt)
        self.delay = self.delay.type(torch.int64)
        self.max_delay = self.delay.max()
        self.register_buffer("spike_history", torch.zeros((self.max_delay,*self.population_shape), dtype=torch.bool))
        self.configed = True
        return True


    def forward(self, spikes: torch.Tensor) -> None:
        self.update_spike_history(spikes)
        s = self.get_delayed_spikes().clone()
        self.compute_response(s)
        self.s = self.spike_history[0]
        self.minimize_spike_history()


    def compute_response(self, s: torch.Tensor) -> None:
        self.e = s * 1.
        if len(self.shape)>0:
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
            delay = self.delay.unsqueeze(0).repeat([1]+[self.shape[i]//self.delay.shape[i] for i in range(len(self.delay.shape))])
            output = torch.gather(output, dim=0, index=delay)
            return output[0]


    def reset(self) -> None:
        self.spike_history.zero_()
        super().reset()


    def neurotransmitters(self) -> None:
        return self.to_singlton_terminal_shape(self.scale) * super().neurotransmitters()




class SRFAxonSet(SimpleAxonSet): #Spike response function
    def __init__(
        self,
        name: str = None,
        tau: Union[float, torch.Tensor] = 10.,
        msatst: int = 10, #max_spikes_at_the_same_time
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__(config_prohibit=True, name=name, **kwargs)
        self.register_buffer("tau", torch.tensor(tau))
        self.msatst = msatst
        self.config_prohibit = config_prohibit
        self.config()

    
    def config(self) -> bool:
        if not super().config():
            return False
        self.tau = self.to_singlton_terminal_shape(self.tau)
        self.register_buffer("lstd", torch.ones(self.msatst, *self.population_shape, *self.delay.shape[len(self.population_shape):])*float("Inf")) #last spike time difference
        self.register_buffer("sub_e", torch.zeros(*self.lstd.shape, *self.tau.shape[len(self.lstd.shape)-1:]))
        if len(self.shape)>=len(self.lstd.shape):
            self.lstd = self.to_singlton_terminal_shape(self.lstd).unsqueeze(-1)
        if len(self.shape)>=len(self.sub_e.shape):
            self.sub_e = self.to_singlton_terminal_shape(self.sub_e).unsqueeze(-1)
        return True


    def compute_response(self, s: torch.Tensor) -> None:
        self.lstd += self.dt
        self.lstd = masked_shift(self.lstd, s)
        self.sub_e = masked_shift(self.sub_e, s)
        remained_spike_effect = 1 - self.lstd/self.tau
        d_epsilon = self.dt * remained_spike_effect * torch.exp(remained_spike_effect) / self.tau
        d_epsilon[torch.isnan(d_epsilon)] = 0
        self.sub_e += d_epsilon
        e = self.sub_e.sum(axis=0)
        super().compute_response(e)


    def reset(self) -> None:
        self.lstd = torch.ones(self.lstd.shape)*float("Inf")
        self.sub_e.zero_()
        super().reset()




class SimpleDecayAxonSet(SimpleAxonSet): # de/dt = -e/tau + s(t)
    def __init__(
        self,
        name: str = None,
        tau: Union[float, torch.Tensor] = 10.,
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.register_buffer("tau", torch.tensor(tau))


    def compute_response(self, s: torch.Tensor) -> None:
        e_0 = self.e.clone()
        super().compute_response(s)
        self.e = self.dt * (self.e + e_0 * (1/self.dt - 1/self.tau))