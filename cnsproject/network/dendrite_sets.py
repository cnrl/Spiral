"""
Module for connections between neural populations.
"""

from abc import ABC, abstractmethod
from typing import Union, Iterable
import torch
from .weight_initializations import constant_initialization
from .filters import AbstractFilter

class AbstractDendriteSet(ABC, torch.nn.Module):
    def __init__(
        self,
        name: str = None,
        terminal: Iterable[int] = None,
        population: Iterable[int] = None,
        wmin: Union[float, torch.Tensor] = 0.,
        wmax: Union[float, torch.Tensor] = 1.,
        dt: float = None,
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__()

        self.name = None
        self.set_name(name)
        self.register_buffer("wmin", torch.tensor(wmin))
        self.register_buffer("wmax", torch.tensor(wmax))
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
        self.shape = (*self.terminal_shape,*self.population_shape)
        self.register_buffer("I", torch.zeros(*self.shape)) #mA
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


    def set_terminal_shape(self, terminal: Iterable[int] = ()) -> bool:
        if self.configed:
            return False
        self.terminal_shape = terminal
        self.config()
        return True


    def set_population_shape(self, population: Iterable[int]) -> bool:
        if self.configed:
            return False
        self.population_shape = population
        self.config()
        return True


    def to_singlton_population_shape(self, tensor: torch.Tensor):
        if tensor.numel()==1 or tensor.shape==self.shape:
            return tensor
        else:
            return tensor.reshape((*tensor.shape,*[1]*(len(self.shape)-len(tensor.shape))))


    @abstractmethod
    def forward(self, neurotransmitters: torch.Tensor) -> None: #e: spike resonse  in shape (*self.terminal_shape,*self.population_shape)
        pass


    def backward(self, spikes: torch.Tensor) -> None: # population spike in shape(self.population_shape)
        self.s = spikes


    def reset(self) -> None:
        self.I.zero_()


    @abstractmethod
    def currents(self) -> torch.Tensor: # in shape *self.population_shape
        pass


    def spikes(self) -> torch.Tensor: # in shape (*self.population_shape)
        return self.s


    def __str__(self):
        if self.configed:
            return self.name
        else:
            return f"{self.name}(X)"




class SimpleDendriteSet(AbstractDendriteSet):
    def __init__(
        self,
        name: str = None,
        w: torch.Tensor = None, # in shape (*self.terminal_shape, *self.population_shape) or *self.population_shape or 1
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__(config_prohibit=True, name=name, **kwargs)
        if w is None:
            w = constant_initialization((self.wmax + self.wmin)/2)
        self.w_func = w
        self.config_prohibit = config_prohibit
        self.config()

    
    def config(self) -> bool:
        if not super().config():
            return False
        self.register_buffer("w", self.w_func(self.terminal_shape, self.population_shape))
        self.w[self.w<self.wmin] = self.wmin
        self.w[self.w>self.wmax] = self.wmax
        return True


    def forward(self, neurotransmitters: torch.Tensor) -> None: #doesn't replace nan values
        neurotransmitters_singleton = self.to_singlton_population_shape(neurotransmitters)
        self.I = neurotransmitters_singleton * self.w


    def currents(self) -> torch.Tensor:
        return self.I.sum(axis=list(range(len(self.terminal_shape))))




class NanBlockerDendriteSet(SimpleDendriteSet):
    def __init__(
        self,
        name: str=None,
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)


    def forward(self, neurotransmitters: torch.Tensor) -> None: #doesn't replace nan values
        neurotransmitters_singleton = self.to_singlton_population_shape(neurotransmitters)
        I = neurotransmitters_singleton * self.w
        I[neurotransmitters.isnan()] = 0
        self.I = self.I*neurotransmitters_singleton.isnan() + I




class FilteringDendriteSet2D(AbstractDendriteSet):
    def __init__(
        self,
        name: str = None,
        filt: AbstractFilter = None,
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            name=name,
            config_prohibit=True,
            **kwargs
        )
        self.config_prohibit = config_prohibit
        self.set_filter(filt)


    def config_permit(self):
        return (super().config_permit() and (self.filter is not None))

    
    def config(self) -> bool:
        if not self.config_permit():
            return False
        assert self.required_population_shape()==self.population_shape, "terminal shape doesn't match with population shape according to filter"
        return super().config()


    def set_filter(self, filt) -> bool:
        if self.configed:
            return False
        self.filter = filt
        if self.filter is not None:
            self.add_module('filter', self.filter)
        self.config()
        return True


    def required_population_shape(self) -> Iterable[int]:
        assert (self.terminal_shape is not None and self.filter is not None), \
            "please set terminal and filter at the first place."
        return self.filter(torch.zeros(self.terminal_shape)).shape


    def forward(self, neurotransmitters: torch.Tensor) -> None: #doesn't replace nan values
        self.I = self.filter(neurotransmitters)


    def currents(self) -> torch.Tensor:
        return self.I