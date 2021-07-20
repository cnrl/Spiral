"""
Module for connections between neural populations.
"""


from abc import ABC, abstractmethod
from typing import Union, Iterable
from typeguard import typechecked
import torch
from .weight_initialization import constant_initialization
from ..filter.filter import AbstractFilter




@typechecked
class Dendrite(torch.nn.Module, CRI, ABC):
    def __init__(
        self,
        name: str = None,
        shape: Iterable[int] = None,
        spine: Iterable[int] = None,
        dt: float = None,
        synaptic_plasticity: SynapticPlasticity = None,
        analyzable: bool = False,
        construction_permission: bool = True,
    ) -> None:
        torch.nn.Module.__init__(self)
        CPP.protect(self, 'name')
        CPP.protect(self, 'shape')
        CPP.protect(self, 'spine')
        CPP.protect(self, 'dt')
        CPP.protect(self, 'current')
        self.synaptic_plasticity = SynapticPlasticity() if synaptic_plasticity is None else synaptic_plasticity
        Analyzer.__init__(self, analyzable)
        Analyzer.scout(self, state_variables=['current'])
        CRI.__init__(
            self,
            name=name,
            shape=shape,
            spine=spine,
            dt=dt,
            construction_permission=construction_permission,
            ignore_overwrite_error=True,
        )


    def __construct__(
        self,
        name: str,
        shape: Iterable[int],
        spine: Iterable[int],
        dt: Union[float, torch.Tensor],
    ) -> None:
        self._name = name
        self._shape = shape
        self._spine = spine
        self.register_buffer("_dt", torch.as_tensor(dt))
        self.register_buffer("_current", torch.zeros(self.shape))
        self.response_function.meet_requirement(spine=self.spine)
        self.response_function.meet_requirement(shape=self.shape)
        self.response_function.meet_requirement(dt=self.dt)


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
        self.dt = torch.as_tensor(dt) if dt is not None else dt
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




class SimpleDendriteSet(Dendrite):
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




class FilteringDendriteSet2D(Dendrite):
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



# class AbstractKernelWeightLRE(CombinableLRE):
#     def __init__(
#         self,
#         name: str = None,
#         **kwargs
#     ) -> None:
#         super().__init__(name=name, **kwargs)


#     @abstractmethod
#     def compute_updatings(self) -> torch.Tensor: # output = dw
#         pass


#     def update(self, dw: torch.Tensor) -> None:
#         w = self.synapse.dendrite.filter.core.weight.data
#         wmin = self.synapse.dendrite.wmin
#         wmax = self.synapse.dendrite.wmax
#         w += dw
#         w[w<wmin] = wmin
#         w[w>wmax] = wmax
#         self.synapse.dendrite.filter.core.weight.data = w