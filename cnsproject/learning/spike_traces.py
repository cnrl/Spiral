"""
Module for SpikeTraces.
"""

from abc import ABC, abstractmethod
from typing import Union, Iterable

import numpy as np
import torch

from ..network.synapse_sets import AbstractSynapseSet
from ..utils import Serializer
from .learning_rates import constant_wdlr,stdp_wdlr


class AbstractSpikeTrace(ABC, torch.nn.Module):
    def __init__(
        self,
        shape: Iterable[int] = None,
        dt: float = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.set_shape(shape)
        self.set_dt(dt)


    def set_shape(self, shape):
        self.shape = shape
        if self.shape is not None:
            self.register_buffer("tr", torch.zeros(self.shape)) #tr = traces


    def set_dt(self, dt:float):
        self.dt = torch.tensor(dt) if dt is not None else dt

    
    def forward(self, s: torch.Tensor) -> None: #s: spikes
        pass


    def traces(self) -> torch.Tensor:
        return self.tr


    def reset(self) -> None:
        self.traces.zero_()




class SimpleSpikeTrace(AbstractSpikeTrace):
    def __init__(
        self,
        scale: Union[float, torch.Tensor] = 1.,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.register_buffer("scale", torch.tensor(scale))


    def forward(self, s: torch.Tensor) -> None: #s: spikes
        self.tr += self.flip_traces(s) - self.decay_traces(s)


    def flip_traces(self, s: torch.Tensor) -> torch.Tensor: #s: spikes
        return s * self.scale


    def decay_traces(self, s: torch.Tensor) -> torch.Tensor: #s: spikes
        return self.tr



class AdditiveSpikeTrace(SimpleSpikeTrace):
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)


    def decay_traces(self, s: torch.Tensor) -> torch.Tensor:
        return 0



class STDPSpikeTrace(SimpleSpikeTrace):
    def __init__(
        self,
        tau: Union[float, torch.Tensor] = 15.,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.register_buffer("tau", torch.tensor(tau))

    
    def decay_traces(self, s: torch.Tensor) -> None: #s: spikes
        return self.dt * self.tr / self.tau




class NASTDPST(STDPSpikeTrace): # Non-Additive-STDP-Spike-Trace
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

    
    def decay_traces(self, s: torch.Tensor) -> None: #s: spikes
        decay = super().decay_traces()
        decay[s] = self.tr
        return decay