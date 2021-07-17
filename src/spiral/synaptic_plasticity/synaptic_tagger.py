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


class AbstractSynapticTagger(ABC, torch.nn.Module):
    def __init__(
        self,
        shape: Iterable[int] = None,
        dt: float = None,
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.configed = False
        self.config_prohibit = config_prohibit
        self.set_shape(shape)
        self.set_dt(dt)


    def config_permit(self):
        return (
            self.shape is not None and
            self.dt is not None and
            not self.config_prohibit and
            not self.configed
        )


    def config(self) -> bool:
        if not self.config_permit():
            return False
        self.register_buffer("tr", torch.zeros(self.shape)) #tr = traces
        self.configed = True
        return True


    def set_shape(self, shape):
        if self.configed:
            return False
        self.shape = shape
        self.config()
        return True


    def set_dt(self, dt:float):
        if self.configed:
            return False
        self.dt = torch.tensor(dt) if dt is not None else dt
        self.config()
        return True

    
    def forward(self, s: torch.Tensor) -> None: #s: spikes
        pass


    def traces(self) -> torch.Tensor:
        return self.tr


    def reset(self) -> None:
        self.tr.zero_()




class SimpleSynapticTagger(AbstractSynapticTagger):
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



class AdditiveSynapticTagger(SimpleSynapticTagger):
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)


    def decay_traces(self, s: torch.Tensor) -> torch.Tensor:
        return 0



class STDPST(SimpleSynapticTagger): # STDP-SynapticTagger
    def __init__(
        self,
        tau: Union[float, torch.Tensor] = 15.,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.register_buffer("tau", torch.tensor(tau))

    
    def decay_traces(self, s: torch.Tensor) -> None: #s: spikes
        return self.dt * self.tr / self.tau



class FSTDPST(SimpleSynapticTagger): # Flat-STDP-SynapticTagger
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

    
    def decay_traces(self, s: torch.Tensor) -> None: #s: spikes
        return 0



class LFSTDPST(AbstractSynapticTagger): # Limited Flat-STDP-SynapticTagger
    def __init__(
        self,
        time: float = 10.,
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__(config_prohibit=True, **kwargs)
        self.time = time
        self.config_prohibit = config_prohibit
        self.config()


    def config(self) -> bool:
        if not self.config_permit():
            return False
        self.length = int(self.time//self.dt)
        self.register_buffer("spike_history", torch.zeros((self.length,*self.shape), dtype=torch.bool))
        self.configed = True
        return True

    
    def forward(self, s: torch.Tensor) -> None: #s: spikes
        self.spike_history = torch.cat([s.unsqueeze(0), self.spike_history[:-1]])


    def traces(self) -> torch.Tensor:
        return self.spike_history.sum(axis=0)

    
    def reset(self) -> None:
        self.spike_history.zero_()




class NASTDPST(STDPST): # Non-Additive-STDP-SynapticTagger
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

    
    def decay_traces(self, s: torch.Tensor) -> None: #s: spikes
        decay = super().decay_traces()
        decay[s] = self.tr
        return decay