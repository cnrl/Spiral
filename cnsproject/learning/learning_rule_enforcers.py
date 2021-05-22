"""
Module for learning rules.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Sequence, Callable, Iterable

import numpy as np
import torch

from ..network.synapse_sets import AbstractSynapseSet
from .learning_rates import stdp_wdlr
from .synaptic_taggers import AbstractSynapticTagger, STDPST, FSTDPST
from ..network.axon_sets import AbstractAxonSet


class AbstractLearningRuleEnforcer(ABC, torch.nn.Module):
    def __init__(
        self,
        name: str = None,
        synapse: AbstractSynapseSet = None,
        dt: float = None,
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__()

        self.name = None
        self.configed = False
        self.set_name(name)
        self.synapse = None
        self.dt = None
        self.config_prohibit = config_prohibit
        self.set_dt(dt)
        self.set_synapse(synapse)


    def config_permit(self):
        return (
            self.synapse is not None and
            self.dt is not None and
            not self.config_prohibit and
            not self.configed
        )


    def config(self) -> bool:
        if not self.config_permit():
            return False
        self.configed = True
        self.set_name()
        return True


    def set_synapse(self, synapse: AbstractSynapseSet) -> bool:
        if self.configed:
            return False
        self.synapse = synapse
        self.config()
        return True


    def set_name(self, name:str=None) -> None:
        if name is not None:
            self.name = name
        if self.name is None and self.configed:
            self.name = self.synapse.name+"_LRE"


    def set_dt(self, dt:float) -> bool:
        if self.configed:
            return False
        self.dt = torch.tensor(dt) if dt is not None else dt
        self.config()
        return True


    def to_singlton_dendrite_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        return \
            self.synapse.dendrite.to_singlton_population_shape(
                self.synapse.axon.to_singlton_terminal_shape(
                    tensor
                )
            )

    
    def forward(self, direct_input: torch.Tensor = torch.tensor(0.)) -> None:
        pass

    
    @abstractmethod
    def backward(self) -> None:
        pass


    def reset(self) -> None:
        pass
            

    def add_axon(self, axon_set: Union[AbstractAxonSet, Iterable]) -> None:
        pass #### for Neuromodulatory LREs


    def remove_axon(self, name: str) -> None:
        pass #### for Neuromodulatory LREs

    
    def __str__(self) -> str:
        if self.configed:
            return f"{self.name} on: {self.synapse.__str__()}"
        else:
            return f"{self.name}(X)"




class CombinedLRE(AbstractLearningRuleEnforcer):
    def __init__(
        self,
        LRs: Iterable[AbstractLearningRuleEnforcer],
        **kwargs
    ) -> None:
        super().__init__(synapse=LRs[0].synapse, **kwargs)
        self.LRs = list(LRs)


    def set_synapse(self, synapse: AbstractSynapseSet) -> bool:
        super().set_synapse(synapse)
        if hasattr(self, "LRs"):
            [lr.set_synapse(synapse) for lr in self.LRs]


    def set_name(self, name:str=None) -> None:
        super().set_name(name)
        if hasattr(self, "LRs"):
            [lr.set_name(name) for lr in self.LRs]


    def set_dt(self, dt:float):
        super().set_dt(dt)
        if hasattr(self, "LRs"):
            [lr.set_dt(dt) for lr in self.LRs]

    
    def forward(self, direct_input: torch.Tensor = torch.tensor(0.)) -> None:
        [lr.forward(direct_input=direct_input) for lr in self.LRs]

    
    def backward(self) -> None:
        updatings = [lr.compute_updatings() for lr in self.LRs]
        [lr.update(u) for lr,u in zip(self.LRs, updatings)]


    def reset(self) -> None:
        [lr.reset() for lr in self.LRs]
            

    def add_axon(self, axon_set: Union[AbstractAxonSet, Iterable]) -> None:
        [lr.add_axon(axon_set) for lr in self.LRs]


    def remove_axon(self, name: str) -> None:
        [lr.remove_axon(name) for lr in self.LRs]


    def __add__(self, other: AbstractLearningRuleEnforcer):
        if type(other) is CombinedLRE:
            return CombinedLRE(self.LRs+other.LRs)
        else:
            return CombinedLRE(self.LRs+[other])


    def __str__(self) -> str:
        string = f"{self.name}, Combination of:\n"
        for lr in self.LRs:
            string += '\t'+lr.__str__()+'\n'
        return string


    def __getitem__(self, index: int) -> AbstractLearningRuleEnforcer:
        return self.LRs[index]



class CombinableLRE(AbstractLearningRuleEnforcer):
    def __init__(
        self,
        name: str = None,
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)


    @abstractmethod
    def compute_updatings(self):
        pass


    @abstractmethod
    def update(self, *data) -> None: # data = compute_updatings output
        pass


    def backward(self) -> None:
        self.update(self.compute_updatings())


    def __add__(self, other: AbstractLearningRuleEnforcer) -> CombinedLRE:
        if type(other) is CombinedLRE:
            return CombinedLRE([self]+other.LRs)
        else:
            return CombinedLRE([self, other])




class NoOp(CombinableLRE):
    def __init__(
        self,
        name: str = None,
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)

    
    def compute_updatings(self):
        return


    def update(self): # data = compute_updatings output
        return




class AbstractWeightLRE(CombinableLRE):
    """
    Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of STDP learning rule.You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        name: str = None,
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)


    @abstractmethod
    def compute_updatings(self) -> torch.Tensor: # output = dw
        pass


    def update(self, dw: torch.Tensor) -> None:
        w = self.synapse.dendrite.w
        wmin = self.synapse.dendrite.wmin
        wmax = self.synapse.dendrite.wmax
        w += dw
        w[w<wmin] = wmin
        w[w>wmax] = wmax




class SimpleWeightDecayLRE(AbstractWeightLRE):
    def __init__(
        self,
        name: str = None,
        decay: Union[float, torch.Tensor] = 0.,
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.register_buffer("decay", torch.tensor(decay))

    
    def compute_updatings(self) -> None:
        return self.synapse.dendrite.w * -self.decay




class CentristWeightDecayLRE(AbstractWeightLRE):
    def __init__(
        self,
        name: str = None,
        decay: Union[float, torch.Tensor] = 0.,
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.register_buffer("decay", torch.tensor(decay))

    
    def compute_updatings(self) -> None:
        return (self.synapse.dendrite.w - (self.synapse.dendrite.wmax + self.synapse.dendrite.wmin)/2) * -self.decay




class STDP(AbstractWeightLRE):
    """
    Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of STDP learning rule.You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        name: str = None,
        pre_traces: AbstractSynapticTagger = None,
        post_traces: AbstractSynapticTagger = None,
        ltp_wdlr: Callable = stdp_wdlr(.1), #LTP weight dependent learning rate
        ltd_wdlr: Callable = stdp_wdlr(.1), #LTD weight dependent learning rate
        config_prohibit: bool = False,
        **kwargs
    ) -> None:  
        super().__init__(config_prohibit=True, name=name, **kwargs)
        self.pre_traces = pre_traces if pre_traces is not None else STDPST()
        self.post_traces = post_traces if post_traces is not None else STDPST()
        self.ltp_wdlr = ltp_wdlr
        self.ltd_wdlr = ltd_wdlr
        self.config_prohibit = config_prohibit
        self.config()

    
    def config(self) -> bool:
        if not super().config():
            return False
        self.pre_traces.set_shape(self.synapse.axon.population_shape)
        self.post_traces.set_shape(self.synapse.dendrite.population_shape)
        self.pre_traces.set_dt(self.dt)
        self.post_traces.set_dt(self.dt)
        return True


    def compute_lrs(self) -> tuple: # ltp_lr,ltd_lr
        w = self.synapse.dendrite.w
        wmin = self.synapse.dendrite.wmin
        wmax = self.synapse.dendrite.wmax
        ltp_lr = self.ltd_wdlr(w, wmin, wmax)
        ltd_lr = self.ltd_wdlr(w, wmin, wmax)
        return ltp_lr,ltd_lr


    def compute_updatings(self) -> torch.Tensor:
        ltp_lr,ltd_lr = self.compute_lrs()
        ltp = ltp_lr * self.to_singlton_dendrite_shape(self.pre_traces.traces()) * self.synapse.dendrite.spikes()
        ltd = ltd_lr * self.post_traces.traces() * self.to_singlton_dendrite_shape(self.synapse.axon.spikes())
        dw = self.dt * (ltp - ltd)
        return dw


    def forward(self, direct_input: torch.Tensor = torch.tensor(0.)) -> None:
        self.pre_traces.forward(self.synapse.axon.spikes())
        self.post_traces.forward(self.synapse.dendrite.spikes())
        super().forward()


    def reset(self) -> None:
        self.pre_traces.reset()
        self.post_traces.reset()
        super().reset()




class FlatSTDP(STDP):
    def __init__(
        self,
        name: str = None,
        pre_time: Union[float, torch.Tensor] = 10.,
        post_time: Union[float, torch.Tensor] = 10.,
        **kwargs
    ) -> None:
        super().__init__(
            name=name,
            pre_traces=FSTDPST(pre_time),
            post_traces=FSTDPST(post_time),
            **kwargs
        )




class AbstractNeuromodulatoryLRE(CombinableLRE):
    def __init__(
        self,
        name: str = None,
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.axons = {}
        self.register_buffer("neuromodulators", torch.tensor(0.))
            

    def add_axon(self, axon_set: Union[AbstractAxonSet, Iterable]) -> None:
        if hasattr(axon_set, '__iter__'):
            for o in axon_set:
                self.usadd_axone(o)
        else:
            self.axons[axon_set.name] = axon_set


    def remove_axon(self, name: str) -> None:
        del self.axons[name]


    def collect_neuromodulators(self, direct_input: torch.Tensor = torch.tensor(0.)):
        neuromodulators = direct_input
        for axon_set in self.axons.values():
            neuromodulators = neuromodulators + axon_set.neurotransmitters()
        return neuromodulators


    def forward(self, direct_input: torch.Tensor = torch.tensor(0.)) -> None:
        self.neuromodulators = self.collect_neuromodulators(direct_input=direct_input)


    def reset(self) -> None:
        self.neuromodulators.zero_()
        super().reset()

    
    def __str__(self) -> str:
        string = super().__str__()+'\n\t\t'
        string += "affected by: "+ ', '.join([a.__str__() for a in self.axons])
        return string




class AbstractNeuromodulatoryWeightLRE(AbstractNeuromodulatoryLRE, AbstractWeightLRE):
    def __init__(
        self,
        name: str = None,
        **kwargs
    ) -> None:
        super().__init__(name, **kwargs)




class RSTDP(AbstractNeuromodulatoryWeightLRE):
    """
    Reward-modulated Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of RSTDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        name: str = None,
        stdp: STDP = STDP(),
        tau: Union[float, torch.Tensor] = 1000.,
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__(config_prohibit=True, name=name, **kwargs)
        self.stdp = stdp
        self.register_buffer("tau", torch.tensor(tau))
        self.register_buffer("c", torch.tensor(0.))
        self.config_prohibit = config_prohibit
        self.config()

    
    def config(self) -> bool:
        if not super().config():
            return False
        self.stdp.set_name(self.name)
        self.stdp.set_synapse(self.synapse)
        self.stdp.set_dt(self.dt)
        return True


    def forward(self, direct_input: torch.Tensor = torch.tensor(0.)) -> None:
        self.stdp.forward(direct_input=direct_input)
        stdp_output = self.stdp.compute_updatings()
        delta = (self.synapse.dendrite.spikes() + self.to_singlton_dendrite_shape(self.synapse.axon.spikes()))
        self.c = self.c + stdp_output*delta - self.dt * self.c / self.tau
        super().forward()


    def compute_updatings(self) -> torch.Tensor:
        dw = self.c * self.neuromodulators
        return dw


    def reset(self) -> None:
        self.stdp.reset()
        self.c.zero_()
        super().reset()




class FlatRSTDP(AbstractNeuromodulatoryWeightLRE):
    def __init__(
        self,
        name: str = None,
        stdp: STDP = FlatSTDP(),
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__(config_prohibit=True, name=name, **kwargs)
        self.stdp = stdp
        self.config_prohibit = config_prohibit
        self.config()

    
    def config(self) -> bool:
        if not super().config():
            return False
        self.stdp.set_name(self.name)
        self.stdp.set_synapse(self.synapse)
        self.stdp.set_dt(self.dt)
        return True


    def forward(self, direct_input: torch.Tensor = torch.tensor(0.)) -> None:
        self.stdp.forward(direct_input=direct_input)
        super().forward()


    def compute_updatings(self) -> torch.Tensor:
        dw = self.stdp.compute_updatings() * self.neuromodulators
        return dw


    def reset(self) -> None:
        self.stdp.reset()
        super().reset()