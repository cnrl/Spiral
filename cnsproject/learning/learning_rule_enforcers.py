"""
Module for learning rules.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Sequence, Callable, Iterable

import numpy as np
import torch

from ..network.synapse_sets import AbstractSynapseSet
from .learning_rates import constant_wdlr
from .synaptic_taggers import AbstractSynapticTagger, STDPST, FSTDPST


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

    
    def forward(self, neuromodulators: torch.Tensor = None) -> None:
        pass

    
    @abstractmethod
    def backward(self, **args) -> None:
        pass


    def reset(self) -> None:
        pass

    
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

    
    def forward(self, neuromodulators: torch.Tensor = None) -> None:
        [lr.forward(neuromodulators) for lr in self.LRs]

    
    def backward(self, **args) -> None:
        updatings = [lr.compute_updatings(**args) for lr in self.LRs]
        [lr.update(u) for lr,u in zip(self.LRs, updatings)]


    def reset(self) -> None:
        [lr.reset() for lr in self.LRs]


    def __add__(self, other: AbstractLearningRuleEnforcer):
        if type(other) is CombinedLRE:
            return CombinedLRE(self.LRs+other.LRs)
        else:
            return CombinedLRE(self.LRs+[other])




class CombinableLRE(AbstractLearningRuleEnforcer):
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)


    @abstractmethod
    def compute_updatings(self, **args):
        pass


    @abstractmethod
    def update(self, *data) -> None: # data = compute_updatings output
        pass


    def backward(self, **args) -> None:
        self.update(self.compute_updatings(**args))


    def __add__(self, other: AbstractLearningRuleEnforcer) -> CombinedLRE:
        if type(other) is CombinedLRE:
            return CombinedLRE([self]+other.LRs)
        else:
            return CombinedLRE([self, other])




class NoOp(CombinableLRE):
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

    
    def compute_updatings(self, **args):
        return


    def update(self, *data): # data = compute_updatings output
        return




class AbstractWeightLRE(CombinableLRE):
    """
    Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of STDP learning rule.You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)


    @abstractmethod
    def compute_updatings(self, **args) -> torch.Tensor: # output = dw
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
        decay: Union[float, torch.Tensor] = 0.,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.register_buffer("decay", torch.tensor(decay))

    
    def compute_updatings(self, **args) -> None:
        return self.synapse.dendrite.w * -self.decay




class STDP(AbstractWeightLRE):
    """
    Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of STDP learning rule.You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        pre_traces: AbstractSynapticTagger = None,
        post_traces: AbstractSynapticTagger = None,
        ltp_wdlr: Callable = constant_wdlr(.1), #LTP weight dependent learning rate
        ltd_wdlr: Callable = constant_wdlr(.1), #LTD weight dependent learning rate
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__(config_prohibit=True, **kwargs)
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


    def to_singlton_dendrite_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        return \
            self.synapse.dendrite.to_singlton_population_shape(
                self.synapse.axon.to_singlton_terminal_shape(
                    tensor
                )
            )


    def compute_updatings(self) -> torch.Tensor:
        ltp_lr,ltd_lr = self.compute_lrs()
        ltp = ltp_lr * self.to_singlton_dendrite_shape(self.pre_traces.traces()) * self.synapse.dendrite.spikes()
        ltd = ltd_lr * self.post_traces.traces() * self.to_singlton_dendrite_shape(self.synapse.axon.spikes())
        dw = self.dt * (ltp - ltd)
        return dw


    def forward(self, neuromodulators: torch.Tensor = None) -> None:
        self.pre_traces.forward(self.synapse.axon.spikes())
        self.post_traces.forward(self.synapse.dendrite.spikes())
        super().forward(neuromodulators=neuromodulators)


    def reset(self) -> None:
        self.pre_traces.reset()
        self.post_traces.reset()
        super().reset()




class FlatSTDP(STDP):
    def __init__(
        self,
        pre_time: Union[float, torch.Tensor] = 10.,
        post_time: Union[float, torch.Tensor] = 10.,
        **kwargs
    ) -> None:
        super().__init__(
            pre_traces=FSTDPST(pre_time),
            post_traces=FSTDPST(post_time),
            **kwargs
        )




# class RSTDP(AbstractWeightLRE):
#     """
#     Reward-modulated Spike-Time Dependent Plasticity learning rule.

#     Implement the dynamics of RSTDP learning rule. You might need to implement\
#     different update rules based on type of connection.
#     """

#     def __init__(
#         self,
#         connection: AbstractConnection,
#         lr: Optional[Union[float, Sequence[float]]] = None,
#         weight_decay: float = 0.,
#         **kwargs
#     ) -> None:
#         super().__init__(
#             connection=connection,
#             lr=lr,
#             weight_decay=weight_decay,
#             **kwargs
#         )
#         """
#         TODO.

#         Consider the additional required parameters and fill the body\
#         accordingly.
#         """

#     def update(self, **kwargs) -> None:
#         """
#         TODO.

#         Implement the dynamics and updating rule. You might need to call the
#         parent method. Make sure to consider the reward value as a given keyword
#         argument.
#         """
#         pass


# class FlatRSTDP(LearningRule):
#     """
#     Flattened Reward-modulated Spike-Time Dependent Plasticity learning rule.

#     Implement the dynamics of Flat-RSTDP learning rule. You might need to implement\
#     different update rules based on type of connection.
#     """

#     def __init__(
#         self,
#         connection: AbstractConnection,
#         lr: Optional[Union[float, Sequence[float]]] = None,
#         weight_decay: float = 0.,
#         **kwargs
#     ) -> None:
#         super().__init__(
#             connection=connection,
#             lr=lr,
#             weight_decay=weight_decay,
#             **kwargs
#         )
#         """
#         TODO.

#         Consider the additional required parameters and fill the body\
#         accordingly.
#         """

#     def update(self, **kwargs) -> None:
#         """
#         TODO.

#         Implement the dynamics and updating rule. You might need to call the
#         parent method. Make sure to consider the reward value as a given keyword
#         argument.
#         """
#         pass
