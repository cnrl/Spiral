"""
Module for learning rules.
"""

from abc import ABC
from typing import Union, Optional, Sequence, Callable

import numpy as np
import torch

from ..network.synapse_sets import AbstractSynapseSet
from ..utils import Serializer
from .learning_rates import constant_wdlr
from .spike_traces import AbstractSpikeTrace, STDPSpikeTrace, AdditiveSpikeTrace


class AbstractLearningRuler(ABC, torch.nn.Module):
    def __init__(
        self,
        synapse_set: AbstractSynapseSet,
        dt: float = None,
        **kwargs
    ) -> None:
        super().__init__()

        self.synapse_set = synapse_set
        self.set_dt(dt)


    def set_dt(self, dt:float):
        self.dt = torch.tensor(dt) if dt is not None else dt

    
    @abstractmethod
    def forward(self, neuromodulators: torch.Tensor = None) -> None:
        pass

    
    @abstractmethod
    def backward(self, **args) -> None:
        pass


    @abstractmethod
    def reset(self) -> None:
        pass


    def __add__(self, other) -> AbstractLearningRuler:
        return Serializer([self, other])




class NoOp(AbstractLearningRuler):
    def __init__(
        self,
        synapse_set: AbstractSynapseSet,
        **kwargs
    ) -> None:
        super().__init__(synapse_set=synapse_set)




class AbstractWeightLR(AbstractLearningRuler):
    """
    Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of STDP learning rule.You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        synapse_set: AbstractSynapseSet,
        **kwargs
    ) -> None:
        super().__init__(synapse_set=AbstractSynapseSet)


    @abstractmethod
    def compute_dw(self) -> torch.Tensor:
        pass


    @abstractmethod
    def update_w(self, dw: torch.Tensor) -> None:
        w = self.synapse_set.dendrite_set.w
        wmin = self.synapse_set.dendrite_set.wmin
        wmax = self.synapse_set.dendrite_set.wmax
        w += dw
        w[w<wmin] = wmin
        w[w>wmax] = wmax

    
    def forward(self, neuromodulators: torch.Tensor = None) -> None:
        pass


    @abstractmethod
    def backward(self, **args) -> None:
        self.update_w(self.compute_dw())




class SimpleWeightDecayLR(AbstractWeightLR):
    def __init__(
        self,
        synapse_set: AbstractSynapseSet,
        weight_decay: Union[float, torch.Tensor] = 0.,
        **kwargs
    ) -> None:
        super().__init__(synapse_set=synapse_set)

        self.register_buffer("weight_decay", torch.tensor(weight_decay))

    
    def compute_dw(self, **args) -> None:
        return self.synapse_set.dendrite_set.w *= -self.weight_decay




class STDP(AbstractWeightLR):
    """
    Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of STDP learning rule.You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        synapse_set: AbstractSynapseSet,
        pre_traces: AbstractSpikeTrace = None,
        post_traces: AbstractSpikeTrace = None,
        ltp_wdlr: Callable = constant_wdlr(.1), #LTP weight dependent learning rate
        ltd_wdlr: Callable = constant_wdlr(.1), #LTD weight dependent learning rate
        **kwargs
    ) -> None:
        super().__init__(synapse_set=AbstractSynapseSet)
        self.pre_traces = pre_traces if pre_traces is not None else STDPSpikeTrace()
        self.post_traces = post_traces if post_traces is not None else STDPSpikeTrace()
        self.pre_traces.set_shape(self.synapse_set.axon_set.population_shape)
        self.post_traces.set_shape(self.synapse_set.dendrite_set.population_shape)
        self.ltp_wdlr = ltp_wdlr
        self.ltd_wdlr = ltd_wdlr


    def compute_lrs(self) -> tuple: # ltp_lr,ltd_lr
        w = self.synapse_set.dendrite_set.w
        wmin = self.synapse_set.dendrite_set.wmin
        wmax = self.synapse_set.dendrite_set.wmax
        ltp_lr = self.ltd_wdlr(w, wmin, wmax)
        ltd_lr = self.ltd_wdlr(w, wmin, wmax)
        return ltp_lr,ltd_lr


    def compute_dw(self) -> torch.Tensor:
        ltp_lr,ltd_lr = compute_lrs()
        ltp = ltp_lr * self.pre_traces.traces() * self.synapse_set.dendrite_set.s
        ltd = ltd_lr * self.post_traces.traces() * self.synapse_set.axon_set.s
        dw = self.dt * (ltp - ltd)
        return dw


    def forward(self, neuromodulators: torch.Tensor = None) -> None:
        self.pre_traces.forward(self.synapse_set.axon_set.s)
        self.pre_traces.forward(self.synapse_set.dendrite_set.s)
        super().forward(neuromodulators=neuromodulators)




class FlatSTDP(STDP):
    def __init__(
        self,
        synapse_set: AbstractSynapseSet,
        pre_traces: AbstractSpikeTrace = None,
        post_traces: AbstractSpikeTrace = None,
        **kwargs
    ) -> None:
        super().__init__(
            synapse_set=AbstractSynapseSet,
            pre_traces: AdditiveSpikeTrace(),
            post_traces: AdditiveSpikeTrace(),
        )




# class RSTDP(LearningRule):
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
