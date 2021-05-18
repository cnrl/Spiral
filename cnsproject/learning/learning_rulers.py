"""
Module for learning rules.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Sequence, Callable, Iterable

import numpy as np
import torch

from ..network.synapse_sets import AbstractSynapseSet
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


    def reset(self) -> None:
        pass




class SerializedLR(AbstractLearningRuler):
    def __init__(
        self,
        LRs: Iterable[AbstractLearningRuler],
        **kwargs
    ) -> None:
        super().__init__(synapse_set=LRs[0].synapse_set, **kwargs)
        self.LRs = list(LRs)


    def set_dt(self, dt:float):
        if dt is not None:
            [lr.set_dt(dt) for lr in self.LRs]

    
    def forward(self, neuromodulators: torch.Tensor = None) -> None:
        [lr.forward(neuromodulators) for lr in self.LRs]

    
    def backward(self, **args) -> None:
        [lr.backward(**args) for lr in self.LRs]


    def reset(self) -> None:
        [lr.reset() for lr in self.LRs]


    @classmethod
    def __add__(self, other: Union[AbstractLearningRuler]):
        if type(other) is SerializedLR:
            return SerializedLR(self.LRs+other.LRs)
        else:
            return SerializedLR(self.LRs+[other])




class SerializableLR(AbstractLearningRuler):
    def __init__(
        self,
        synapse_set: AbstractSynapseSet,
        **kwargs
    ) -> None:
        super().__init__(synapse_set=synapse_set, **kwargs)


    def __add__(self, other: Union[AbstractLearningRuler]) -> SerializedLR:
        if type(other) is SerializedLR:
            return SerializedLR([self]+other.LRs)
        else:
            return SerializedLR([self, other])




class NoOp(SerializableLR):
    def __init__(
        self,
        synapse_set: AbstractSynapseSet,
        **kwargs
    ) -> None:
        super().__init__(synapse_set=synapse_set)




class AbstractWeightLR(SerializableLR):
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
        super().__init__(synapse_set=synapse_set)


    def compute_dw(self) -> torch.Tensor:
        pass


    def update_w(self, dw: torch.Tensor) -> None:
        w = self.synapse_set.dendrite_set.w
        wmin = self.synapse_set.dendrite_set.wmin
        wmax = self.synapse_set.dendrite_set.wmax
        w += dw
        w[w<wmin] = wmin
        w[w>wmax] = wmax

    
    def forward(self, neuromodulators: torch.Tensor = None) -> None:
        pass


    def backward(self, **args) -> None:
        self.update_w(self.compute_dw())




class SimpleWeightDecayLR(AbstractWeightLR):
    def __init__(
        self,
        synapse_set: AbstractSynapseSet,
        decay: Union[float, torch.Tensor] = 0.,
        **kwargs
    ) -> None:
        super().__init__(synapse_set=synapse_set)

        self.register_buffer("decay", torch.tensor(decay))

    
    def compute_dw(self, **args) -> None:
        return self.synapse_set.dendrite_set.w * -self.decay




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
        dt: float = None,
        **kwargs
    ) -> None:
        super().__init__(synapse_set=synapse_set, dt=None)
        self.pre_traces = pre_traces if pre_traces is not None else STDPSpikeTrace()
        self.post_traces = post_traces if post_traces is not None else STDPSpikeTrace()
        self.pre_traces.set_shape(self.synapse_set.axon_set.population_shape)
        self.post_traces.set_shape(self.synapse_set.dendrite_set.population_shape)
        self.ltp_wdlr = ltp_wdlr
        self.ltd_wdlr = ltd_wdlr
        self.set_dt(dt)


    def set_dt(self, dt:float) -> None:
        super().set_dt(dt)
        if dt is not None:
            self.pre_traces.set_dt(dt)
            self.post_traces.set_dt(dt)


    def compute_lrs(self) -> tuple: # ltp_lr,ltd_lr
        w = self.synapse_set.dendrite_set.w
        wmin = self.synapse_set.dendrite_set.wmin
        wmax = self.synapse_set.dendrite_set.wmax
        ltp_lr = self.ltd_wdlr(w, wmin, wmax)
        ltd_lr = self.ltd_wdlr(w, wmin, wmax)
        return ltp_lr,ltd_lr


    def to_singlton_dendrite_shape(self, tensor: torch.Tensor) -> torch.Tensor:
        return \
            self.synapse_set.dendrite_set.to_singlton_population_shape(
                self.synapse_set.axon_set.to_singlton_terminal_shape(
                    tensor
                )
            )


    def compute_dw(self) -> torch.Tensor:
        ltp_lr,ltd_lr = self.compute_lrs()
        ltp = ltp_lr * self.to_singlton_dendrite_shape(self.pre_traces.traces()) * self.synapse_set.dendrite_set.spikes()
        ltd = ltd_lr * self.post_traces.traces() * self.to_singlton_dendrite_shape(self.synapse_set.axon_set.spikes())
        dw = self.dt * (ltp - ltd)
        return dw


    def forward(self, neuromodulators: torch.Tensor = None) -> None:
        self.pre_traces.forward(self.synapse_set.axon_set.spikes())
        self.post_traces.forward(self.synapse_set.dendrite_set.spikes())
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
            synapse_set=synapse_set,
            pre_traces=AdditiveSpikeTrace(),
            post_traces=AdditiveSpikeTrace(),
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
