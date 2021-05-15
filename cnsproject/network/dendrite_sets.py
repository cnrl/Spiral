"""
Module for connections between neural populations.
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Callable, Iterable

import torch

from .weight_initializations import constant_initialization

class AbstractDendriteSet(ABC, torch.nn.Module):
    def __init__(
        self,
        terminal: Iterable[int],
        population: Iterable[int],
        wmin: Union[float, torch.Tensor] = 0.,
        wmax: Union[float, torch.Tensor] = 1.,
        dt: float = None,
        **kwargs
    ) -> None:
        super().__init__()

        self.terminal_shape = terminal
        self.population_shape = population
        self.shape = (*self.terminal_shape,*self.population_shape)
        self.register_buffer("wmin", torch.tensor(wmin))
        self.register_buffer("wmax", torch.tensor(wmax))
        self.register_buffer("I", torch.zeros(*self.population_shape)) #mA
        self.set_dt(dt)


    def add_population_shape(self, tensor: torch.Tensor):
        if tensor.numel()==1 or tensor.shape==self.shape:
            return tensor
        else:
            return tensor.reshape((*self.terminal_shape,*[1]*len(self.population_shape)))


    def set_dt(self, dt:float):
        self.dt = torch.tensor(dt) if dt is not None else dt


    @abstractmethod
    def forward(self, e: torch.Tensor) -> None: #e: spike resonse  in shape (*self.terminal_shape,*self.population_shape)
        pass


    def reset(self) -> None:
        self.I.zero_()


    def get_output(self): # in shape *self.population_shape
        return self.I

    # @abstractmethod
    # def update(self, **kwargs) -> None:
    #     """
    #     Compute connection's learning rule and weight update.

    #     Keyword Arguments
    #     -----------------
    #     learning : bool
    #         Whether learning is enabled or not. The default is True.
    #     mask : torch.ByteTensor
    #         Define a mask to determine which weights to clamp to zero.

    #     Returns
    #     -------
    #     None

    #     """
    #     learning = kwargs.get("learning", True)

    #     if learning:
    #         self.learning_rule.update(**kwargs)

    #     mask = kwargs.get("mask", None)
    #     if mask is not None:
    #         self.w.masked_fill_(mask, 0)




class SimpleDendriteSet(AbstractDendriteSet):
    def __init__(
        self,
        w: torch.Tensor = None, # in shape (*self.terminal_shape, *self.population_shape) or *self.population_shape or 1
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if w is None:
            w = constant_initialization(
                self.terminal_shape,
                self.population_shape,
                self.wmin +
                (self.wmax - self.wmin) /
                (2 * torch.prod(torch.tensor(self.terminal_shape)))
            )
        self.register_buffer("w", w)
        self.w[self.w<self.wmin] = self.wmin
        self.w[self.w>self.wmax] = self.wmax

    def forward(self, e: torch.Tensor) -> None:
        self.I = self.add_population_shape(e) * self.w
        self.I = self.I.sum(axis=list(range(len(self.terminal_shape))))





# class ConvolutionalConnection(AbstractConnection):
#     """
#     Specify a convolutional synaptic connection between neural populations.

#     Implement the convolutional connection pattern following the abstract\
#     connection template.
#     """

#     def __init__(
#         self,
#         pre: NeuralPopulation,
#         post: NeuralPopulation,
#         lr: Union[float, Sequence[float]] = None,
#         weight_decay: float = 0.0,
#         **kwargs
#     ) -> None:
#         super().__init__(
#             pre=pre,
#             post=post,
#             lr=lr,
#             weight_decay=weight_decay,
#             **kwargs
#         )
#         """
#         TODO.

#         1. Add more parameters if needed.
#         2. Fill the body accordingly.
#         """

#     def forward(self, s: torch.Tensor) -> None:
#         """
#         TODO.

#         Implement the computation of post-synaptic population activity given the
#         activity of the pre-synaptic population.
#         """
#         pass

#     def update(self, **kwargs) -> None:
#         """
#         TODO.

#         Update the connection weights based on the learning rule computations.
#         You might need to call the parent method.
#         """
#         pass

#     def reset_state_variables(self) -> None:
#         """
#         TODO.

#         Reset all the state variables of the connection.
#         """
#         pass


# class PoolingConnection(AbstractConnection):
#     """
#     Specify a pooling synaptic connection between neural populations.

#     Implement the pooling connection pattern following the abstract connection\
#     template. Consider a parameter for defining the type of pooling.

#     Note: The pooling operation does not support learning. You might need to\
#     make some modifications in the defined structure of this class.
#     """

#     def __init__(
#         self,
#         pre: NeuralPopulation,
#         post: NeuralPopulation,
#         lr: Union[float, Sequence[float]] = None,
#         weight_decay: float = 0.0,
#         **kwargs
#     ) -> None:
#         super().__init__(
#             pre=pre,
#             post=post,
#             lr=lr,
#             weight_decay=weight_decay,
#             **kwargs
#         )
#         """
#         TODO.

#         1. Add more parameters if needed.
#         2. Fill the body accordingly.
#         """

#     def forward(self, s: torch.Tensor) -> None:
#         """
#         TODO.

#         Implement the computation of post-synaptic population activity given the
#         activity of the pre-synaptic population.
#         """
#         pass

#     def update(self, **kwargs) -> None:
#         """
#         TODO.

#         Update the connection weights based on the learning rule computations.\
#         You might need to call the parent method.

#         Note: You should be careful with this method.
#         """
#         pass

#     def reset_state_variables(self) -> None:
#         """
#         TODO.

#         Reset all the state variables of the connection.
#         """
#         pass
