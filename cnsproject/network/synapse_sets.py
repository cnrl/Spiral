"""
Module for connections between neural populations.
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Iterable

import torch

from .connectivity_patterns import dense_connectivity
from .axon_sets import AbstractAxonSet
from .dendrite_sets import AbstractDendriteSet
from ..utils import SliceMaker


class AbstractSynapseSet(ABC, torch.nn.Module):
    def __init__(
        self,
        axon_set: AbstractAxonSet,
        dendrite_set: AbstractDendriteSet,
        dt: float = None,
        **kwargs
    ) -> None:
        super().__init__()

        self.axon_set = axon_set
        self.dendrite_set = dendrite_set
        self.set_dt(dt)


    def set_dt(self, dt:float):
        self.dt = torch.tensor(dt) if dt is not None else dt
        self.axon_set.set_dt(dt)
        self.dendrite_set.set_dt(dt)


    @abstractmethod
    def forward(self, mask: torch.Tensor = torch.tensor(True)) -> None: #s: spike
        pass

    def reset(self) -> None:
        self.dendrite_set.reset()




class SimpleSynapseSet(AbstractSynapseSet):
    def __init__(
        self,
        connectivity: torch.Tensor = None, # in shape (*self.axon_set.shape, self.dendrite_set.population_shape)
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        assert self.axon_set.shape==self.dendrite_set.terminal_shape, \
            "the shape of Axon output (population * terminal shape) must match the shape of Dendrite terminal shape\n"+\
            f"Axon: (population: {self.axon_set.population_shape}, terminal: {self.axon_set.terminal_shape}) and "+\
            f"Dendrite: (terminal: {self.dendrite_set.terminal_shape}, population: {self.dendrite_set.population_shape})"

        if connectivity is None:
            connectivity = dense_connectivity(
                self.dendrite_set.terminal_shape,
                self.dendrite_set.population_shape
            )
        else:
            assert connectivity.shape==self.dendrite_set.shape, \
                "the shape of connectivity must match the shape of Dendrite\n"+\
                f"connectivity: {connectivity.shape}, Dendrite: {self.dendrite_set.shape}"

        self.register_buffer("connectivity", connectivity)



    def forward(self, mask: torch.Tensor = torch.tensor(True)) -> None:
        e = self.axon_set.get_output()
        e = e.reshape((*self.axon_set.shape, *[1]*len(self.dendrite_set.population_shape)))
        e = e * self.connectivity
        e *= mask
        self.dendrite_set.forward(e)




class FilterSynapseSet(AbstractSynapseSet):
    def __init__(
        self,
        connectivity: torch.Tensor = None, # in shape (*self.axon_passage_shape, self.dendrite_set.population_shape)
        passage: torch.Tensor = torch.tensor(True), # in shape (*self.axon_passage_shape, self.dendrite_set.population_shape)
        axon_passage: Iterable = None, #list
        dendrite_terminal_passage: Iterable = None, #list
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.axon_passage,self.axon_passage_shape = self.parse_slice_passage(axon_passage, self.axon_set.shape)
        self.dendrite_terminal_passage,self.dendrite_terminal_passage_shape = self.parse_slice_passage(dendrite_terminal_passage, self.dendrite_set.terminal_shape)

        assert self.axon_passage_shape==self.dendrite_terminal_passage_shape, \
            "the shape of Axon output must match the shape of Dendrite terminal shape\n"+\
            f"Axon: {self.axon_passage_shape},"+\
            f"Dendrite: {self.dendrite_terminal_passage_shape}"

        if connectivity is None:
            connectivity = dense_connectivity(
                self.dendrite_terminal_passage_shape,
                self.dendrite_set.population_shape
            )
        else:
            assert connectivity.shape==(*self.dendrite_terminal_passage_shape, *self.dendrite_set.population_shape), \
                "the shape of connectivity must match the shape of Dendrite\n"+\
                f"connectivity: {connectivity.shape},"+\
                f"Dendrite: {(*self.dendrite_terminal_passage_shape, *self.dendrite_set.population_shape)}"
        self.register_buffer("connectivity", connectivity)

        assert passage.shape==connectivity.shape, \
                "the shape of passage must match the shape of connectivity\n"+\
                f"connectivity: {connectivity.shape},"+\
                f"passage: {passage.shape}"
        self.passage = passage


    def parse_slice_passage(self, passage, source_shape):
        passage = passage if passage is not None else []
        passage = [True]*(len(source_shape)-len(passage)) + passage
        passage = [SliceMaker()[:] if p is True else p for p in passage]
        passage_shape = torch.zeros(source_shape)[passage].shape
        return passage,passage_shape

    def forward(self, mask: torch.Tensor = torch.tensor(True)) -> None:
        e = self.axon_set.get_output()
        e = e[self.axon_passage]
        e = e.reshape((*self.axon_passage_shape, *[1]*len(self.dendrite_set.population_shape)))
        e = e * self.connectivity
        e[self.passage] = float("nan")
        out_e = torch.ones(self.dendrite_set.shape)*float("nan")
        out_e[self.dendrite_terminal_passage] = e
        out_e *= mask
        self.dendrite_set.forward(out_e)


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
