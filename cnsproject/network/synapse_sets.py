"""
Module for connections between neural populations.
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Callable

import torch

from .connectivity_patterns import dense_connectivity
from .axon_sets import AbstractAxonSet
from .dendrite_sets import AbstractDendriteSet


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

        if connectivity is None:
            connectivity = dense_connectivity(
                self.axon_set.shape,
                self.dendrite_set.population_shape
            )
        self.register_buffer("connectivity", connectivity)

        self.filtering = (self.axon_set.shape!=self.dendrite_set.terminal_shape)
        if self.filtering:
            connectivity = connectivity.reshape(torch.tensor(self.axon_set.shape).prod(), -1)
            max_conn_no = connectivity.sum(axis=0).max()
            assert (max_conn_no==torch.tensor(self.dendrite_set.terminal_shape).prod()), \
                "the shape of Axon output (population * terminal shape) must match the shape of Dendrite terminal shape - or - "+\
                "the number of Axon-Dendrite connections per each Dendrite (connectivity sum on Axon axis), "+\
                "must match the Dendrite input capacity (terminal shape)\n"+\
                f"Axon: (population: {self.axon_set.population_shape}, terminal: {self.axon_set.terminal_shape}) and "+\
                f"Dendrite: (terminal: {self.dendrite_set.terminal_shape}, population: {self.dendrite_set.population_shape})\n"+\
                f"max_conn_no: {max_conn_no}, terminal_neuron_no: {torch.tensor(self.dendrite_set.terminal_shape).prod()})"
            self.filter = lambda x: x.\
                                    reshape(torch.tensor(self.axon_set.shape).prod(), -1).\
                                    gather(0, connectivity.argsort(axis=0, descending=True))\
                                    [:max_conn_no].reshape(*self.dendrite_set.terminal_shape, *self.dendrite_set.population_shape)


    def forward(self, mask: torch.Tensor = torch.tensor(True)) -> None:
        e = self.axon_set.get_output()
        e = e.reshape((*self.axon_set.shape, *[1]*len(self.dendrite_set.population_shape)))
        e = e * self.connectivity
        e *= mask
        if self.filtering:
            e = self.filter(e)
        self.dendrite_set.forward(e)



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
