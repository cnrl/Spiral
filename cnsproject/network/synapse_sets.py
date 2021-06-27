"""
Module for connections between neural populations.
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Iterable
import torch
from .connectivity_patterns import dense_connectivity
from .axon_sets import AbstractAxonSet
from .dendrite_sets import AbstractDendriteSet


class AbstractSynapseSet(ABC, torch.nn.Module):
    def __init__(
        self,
        name: str = None,
        axon: AbstractAxonSet = None,
        dendrite: AbstractDendriteSet = None,
        dt: float = None,
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__()

        self.axon = None
        self.dendrite = None
        self.dt = None
        self.configed = False
        self.name = None
        self.set_name(name)
        self.config_prohibit = config_prohibit
        self.set_dt(dt)
        self.set_axon(axon)
        self.set_dendrite(dendrite)


    def config_permit(self):
        return (
            self.axon is not None and
            self.dendrite is not None and
            self.dt is not None and
            not self.config_prohibit and
            not self.configed
        )


    def config(self) -> bool:
        if not self.config_permit():
            return False
        assert self.axon.configed, "the axon is not configed yet. you can not config a synapse using it."
        self.dendrite.set_terminal_shape(self.axon.shape)
        assert self.dendrite.configed, "the dendrite is not configed yet. you can not config a synapse using it."
        self.dendrite.set_dt(self.dt)
        self.configed = True
        self.set_name()
        return True


    def set_name(self, name: str = None) -> None:
        if self.name is None:
            self.name = name
        if self.name is None and self.configed:
            self.name = self.axon.name+"-"+self.dendrite.name


    def set_axon(self, axon: AbstractAxonSet):
        self.axon = axon
        self.config()


    def set_dendrite(self, dendrite: AbstractDendriteSet):
        self.dendrite = dendrite
        self.config()


    def set_dt(self, dt:float):
        self.dt = torch.tensor(dt) if dt is not None else dt
        self.config()


    @abstractmethod
    def forward(self, mask: torch.Tensor = torch.tensor(True)) -> None:
        pass


    def reset(self) -> None:
        self.dendrite.reset()


    def __str__(self):
        if self.configed:
            return f"{self.axon.__str__()} -> {self.name} -> {self.dendrite.__str__()}"
        else:
            return f"{self.axon.__str__()} -> {self.name}(X) -> {self.dendrite.__str__()}"




class SimpleSynapseSet(AbstractSynapseSet):
    def __init__(
        self,
        name: str = None,
        connectivity: torch.Tensor = dense_connectivity(), # in shape (*self.axon.shape, self.dendrite.population_shape)
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__(config_prohibit=True, name=name, **kwargs)
        self.connectivity_func = connectivity
        self.config_prohibit = config_prohibit
        self.config()


    def config(self) -> bool:
        if not super().config():
            return False

        assert self.axon.shape==self.dendrite.terminal_shape, \
            "the shape of Axon output (population * terminal shape) must match the shape of Dendrite terminal shape\n"+\
            f"Axon: (population: {self.axon.population_shape}, terminal: {self.axon.terminal_shape}) and "+\
            f"Dendrite: (terminal: {self.dendrite.terminal_shape}, population: {self.dendrite.population_shape})"

        connectivity = self.connectivity_func(
            self.dendrite.terminal_shape,
            self.dendrite.population_shape
        )

        assert connectivity.shape==self.dendrite.shape, \
            "the shape of connectivity must match the shape of Dendrite\n"+\
            f"connectivity: {connectivity.shape}, Dendrite: {self.dendrite.shape}"

        self.register_buffer("connectivity", connectivity)
        return True


    def forward(self, mask: torch.Tensor = torch.tensor(True)) -> None:
        e = self.axon.neurotransmitters()
        e = e.reshape((*self.axon.shape, *[1]*len(self.dendrite.population_shape)))
        e = e * self.connectivity
        e *= mask
        self.dendrite.forward(e)



## doesn't work well with learning
class FilterSynapseSet(AbstractSynapseSet):
    def __init__(
        self,
        name: str = None,
        passage: torch.Tensor = torch.tensor(True), # in shape (*self.axon_passage_shape, self.dendrite.population_shape)
        axon_passage: Iterable = None, #list
        dendrite_terminal_passage: Iterable = None, #list
        config_prohibit: bool = False,
        connectivity: torch.Tensor = dense_connectivity(), # in shape (*self.axon.shape, self.dendrite.population_shape)
        **kwargs
    ) -> None:
        super().__init__(config_prohibit=True, name=name, **kwargs)
        self.connectivity_func = connectivity
        self.axon_passage = axon_passage
        self.dendrite_terminal_passage = dendrite_terminal_passage
        self.passage = passage
        self.config_prohibit = config_prohibit
        self.config()

    def config(self) -> bool:
        if not super().config():
            return False
        
        self.axon_passage,self.axon_passage_shape = self.parse_slice_passage(self.axon_passage, self.axon.shape)
        self.dendrite_terminal_passage,self.dendrite_terminal_passage_shape = self.parse_slice_passage(self.dendrite_terminal_passage, self.dendrite.terminal_shape)

        assert self.axon_passage_shape==self.dendrite_terminal_passage_shape, \
            "the shape of Axon output must match the shape of Dendrite terminal shape\n"+\
            f"Axon: {self.axon_passage_shape},"+\
            f"Dendrite: {self.dendrite_terminal_passage_shape}"

        connectivity = self.connectivity_func(
            self.dendrite_terminal_passage_shape,
            self.dendrite.population_shape
        )

        assert connectivity.shape==(*self.dendrite_terminal_passage_shape, *self.dendrite.population_shape), \
            "the shape of connectivity must match the shape of Dendrite\n"+\
            f"connectivity: {connectivity.shape},"+\
            f"Dendrite: {(*self.dendrite_terminal_passage_shape, *self.dendrite.population_shape)}"
        self.register_buffer("connectivity", connectivity)

        assert self.passage.shape==connectivity.shape, \
                "the shape of passage must match the shape of connectivity\n"+\
                f"connectivity: {connectivity.shape},"+\
                f"passage: {self.passage.shape}"
        return True


    def parse_slice_passage(self, passage, source_shape):
        passage = passage if passage is not None else []
        passage = [True]*(len(source_shape)-len(passage)) + passage
        passage = [slice(None, None, None) if p is True else p for p in passage]
        passage_shape = torch.zeros(source_shape)[passage].shape
        return passage,passage_shape


    def forward(self, mask: torch.Tensor = torch.tensor(True)) -> None:
        e = self.axon.neurotransmitters()
        e = e[self.axon_passage]
        e = e.reshape((*self.axon_passage_shape, *[1]*len(self.dendrite.population_shape)))
        e = e * self.connectivity
        e[self.passage] = float("nan")
        out_e = torch.ones(self.dendrite.shape)*float("nan")
        out_e[self.dendrite_terminal_passage] = e
        out_e *= mask
        self.dendrite.forward(out_e)




class LazySynapseSet(AbstractSynapseSet):
    def __init__(
        self,
        name: str = None,
        config_prohibit: bool = False,
        **kwargs
    ) -> None:
        super().__init__(config_prohibit=True, name=name, **kwargs)
        self.config_prohibit = config_prohibit
        self.config()


    def config(self) -> bool:
        if not self.config_permit():
            return False
        assert self.axon.configed, "the axon is not configed yet. you can not config a synapse using it."
        self.dendrite.set_terminal_shape(self.axon.shape)
        self.dendrite.set_dt(self.dt)
        self.configed = True
        self.set_name()
        return True


    def forward(self, mask: torch.Tensor = torch.tensor(True)) -> None:
        e = self.axon.neurotransmitters()
        e *= mask
        self.dendrite.forward(e)


    def reset(self) -> None:
        self.dendrite.reset()


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
