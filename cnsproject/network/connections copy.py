"""
Module for connections between neural populations.
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Callable

import torch

from .connectivity_patterns import dense_connectivity,constant_weights


class AbstractConnection(ABC, torch.nn.Module):
    """
    Abstract class for implementing connections.

    Make sure to implement the `compute`, `update`, and `reset_state_variables`\
    methods in your child class.

    You will need to define the populations you want to connect as `pre` and `post`.\
    In case of learning, you will need to define the learning rate (`lr`) and the \
    learning rule to follow. Attribute `w` is reserved for synaptic weights.\
    However, it has not been predefined or allocated, as it depends on the \
    pattern of connectivity. So make sure to define it in child class initializations \
    appropriately to indicate the pattern of connectivity. The default range of \
    each synaptic weight is [0, 1] but it can be controlled by `wmin` and `wmax`. \
    Synaptic strengths might decay in time and do not last forever. To define \
    the decay rate of the synaptic weights, use `weight_decay` attribute. Also, \
    if you want to control the overall input synaptic strength to each neuron, \
    use `norm` argument to normalize the synaptic weights.

    In case of learning, you have to implement the methods `compute` and `update`. \
    You will use the `compute` method to calculate the activity of post-synaptic \
    population based on the pre-synaptic one. Update of weights based on the \
    learning rule will be implemented in the `update` method. If you find this \
    architecture mind-bugling, try your own architecture and make sure to redefine \
    the learning rule architecture to be compatible with this new architecture \
    of yours.

    Arguments
    ---------
    pre : NeuralPopulation
        The pre-synaptic neural population.
    post : NeuralPopulation
        The post-synaptic neural population.
    lr : float or (float, float), Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float, Optional
        Define rate of decay in synaptic strength. The default is 0.0.

    Keyword Arguments
    -----------------
    learning_rule : LearningRule
        Define the learning rule by which the network will be trained. The\
        default is NoOp (see learning/learning_rules.py for more details).
    wmin : float
        The minimum possible synaptic strength. The default is 0.0.
    wmax : float
        The maximum possible synaptic strength. The default is 1.0.
    norm : float
        Define a normalization on input signals to a population. If `None`,\
        there is no normalization. The default is None.

    """

    def __init__(
        self,
        pre: Iterable[int] = None,
        post: Iterable[int] = None,
        wmin: Union[float, torch.Tensor] = 0.,
        wmax: Union[float, torch.Tensor] = 1.,
        scale: Union[float, torch.Tensor] = 1.,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        dt: float = None,
        **kwargs
    ) -> None:
        super().__init__()

        self.pre = pre
        self.post = post
        self.register_buffer("wmin", torch.tensor(wmin))
        self.register_buffer("wmax", torch.tensor(wmax))
        self.register_buffer("scale", torch.tensor(scale))

        # self.weight_decay = weight_decay

        # self.norm = kwargs.get('norm', None)

        # from ..learning.learning_rules import NoOp

        # learning_rule = kwargs.get('learning_rule', NoOp)

        # self.learning_rule = learning_rule(
        #     connection=self,
        #     lr=lr,
        #     weight_decay=weight_decay,
        #     **kwargs
        # )
        self.register_buffer("I", torch.zeros(*self.post.shape)) #mA
        self.set_dt(dt)

    def set_dt(self, dt:float):
        self.dt = torch.tensor(dt) if dt is not None else dt

    @abstractmethod
    def forward(self, s: torch.Tensor) -> None: #s: spike
        """
        Compute the post-synaptic neural population activity based on the given\
        spikes of the pre-synaptic population.

        Parameters
        ----------
        s : torch.Tensor
            The pre-synaptic spikes tensor.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Compute connection's learning rule and weight update.

        Keyword Arguments
        -----------------
        learning : bool
            Whether learning is enabled or not. The default is True.
        mask : torch.ByteTensor
            Define a mask to determine which weights to clamp to zero.

        Returns
        -------
        None

        """
        learning = kwargs.get("learning", True)

        if learning:
            self.learning_rule.update(**kwargs)

        mask = kwargs.get("mask", None)
        if mask is not None:
            self.w.masked_fill_(mask, 0)

    @abstractmethod
    def reset(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        pass

    def output(self):
        return self.I * self.scale



class SimpleConnection(AbstractConnection):
    def __init__(
        self,
        connectivity: Callable = dense_connectivity,
        w: Callable = constant_weights,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.shape = (*self.pre, *self.post)
        self.register_buffer("connectivity", connectivity(self, **kwargs))
        self.register_buffer("w", w(self, **kwargs))
        self.w /= self.connectivity.sum(axis=[i for i in range(len(self.pre))])
        self.w *= self.connectivity
        self.w[self.w<self.wmin] = wmin
        self.w[self.w>self.wmax] = wmax

    def forward(self, s: torch.Tensor) -> None:
        I = s.reshape(*self.pre, *[1 for i in self.post])
        I *= self.w
        I = I.sum(axis=[i for i in range(len(self.pre))])
        self.I = I

    def update(self, **kwargs) -> None:
        super().update(**Keyword)

    def reset(self) -> None:
        self.I.zero_()


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
