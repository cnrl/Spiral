"""
==============================================================================.

CNS2021 PROJECT TEMPLATE

==============================================================================.
                                                                              |
network/connections.py                                                        |
                                                                              |
Copyright (C) 2020-2021 CNRL <cnrl.ut.ac.ir>                                  |
                                                                              |
This program is free software:  you can redistribute it and/or modify it under|
the terms of the  GNU General Public License as published by the Free Software|
Foundation, either version 3 of the License, or(at your option) any later ver-|
sion.                                                                         |
                                                                              |
This file has been provided for educational purpose.  The aim of this template|
is to help students with developing a Spiking Neural Network framework from s-|
cratch and learn the basics. Follow the documents and comments to complete the|
code.                                                                         |
                                                                              |
==============================================================================.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Sequence

import torch

from .neural_populations import NeuralPopulation
from ..learning.learning_rules import LearningRule, NoOp


class AbstractConnection(ABC, torch.nn.Module):
    """
    Abstract class for implementing connections.

    Make sure to implement the `compute`, `update`, and `reset_state_variables`
    methods in your child class.

    Arguments
    ---------
    pre : NeuralPopulation
        The pre-synaptic neural population.
    post : NeuralPopulation
        The post-synaptic neural population.
    learning_rule : LearningRule, Optional
        Define the learning rule by which the network will be trained. The def-
        ault is NoOp (see learning/learning_rules.py for more details).
    lr : float or (float, float), Optional
        The learning rate for training procedure. If a tuple is given, the fir-
        st value defines potentiation learning rate and the second one depicts
        the depression learning rate. The default is None.
    weight_decay : float, Optional
        Define rate of decay in synaptic strength. The default is 0.0.

    Keyword Arguments
    -----------------
    wmin : float
        The minimum possible synaptic strength. The default is 0.0.
    wmax : float
        The maximum possible synaptic strength. The default is 1.0.
    norm : float
        Define a normalization on input signals to a population. If `None`, th-
        ere is no normalization. The default is None.

    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        learning_rule: LearningRule = NoOp,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__()

        assert isinstance(pre, NeuralPopulation), \
            "Pre is not a NeuralPopulation instance"
        assert isinstance(post, NeuralPopulation), \
            "Post is not a NeuralPopulation instance"

        self.pre = pre
        self.post = post

        self.weight_decay = weight_decay

        self.learning_rule = learning_rule(
            connection=self,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.wmin = kwargs.get('wmin', 0.)
        self.wmax = kwargs.get('wmax', 1.)
        self.norm = kwargs.get('norm', None)

    @abstractmethod
    def compute(self, s: torch.Tensor) -> None:
        """
        Compute the pre-synaptic neural population activity based on the given\
        spikes of the post-synaptic population.

        Parameters
        ----------
        s : torch.Tensor
            The post-synaptic spikes tensor.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Compute connection's learning rule.

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
    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        pass


class DenseConnection(AbstractConnection):
    """
    Specify a fully-connected synapse between neural populations.

    Implement the dense connection pattern following the abstract connection t-
    emplate.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        learning_rule: LearningRule = NoOp,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            learning_rule=learning_rule,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        1. Add more parameters if needed.
        2. Fill the body accordingly.
        """

    def compute(self, s: torch.Tensor) -> None:
        """
        TODO.

        Implement the computation of pre-synaptic population activity given the
        activity of the post-synaptic population.
        """
        pass

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.
        You might need to call the parent method.
        """
        pass

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass


class RandomConnection(AbstractConnection):
    """
    Specify a random synaptic connection between neural populations.

    Implement the random connection pattern following the abstract connection
    template.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        learning_rule: LearningRule = NoOp,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            learning_rule=learning_rule,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        1. Add more parameters if needed.
        2. Fill the body accordingly.
        """

    def compute(self, s: torch.Tensor) -> None:
        """
        TODO.

        Implement the computation of pre-synaptic population activity given the
        activity of the post-synaptic population.
        """
        pass

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.
        You might need to call the parent method.
        """
        pass

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass


class ConvolutionalConnection(AbstractConnection):
    """
    Specify a convolutional synaptic connection between neural populations.

    Implement the convolutional connection pattern following the abstract conn-
    ection template.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        learning_rule: LearningRule = NoOp,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            learning_rule=learning_rule,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        1. Add more parameters if needed.
        2. Fill the body accordingly.
        """

    def compute(self, s: torch.Tensor) -> None:
        """
        TODO.

        Implement the computation of pre-synaptic population activity given the
        activity of the post-synaptic population.
        """
        pass

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.
        You might need to call the parent method.
        """
        pass

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass


class PoolingConnection(AbstractConnection):
    """
    Specify a pooling synaptic connection between neural populations.

    Implement the pooling connection pattern following the abstract connection
    template. Consider a parameter for defining the type of pooling.

    Note: The pooling operation does not support learning. You might need to m-
    ake some modifications in the defined structure of this class.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        learning_rule: LearningRule = NoOp,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            learning_rule=learning_rule,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        1. Add more parameters if needed.
        2. Fill the body accordingly.
        """

    def compute(self, s: torch.Tensor) -> None:
        """
        TODO.

        Implement the computation of pre-synaptic population activity given the
        activity of the post-synaptic population.
        """
        pass

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.
        You might need to call the parent method.

        Note: You should be careful with this method.
        """
        pass

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass
