"""
Module for spiking neural network construction and simulation.
"""

from typing import Optional, Dict

import torch

from .neural_populations import NeuralPopulation
from .connections import AbstractConnection
from .monitors import Monitor
from ..learning.rewards import AbstractReward
from ..decision.decision import AbstractDecision


class Network(torch.nn.Module):
    """
    The class responsible for creating a neural network and its simulation.

    Arguments
    ---------
    dt : float, Optional
        Specify simulation timestep. The default is 1.0.
    learning: bool, Optional
        Whether to allow weight update and learning. The default is True.
    reward : AbstractReward, Optional
        The class to allow reward modifications in case of reward-modulated
        learning. The default is None.
    decision: AbstractDecision, Optional
        The class to enable decision making. The default is None.

    """

    def __init__(
        self,
        dt: float = 1.0,
        learning: bool = True,
        reward: Optional[AbstractReward] = None,
        decision: Optional[AbstractDecision] = None,
        **kwargs
    ) -> None:
        super().__init__()

        self.dt = dt

        self.layers = {}
        self.connections = {}
        self.monitors = {}

        self.train(learning)

        self.reward = reward(**kwargs)
        self.decision = decision(**kwargs)

    def add_layer(self, layer: NeuralPopulation, name: str) -> None:
        """
        Add a neural population to the network.

        Parameters
        ----------
        layer : NeuralPopulation
            The neural population to be added.
        name : str
            Name of the layer for further referencing.

        Returns
        -------
        None

        """
        self.layers[name] = layer
        self.add_module(name, layer)

        layer.train(self.learning)
        layer.dt = self.dt

    def add_connection(
        self,
        connection: AbstractConnection,
        pre: str,
        post: str
    ) -> None:
        """
        Add a connection between neural populations to the network. The\
        reference name will be in the format `{pre}_to_{post}`.

        Parameters
        ----------
        connection : AbstractConnection
            The connection to be added.
        pre : str
            Reference name of pre-synaptic population.
        post : str
            Reference name of post-synaptic population.

        Returns
        -------
        None

        """
        self.connections[f"{pre}_to_{post}"] = connection
        self.add_module(f"{pre}_to_{post}", connection)

        connection.train(self.learning)
        connection.dt = self.dt

    def add_monitor(self, monitor: Monitor, name: str) -> None:
        """
        Add a monitor on a network object to the network.

        Parameters
        ----------
        monitor : Monitor
            The monitor instance to be added.
        name : str
            Name of the monitor instance for further referencing.

        Returns
        -------
        None

        """
        self.monitors[name] = monitor
        monitor.dt = self.dt

    def run(
        self,
        time: int,
        inputs: Dict[str, torch.Tensor] = {},
        one_step: bool = False,
        **kwargs
    ) -> None:
        """
        Simulate network for a specific time duration with the possible given\
        input.

        TODO.

        Implement the body of this method.

        Parameters
        ----------
        time : int
            Simulation time.
        inputs : Dict[str, torch.Tensor], optional
            Mapping of input layer names to their input spike tensors. The\
            default is {}.
        one_step : bool, optional
            Whether to propagate the inputs all the way through the network in\
            a single simulation step. The default is False.

        Keyword Arguments
        -----------------
        clamp : Dict[str, torch.Tensor]
            Mapping of layer names to boolean masks if neurons should be clamped
            to spiking.
        unclamp : Dict[str, torch.Tensor]
            Mapping of layer names to boolean masks if neurons should be clamped
            not to spiking.
        masks : Dict[str, torch.Tensor]
            Mapping of connection names to boolean masks of the weights to clamp
            to zero.

        Note: you can pass the reward and decision arguments as keyword\
        argumeents to this function.

        Returns
        -------
        None

        """
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})

    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        for layer in self.layers:
            self.layers[layer].reset_state_variables()

        for connection in self.connections:
            self.connections[connection].reset_state_variables()

        for monitor in self.monitors:
            self.monitors[monitor].reset_state_variables()

    def train(self, mode: bool = True) -> "torch.nn.Moudle":
        """
        Set the population's training mode.

        Parameters
        ----------
        mode : bool, optional
            Mode of training. `True` turns on the training while `False` turns\
            it off. The default is True.

        Returns
        -------
        torch.nn.Module

        """
        self.learning = mode
        return super().train(mode)