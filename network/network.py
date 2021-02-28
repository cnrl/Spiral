"""
==============================================================================.

CNS2021 PROJECT TEMPLATE

==============================================================================.
                                                                              |
network/network.py                                                            |
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

from typing import Dict

import torch

from .neural_populations import NeuralPopulation
from .connections import AbstractConnection


class Network(torch.nn.Module):
    def __init__(
        self,
        dt: float = 1.0,
        learning: bool = True,
    ) -> None:
        super().__init__()

        self.dt = dt

        self.layers = {}
        self.connections = {}
        self.monitors = {}

        self.train(learning)

    def add_layer(self, layer: NeuralPopulation, name: str) -> None:
        self.layers[name] = layer
        self.add_module(name, layer)

        layer.train(self.learning)
        layer.dt = self.dt

    def add_connection(
        self,
        connection: AbstractConnection,
        pre: str,
        post: str,
    ) -> None:
        self.connections[f"{pre}_to_{post}"] = connection
        self.add_module(f"{pre}_to_{post}", connection)

        connection.train(self.learning)
        connection.dt = self.dt

    def run(
        self,
        time: int,
        inputs: Dict[str, torch.Tensor] = {},
        one_step: bool = False,
        **kwargs
    ) -> None:
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})

        #

    def reset_state_variables(self) -> None:
        for layer in self.layers:
            self.layers[layer].reset_state_variables()

        for connection in self.connections:
            self.connections[connection].reset_state_variables()

        for monitor in self.monitors:
            self.monitors[monitor].reset_state_variables()

    def train(self, mode: bool = True) -> "torch.nn.Moudle":
        self.learning = mode
        return super().train(mode)
