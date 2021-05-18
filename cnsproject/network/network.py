"""
Module for spiking neural network construction and simulation.
"""

from typing import Optional, Dict

import torch

from .neural_populations import AbstractNeuralPopulation
from .synapse_sets import AbstractSynapseSet
# from ..learning.rewards import AbstractReward
from ..decision.decision import AbstractDecision
from ..learning.learning_rulers import AbstractLearningRuler


class Network(torch.nn.Module):
    """
    The class responsible for creating a neural network and its simulation.

    Examples
    --------
    >>> from network.neural_populations import LIFPopulation
    >>> from network.connections import DenseConnection
    >>> from network.monitors import Monitor
    >>> from network import Network
    >>> inp = InputPopulation(shape=(10,))
    >>> out = LIFPopulation(shape=(2,))
    >>> synapse = DenseConnection(inp, out)
    >>> net = Network(learning=False)
    >>> net.add_layer(inp, "input")
    >>> net.add_layer(out, "output")
    >>> net.add_connection(synapse, "input", "output")
    >>> out_m = Monitor(out, state_variables=["s", "v"])
    >>> syn_m = Monitor(synapse, state_variables=["w"])  # `w` indicates synaptic weights
    >>> net.add_monitor(out_m, "output")
    >>> net.add_monitor(syn_m, "synapse")
    >>> net.run(10)
    Here, we create a simple network with two layers and dense connection. We aim to monitor
    the synaptic weights and output layer's spikes and voltages. We simulate the network for
    10 miliseconds.

    You will need to implement the `run` method. This mthod is responsible for the whole simulation \
    procedure of a spiking neural network. You will have to compute number of time steps using \
    `dt` attribute of the class and `time` parameter of the method. then you will iteratively call \
    the procedures for single step simulation of network objects.

    **NOTE:** If you faced any errors related to importing packages, modify the `__init__.py` files \
    accordingly to solve the problem.

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
        # reward: Optional[AbstractReward] = None,
        # decision: Optional[AbstractDecision] = None,
        **kwargs
    ) -> None:
        super().__init__()

        self.dt = dt

        self.populations = {}
        self.synapses = {}
        self.learners = {}

        self.train(learning)

        # Make sure that arguments of your reward and decision classes do not
        # share same names. Their arguments are passed to the network as its
        # keyword arguments.
        # if reward is not None:
        #     self.reward = reward(**kwargs)
        # if decision is not None:
        #     self.decision = decision(**kwargs)


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


    def add_population(self, population: AbstractNeuralPopulation, name: str) -> None:
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
        self.populations[name] = population
        self.add_module(name, population)
        population.set_dt(self.dt)


    def add_synapse(
        self,
        synapse: AbstractSynapseSet,
        name: str,
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
        self.synapses[name] = synapse
        self.add_module(name, synapse)
        synapse.set_dt(self.dt)


    def add_LR(
        self,
        learner: AbstractLearningRuler,
        name: str,
    ) -> None:
        self.learners[name] = learner
        self.add_module(name, learner)
        learner.set_dt(self.dt)


    def forward(
        self,
        **kwargs
    ) -> None:
        """
        Simulate network for a specific time duration with the possible given\
        input.

        Input to each layer is given to `inputs` parameter. As you see, it is a \
        dictionary of population's name and tensor of input values through time. \
        There is a parameter named `one_step`. This parameter will define how the \
        input is propagated through the network: does it go forward up to the final \
        layer in one time step or it passes from one layer to the next in each \
        step of simulation. You can easily remove it if it is mind-bugling.

        Also, make sure to call `self.reset_state_variables()` before starting the \
        simulation.

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

        **Note:** you can pass the reward and decision methods' arguments as keyword\
        arguments to this function.

        Returns
        -------
        None

        """
        
        for name,synapse in self.synapses.items():
            synapse.forward(kwargs.get(name+"_mask", torch.tensor(True)))
        
        for name,population in self.populations.items():
            direct_input = kwargs.get(name+"_direct_input", torch.tensor(0))
            clamp = kwargs.get(name+"_clamp", torch.tensor(False))
            unclamp = kwargs.get(name+"_unclamp", torch.tensor(False))
            population.forward(direct_input=direct_input, clamps=clamp, unclamps=unclamp)

        if self.learning:
            for name,population in self.populations.items():
                population.backward()

            for name,learner in self.learners.items():
                learner.forward()


    def encode(self, data: dict) -> None:
        for key,value in data.items():
            self.populations[key].encode(value)


    def backward(
        self,
        **kwargs
    ) -> None:
        for name,learner in self.learners.items():
            learner.backward()


    def run(self, **kwargs):
        self.forward(**kwargs)
        if self.learning:
            self.backward(**kwargs)


    def reset(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        for population in self.populations.values():
            population.reset()

        for synapse in self.synapses.values():
            synapse.reset()
