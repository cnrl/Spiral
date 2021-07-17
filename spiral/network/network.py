"""
Module for spiking neural network construction and simulation.
"""

from typing import Optional, Dict

import torch

from typing import Union, Iterable
from .neural_populations import AbstractNeuralPopulation
from .synapse_sets import AbstractSynapseSet
from ..decision.decision import AbstractDecision
from ..learning.learning_rule_enforcers import AbstractLearningRuleEnforcer
from .neuromodulatory_tissues import AbstractNeuromodulatoryTissue


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

        self.NPs = {} #neural populations
        self.SSs = {} #synapse sets
        self.LREs = {} #learning rule enforcers
        self.NTs = {} #neuromodulatory_tissues
        # The network is responsible for calculating
        # the delay for delivering the neuromodulators to the connections

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


    def add_population(
        self,
        population: AbstractNeuralPopulation
    ) -> None:
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
        population.set_dt(self.dt)
        name = population.name
        self.NPs[name] = population
        self.add_module(name, population)


    def add_synapse(
        self,
        synapse: AbstractSynapseSet,
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
        synapse.set_dt(self.dt)
        name = synapse.name
        self.SSs[name] = synapse
        self.add_module(name, synapse)


    def add_learning_rule_encoder(
        self,
        learner: AbstractLearningRuleEnforcer,
    ) -> None:
        learner.set_dt(self.dt)
        name = learner.name
        self.LREs[name] = learner
        self.add_module(name, learner)


    def add_neuromodulatory_tissue(
        self,
        tissue: AbstractLearningRuleEnforcer,
    ) -> None:
        tissue.set_dt(self.dt)
        name = tissue.name
        self.NTs[name] = tissue
        self.add_module(name, tissue)


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
        
        for name,synapse in self.SSs.items():
            synapse.forward(kwargs.get(name+"_mask", torch.tensor(True)))
        
        for name,population in self.NPs.items():
            direct_input = kwargs.get(name+"_direct_input", torch.tensor(0))
            clamp = kwargs.get(name+"_clamp", torch.tensor(False))
            unclamp = kwargs.get(name+"_unclamp", torch.tensor(False))
            population.forward(direct_input=direct_input, clamps=clamp, unclamps=unclamp)

        if self.learning:
            for name,population in self.NPs.items():
                population.backward()

            for name,learner in self.LREs.items():
                learner.forward()

            for name,tissue in self.NTs.items():
                tissue.forward()


    def encode(self, data: dict) -> None:
        for key,value in data.items():
            self.NPs[key].encode(value)


    def feedback(self, data: dict) -> None:
        for key,value in data.items():
            self.NTs[key].feedback(value)


    def backward(
        self,
        **kwargs
    ) -> None:
        for name,learner in self.LREs.items():
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
        for a in self.NPs.values():
            a.reset()

        for a in self.NTs.values():
            a.reset()

        for a in self.SSs.values():
            a.reset()

        for a in self.LREs.values():
            a.reset()


    def __iadd__(self, other: Union[Iterable, AbstractNeuralPopulation, AbstractSynapseSet, AbstractLearningRuleEnforcer]):
        if hasattr(other, '__iter__'):
            for o in other:
                self.__iadd__(o)
        elif issubclass(type(other), AbstractNeuralPopulation):
            self.add_population(other)
        elif issubclass(type(other), AbstractSynapseSet):
            self.add_synapse(other)
        elif issubclass(type(other), AbstractLearningRuleEnforcer):
            self.add_learning_rule_encoder(other)
        elif issubclass(type(other), AbstractNeuromodulatoryTissue):
            self.add_neuromodulatory_tissue(other)
        else:
            assert False, f"You just can add AbstractNeuralPopulation, AbstractNeuromodulatoryTissue, AbstractSynapseSet or AbstractLearningRuleEnforcer to network. Your object is {type(other)}"
        return self


    def __getitem__(self, name: Union[Iterable, str], level=0):
        if type(name) is str:
            return [self.NPs, self.NTs, self.SSs, self.LREs][level][name]
        else:
            assert hasattr(name, '__iter__'), f"getitem input must be a string or a list of strings"
            assert len(name)==1, f"Length of name list ({name}) must be 1."
            return self.__getitem__(name[0], level+1)


    def __str__(self):
        string = "="*40+" Neural Populations:\n"
        for a in self.NPs.values():
            string += a.__str__()+'\n'
        string += "="*40+" Synapse Sets:\n"
        for a in self.SSs.values():
            string += a.__str__()+'\n'
        string += "="*40+" Learning Rule Enforcers:\n"
        for a in self.LREs.values():
            string += a.__str__()+'\n'
        string += "="*40+" Neuromodulatory Tissues:\n"
        for a in self.NTs.values():
            string += a.__str__()+'\n'
        return string