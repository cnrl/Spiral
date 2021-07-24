"""
Spiral network is a container for neurons and synapses to interact with them more easily.
"""


from __future__ import annotations
import torch
from typing import Union, Dict, Any
from typeguard import typechecked
from constant_properties_protector import CPP
from spiral.soma.soma import Soma
from spiral.synapse.synapse import Synapse




@typechecked
class Network(torch.nn.Module):
    """
    The class responsible for keeping neurons and synapses, run them in order, manange inputs\
    and system calls.

    Properties
    ----------
    batch : int, Protected
        Determines the batch size.\
        Read more about protected properties in constant-properties-protector package documentation.
    dt : torch.Tensor, Protected
        Time step in milliseconds.\
        Read more about protected properties in constant-properties-protector package documentation.
    neurons : Dict[str, Soma]
        Dictionary containing inserted somas.\
        The keys in this dictionary are the names of the corresponding somas.
    synapses : Dict[str, Soma]
        Dictionary containing inserted synapses.\
        The keys in this dictionary are the names of the corresponding synapses.
    global_plasticity : bool
        Indicates the coordination of enabality of synaptic plasticity in all members of the network.\
        If it be true, plasticity signal will be propagate from this modul to every dendrite of every inserted neuron.
    global_myelination : bool
        Indicates the coordination of enabality of myelination in all members of the network.\
        If it be true, myelination signal will be propagate from this modul to every axon of every inserted neuron.
    plasticity : bool
        Plasticity signal of the network.
    plasticity : bool
        Myelination signal of the network.

    Arguments
    ---------
    batch : int, Optional, default: 1
        Determines the batch size.
    dt : float or torch.Tensor, Optional, default: 1.
        Time step in milliseconds.
    global_plasticity : bool, Optional, default: True
        Indicates the coordination of enabality of synaptic plasticity in all members of the network.\
        If it be true, plasticity signal will be propagate from this modul to every dendrite of every inserted neuron.
    global_myelination : bool, Optional, default: True
        Indicates the coordination of enabality of myelination in all members of the network.\
        If it be true, myelination signal will be propagate from this modul to every axon of every inserted neuron.
    """
    def __init__(
        self,
        batch: int = 1,
        dt: Union[float, torch.Tensor] = 1.,
        global_plasticity: bool = True,
        global_myelination: bool = True,
    ) -> None:
        super().__init__()
        CPP.protect(self, 'batch')
        CPP.protect(self, 'dt')
        self._batch = batch
        self.register_buffer("_dt", torch.as_tensor(dt))
        self.neurons = {}
        self.synapses = {}
        self.global_plasticity = global_plasticity
        self.global_myelination = global_myelination
        self.plasticity = global_plasticity
        self.myelination = global_myelination


    @property
    def plasticity(
        self
    ) -> bool:
        """
        Returns plasticity signal of the network.

        Returns
        -------
        plasticity: bool
            The plasticity signal of the network.
        
        """
        return self._plasticity

    @plasticity.setter
    def plasticity(
        self,
        value: bool
    ) -> None:
        """
        Setter for plasticity signal of the network.
        If global_plasticity be true, plasticity signal will be propagate from this modul\
        to every dendrite of every inserted neuron.

        Returns
        -------
        None

        """
        self._plasticity = value
        if self.global_plasticity:
            for neuron in self.neurons.values():
                for dendrite in neuron.dendrites.values():
                    dendrite.plasticity = value

    @property
    def myelination(
        self
    ) -> bool:
        """
        Returns myelination signal of the network.

        Returns
        -------
        myelination: bool
            The myelination signal of the network.
        
        """
        return self._myelination

    @myelination.setter
    def myelination(
        self,
        value: bool
    ) -> None:
        """
        Setter for myelination signal of the network.
        If global_myelination be true, myelination signal will be propagate from this modul\
        to every axon of every inserted neuron.

        Returns
        -------
        None

        """
        self._myelination = value
        if self.global_myelination:
            for neuron in self.neurons.values():
                for axon in neuron.axons.values():
                    axon.myelination = value


    def insert(
        self,
        organ: Union[Soma, Synapse],
    ) -> Network:
        """
        Inserts an organ into the network.
        
        Arguments
        ---------
        organ : Soma or Synapse
            Inserting organ.
        
        Returns
        -------
        self: Network
            With the aim of making chains possible.
        
        """
        name = organ.name if organ.is_constructed or issubclass(type(organ), Soma) else \
               organ.requirement_value('name')
        if name is None:
            raise Exception(f"An unnamed organ ({organ}) can not be added to the network.")
        if name in self.neurons.keys():
            raise Exception(f"The network is already using a neuron named {name}.")
        if name in self.synapses.keys():
            raise Exception(f"The network is already using a synapse named {name}.")
        if not organ.is_constructed:
            organ.meet_requirement(dt=self.dt)
        if not organ.is_constructed:
            organ.meet_requirement(batch=self.batch)
        organs = [self.neurons, self.synapses][issubclass(type(organ), Synapse)]
        organs[name] = organ
        self.add_module(name, organ)
        return self


    def progress(
        self,
        external_inputs: Dict[str, Dict[str, Any]] = {},
    ) -> None:
        """
        Simulate the network activity for a single step.
        
        Returns
        -------
        None
        
        """
        for name,synapse in self.synapses.items():
            inputs = external_inputs.get(name, {})
            synapse.forward(**inputs)
        
        for name,neuron in self.neurons.items():
            inputs = external_inputs.get(name, {})
            neuron.progress(**inputs)


    def reset(
        self
    ) -> None:
        """
        Refractor and reset the somas and connected organs.
        
        Returns
        -------
        None
        
        """
        for synapse in self.synapses.values():
            synapse.reset()
        for neuron in self.neurons.values():
            neuron.reset()


    def __getitem__(
        self,
        name: str
    ) -> Union[Soma, Synapse]:
        """
        Returns the inserted organ (soma or synapse) associated with the given name.
        
        Arguments
        ---------
        name : str
            The name of asked organ.
        
        Returns
        -------
        organ : Soma or Synapse
            The asked organ.
        
        """
        if name in self.neurons.keys():
            return self.neurons[name]
        if name in self.synapses.keys():
            return self.synapses[name]
        raise Exception(f"The network has no neuron or synapse named {name}")