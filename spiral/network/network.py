"""
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
        self._plasticity = global_plasticity
        self._myelination = global_myelination


    @property
    def plasticity(
        self
    ) -> bool:
        return self._plasticity
    @plasticity.setter
    def plasticity(
        self,
        value: bool
    ) -> None:
        self._plasticity = value
        if self.global_plasticity:
            for neuron in self.neurons.values():
                for dendrite in neuron.dendrites.values():
                    dendrite.plasticity = value

    @property
    def myelination(
        self
    ) -> bool:
        return self._myelination
    @myelination.setter
    def myelination(
        self,
        value: bool
    ) -> None:
        self._myelination = value
        if self.global_myelination:
            for neuron in self.neurons.values():
                for axon in neuron.axons.values():
                    axon.myelination = value


    def insert(
        self,
        organ: Union[Soma, Synapse],
    ) -> Network:
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
        for name,synapse in self.synapses.items():
            inputs = external_inputs.get(name, {})
            synapse.forward(**inputs)
        
        for name,neuron in self.neurons.items():
            inputs = external_inputs.get(name, {})
            neuron.progress(**inputs)


    def reset(
        self
    ) -> None:
        for synapse in self.synapses.values():
            synapse.reset()
        for neuron in self.neurons.values():
            neuron.reset()


    def __getitem__(
        self,
        name: str
    ) -> Union[Soma, Synapse]:
        if name in self.neurons.keys():
            return self.neurons[name]
        if name in self.synapses.keys():
            return self.synapses[name]
        raise Exception(f"The network has no neuron or synapse named {name}")