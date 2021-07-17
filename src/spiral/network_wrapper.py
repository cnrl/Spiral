from .network import Network
from .neural_populations import AbstractNeuralPopulation
from .neuromodulatory_tissues import AbstractNeuromodulatoryTissue
from .synapse_sets import AbstractSynapseSet
from ..learning.learning_rule_enforcers import AbstractLearningRuleEnforcer
from .axon_sets import AbstractAxonSet
from .dendrite_sets import AbstractDendriteSet
from typing import Union, Iterable


class Infix:
    def __init__(self, function):
        self.function = function
    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __or__(self, other):
        return self.function(other)


def __using__(
    population: Union[AbstractNeuralPopulation, AbstractNeuromodulatoryTissue],
    obj: Union[AbstractAxonSet, AbstractDendriteSet, Iterable],
) -> AbstractNeuralPopulation:
    population.use(obj)
    return population


def __of__(
    obj: Union[AbstractAxonSet, AbstractDendriteSet],
    population: Union[AbstractNeuralPopulation, AbstractNeuromodulatoryTissue],
) -> Union[AbstractAxonSet, AbstractDendriteSet]:
    population.use(obj)
    return obj


def __to__(
    synapse: AbstractSynapseSet,
    dendrite: AbstractDendriteSet,
) -> AbstractNeuralPopulation:
    synapse.set_dendrite(dendrite)
    return synapse


def __from__(
    synapse: AbstractSynapseSet,
    axon: AbstractAxonSet,
) -> AbstractNeuralPopulation:
    synapse.set_axon(axon)
    return synapse


def __following__(
    synapse: AbstractSynapseSet,
    lr: AbstractLearningRuleEnforcer
) -> AbstractSynapseSet:
    lr.set_synapse(synapse)
    return synapse,lr


def __affected_by__(
    lr: AbstractLearningRuleEnforcer,
    axon: Union[AbstractAxonSet, Iterable]
) -> AbstractLearningRuleEnforcer:
    lr.add_axon(axon)
    return lr



USING = Infix(__using__)
OF = Infix(__of__)
FROM = Infix(__from__)
TO = Infix(__to__)
FOLLOWING = Infix(__following__)
AFFECTED_BY = Infix(__affected_by__)