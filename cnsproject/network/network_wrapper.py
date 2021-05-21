from .network import Network
from .neural_populations import AbstractNeuralPopulation
from .synapse_sets import AbstractSynapseSet
from ..learning.learning_rule_enforcers import AbstractLearningRuleEnforcer
from .axon_sets import AbstractAxonSet
from .dendrite_sets import AbstractDendriteSet
from typing import Union


class Infix:
    def __init__(self, function):
        self.function = function
    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __or__(self, other):
        return self.function(other)


def __using__(
    population: AbstractNeuralPopulation,
    obj: Union[AbstractAxonSet, AbstractDendriteSet],
) -> AbstractNeuralPopulation:
    population.use(obj)
    return population


def __of__(
    obj: Union[AbstractAxonSet, AbstractDendriteSet],
    population: AbstractNeuralPopulation,
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


USING = Infix(__using__)
OF = Infix(__of__)
FROM = Infix(__from__)
TO = Infix(__to__)
FOLLOWING = Infix(__following__)