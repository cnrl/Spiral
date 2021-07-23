"""
"""


from typing import Union, Iterable
from typeguard import typechecked
from spiral.network.network import Network
from spiral.soma.soma import Soma
from spiral.axon.axon import Axon
from spiral.dendrite.dendrite import Dendrite
from spiral.synapse.synapse import Synapse




class CONSIDERED:
    NETWORK = None


class ConsiderationInfix:
    def __init__(self, function):
        self.function = function
    def __or__(self, other):
        return self.function(other)


@typechecked
def __consider__(
    network: Network,
) -> None:
    if CONSIDERED.NETWORK is None:
        CONSIDERED.NETWORK = network
    else:
        raise Exception(f"The previously considered network {CONSIDERED.NETWORK} has not yet been checked out.")


@typechecked
def __insert__(
    organ: Union[Soma, Synapse]
) -> None:
    if CONSIDERED.NETWORK is not None:
        CONSIDERED.NETWORK.insert(organ)
    else:
        raise Exception(f"No network is considered.")


@typechecked
def __checkout__(
    network: Network,
) -> Network:
    if CONSIDERED.NETWORK is None:
        raise Exception(f"No network is considered.")
    if CONSIDERED.NETWORK==network:
        CONSIDERED.NETWORK = None
        return network
    else:
        raise Exception(f"Network {network} is not considered. Rather, Network {CONSIDERED.NETWORK} is considered.")


class Infix:
    def __init__(self, function):
        self.function = function
    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __or__(self, other):
        return self.function(other)


@typechecked
def __using__(
    soma: Soma,
    organ: Union[Axon, Dendrite],
) -> Soma:
    return soma.use(organ)


@typechecked
def __of__(
    organ: Union[Axon, Dendrite],
    soma: Soma,
) -> Union[Axon, Dendrite]:
    soma.use(organ)
    return organ


@typechecked
def __from__(
    synapse: Synapse,
    axon: Axon,
) -> Synapse:
    return synapse.connect(axon)


@typechecked
def __to__(
    synapse: Synapse,
    dendrite: Dendrite,
) -> Synapse:
    return synapse.connect(dendrite)


@typechecked
def __following__(
    synapse: Synapse,
    axon: Axon
) -> Synapse:
    return synapse.follow(axon)


CONSIDER = ConsiderationInfix(__consider__)
INSERT = ConsiderationInfix(__insert__)
CHECKOUT = ConsiderationInfix(__checkout__)
USING = Infix(__using__)
OF = Infix(__of__)
FROM = Infix(__from__)
TO = Infix(__to__)
FOLLOWING = Infix(__following__)