"""
Interneuron spiking soma is a type of spiking soma that is used to transmit information between neurons.
"""


from ..spiking_soma import SpikingSoma
from abc import ABC




class InterneuronSpikingSoma(SpikingSoma, ABC):
    """
    Basic class for all types of interneuron spiking soma.

    This module is empty and was created for the sole purpose of creating categories in types.
    """
    pass